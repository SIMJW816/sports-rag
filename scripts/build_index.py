"""
scripts/build_index.py

Chroma 벡터 인덱스와 BM25 인덱스를 구축하는 멱등성 스크립트.

실행 단계:
  1. 모든 문서 로드 (CSVLoader + TextLoader)
  2. 스플리터 비교 실험 실행 및 보고서 출력
  3. 운영용 청크 생성
  4. Chroma 인덱스 구축 (이미 존재하면 건너뜀 — 임베딩 비용 절감)
  5. BM25 리트리버 구축 후 pickle 저장

비용 절감 포인트:
  - Chroma 인덱스가 이미 존재하면 임베딩 API 호출 없이 로드만 합니다.
  - --skip-splitter-compare: SemanticChunker API 호출 생략 (약 $0.01 절감)
  - --force-rebuild: 기존 인덱스를 삭제하고 재구축 (명시적 요청 시만)

사용법:
  python scripts/build_index.py
  python scripts/build_index.py --force-rebuild
  python scripts/build_index.py --skip-splitter-compare
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Sports RAG용 Chroma 및 BM25 인덱스를 구축합니다."
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="기존 인덱스를 삭제하고 재구축합니다.",
    )
    parser.add_argument(
        "--skip-splitter-compare",
        action="store_true",
        help="SemanticChunker 비교를 건너뜁니다 (OpenAI API 비용 절감).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # ── 1단계: 문서 로드 ───────────────────────────────────────────────────
    logger.info("1/5단계: 문서 로드 중...")
    from ingestion.loaders import load_all_documents, print_loader_stats

    docs = load_all_documents()
    print_loader_stats(docs)

    if len(docs) < 10:
        logger.error("문서가 너무 적습니다 (%d개). data/raw/ 디렉토리를 확인하세요.", len(docs))
        sys.exit(1)

    # ── 2단계: 스플리터 비교 ───────────────────────────────────────────────
    if not args.skip_splitter_compare:
        logger.info("2/5단계: 스플리터 비교 실험 중...")
        from ingestion.splitters import compare_splitters

        article_docs = [d for d in docs if d.metadata.get("doc_type") == "article"]
        if article_docs:
            compare_splitters(article_docs)
        else:
            logger.warning("아티클 문서 없음 — 스플리터 비교 건너뜀.")
    else:
        logger.info("2/5단계: 스플리터 비교 건너뜀 (--skip-splitter-compare).")

    # ── 3단계: 운영 청크 생성 ──────────────────────────────────────────────
    logger.info("3/5단계: 운영 청크 생성 중...")
    from ingestion.splitters import get_production_chunks

    chunks = get_production_chunks(docs)
    logger.info("총 청크 수: %d개", len(chunks))

    if len(chunks) < 150:
        logger.warning(
            "청크 수 %d개 — 목표 150개 미달. 데이터 파일을 추가하는 것을 권장합니다.",
            len(chunks),
        )
    else:
        logger.info("✓ 청크 수 요건 충족: %d개 ≥ 150개", len(chunks))

    # 임베딩 비용 추정 출력
    est_tokens = sum(len(c.page_content.split()) * 1.3 for c in chunks)
    est_cost   = est_tokens / 1_000_000 * 0.02  # text-embedding-3-small 단가
    print(f"\n  임베딩 비용 추정:")
    print(f"    모델     : text-embedding-3-small")
    print(f"    예상 토큰 : {int(est_tokens):,}")
    print(f"    예상 비용 : ${est_cost:.4f}")
    print(f"    ※ 인덱스가 이미 존재하면 비용 발생 없음\n")

    # ── 4단계: Chroma 인덱스 구축 ──────────────────────────────────────────
    logger.info("4/5단계: Chroma 벡터 인덱스 구축 중...")
    from retrieval.dense import build_or_load_vectorstore, CHROMA_PERSIST_DIR

    if args.force_rebuild:
        import shutil
        chroma_path = Path(CHROMA_PERSIST_DIR)
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
            logger.info("기존 Chroma 인덱스 삭제 완료.")

    vectorstore = build_or_load_vectorstore(chunks)
    count       = vectorstore._collection.count()
    logger.info("✓ Chroma 인덱스 준비 완료: %d개 벡터", count)

    # ── 5단계: BM25 인덱스 구축 ────────────────────────────────────────────
    logger.info("5/5단계: BM25 리트리버 구축 중...")
    from retrieval.sparse import (
        build_bm25_retriever, save_bm25_retriever, BM25_PICKLE_PATH
    )

    if args.force_rebuild or not BM25_PICKLE_PATH.exists():
        bm25 = build_bm25_retriever(chunks, k=5)
        save_bm25_retriever(bm25)
        logger.info("✓ BM25 인덱스 저장 완료: %s", BM25_PICKLE_PATH)
    else:
        logger.info("BM25 인덱스 이미 존재 → 건너뜀: %s", BM25_PICKLE_PATH)

    # ── 최종 요약 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'인덱스 구축 완료':^55}")
    print("=" * 55)
    print(f"  로드된 문서 수  : {len(docs)}개")
    print(f"  생성된 청크 수  : {len(chunks)}개")
    print(f"  Chroma 벡터 수  : {count}개")
    print(f"  BM25 인덱스     : {BM25_PICKLE_PATH}")
    print(f"  임베딩 모델     : text-embedding-3-small")
    print("=" * 55)
    print("\n  다음 단계: python scripts/build_graph.py\n")


if __name__ == "__main__":
    main()
