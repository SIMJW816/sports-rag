"""
scripts/run_comparison.py

세 RAG 시스템을 8개 대표 쿼리로 비교하는 메인 실행 스크립트.

기능:
  - 세 가지 RAG 체인을 모두 로드하여 동시 비교
  - 결과를 표 형식으로 출력하고 CSV로 저장
  - --ragas 옵션: Ragas 4대 지표 평가까지 함께 실행
  - --dry-run: 개발 단계 비용 절감 (5개 질문만 평가)

비용 가이드:
  비교 실행(8개 쿼리 × 3시스템)만: 약 $0.05 (gpt-4.1-mini)
  + Ragas dry-run(5개 질문):      약 $0.01 (gpt-4o-mini 기본)
  + Ragas 전체(30개 질문, gpt-4o): 약 $1.44

사용법:
  python scripts/run_comparison.py
  python scripts/run_comparison.py --no-graph
  python scripts/run_comparison.py --ragas --dry-run
  python scripts/run_comparison.py --ragas --judge-model gpt-4o
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
        description="세 가지 RAG 시스템을 8개 쿼리로 비교합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 비교만 실행 (Graph RAG 제외)
  python scripts/run_comparison.py --no-graph

  # 비교 + Ragas 드라이런 (저렴)
  python scripts/run_comparison.py --ragas --dry-run

  # 비교 + Ragas 전체 (최종 제출용)
  python scripts/run_comparison.py --ragas --judge-model gpt-4o
        """,
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Graph RAG 생략 (Neo4j 미사용 시).",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="비교 후 Ragas 4대 지표 평가도 실행합니다.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="--ragas 사용 시 처음 5개 질문만 평가합니다.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        help="Ragas 평가 Judge LLM 모델 (기본: gpt-4o-mini).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # ── 공유 리소스 로드 ───────────────────────────────────────────────────
    logger.info("문서 로드 및 리트리버 구성 중...")
    from ingestion.loaders import load_all_documents
    from ingestion.splitters import get_production_chunks
    from retrieval.dense import build_or_load_vectorstore
    from retrieval.sparse import build_bm25_retriever, load_bm25_retriever, BM25_PICKLE_PATH
    from retrieval.hybrid import build_hybrid_retriever
    from chains.rag_chains import VectorRAGChain, HybridRAGChain

    docs   = load_all_documents()
    chunks = get_production_chunks(docs)
    vs     = build_or_load_vectorstore(chunks)

    # BM25 로드 (없으면 구축)
    try:
        bm25 = load_bm25_retriever()
    except FileNotFoundError:
        logger.info("BM25 인덱스 없음 — 새로 구축합니다.")
        from retrieval.sparse import save_bm25_retriever
        bm25 = build_bm25_retriever(chunks)
        save_bm25_retriever(bm25)

    ensemble     = build_hybrid_retriever(vs, bm25)
    vector_chain = VectorRAGChain(vs)
    hybrid_chain = HybridRAGChain(ensemble)

    # Graph RAG (선택적)
    graph_chain = None
    if not args.no_graph:
        try:
            from graph.build_graph import get_neo4j_graph
            from chains.rag_chains import GraphRAGChain
            neo4j_graph = get_neo4j_graph()
            graph_chain = GraphRAGChain(neo4j_graph)
            logger.info("✓ Graph RAG 로드 완료.")
        except Exception as e:
            logger.warning(
                "Graph RAG 불가 (%s). --no-graph 옵션으로 이 경고를 숨길 수 있습니다.", e
            )

    # ── 시스템 비교 실행 ───────────────────────────────────────────────────
    from evaluation.compare import run_full_comparison
    run_full_comparison(vector_chain, hybrid_chain, graph_chain)

    # ── Ragas 평가 (선택적) ────────────────────────────────────────────────
    if args.ragas:
        from evaluation.ragas_eval import (
            load_testset, evaluate_rag_system, _estimate_cost
        )

        testset = load_testset()
        if args.dry_run:
            testset = testset[:5]
            print(f"\n  [DRY-RUN] {len(testset)}개 질문으로 Ragas 평가를 실행합니다.")

        _estimate_cost(len(testset), args.judge_model)

        evaluate_rag_system(
            vector_chain, testset, "VectorRAG", judge_model=args.judge_model
        )
        evaluate_rag_system(
            hybrid_chain, testset, "HybridRAG", judge_model=args.judge_model
        )

    print("\n  모든 작업 완료! evaluation/results/ 디렉토리에서 결과를 확인하세요.\n")


if __name__ == "__main__":
    main()
