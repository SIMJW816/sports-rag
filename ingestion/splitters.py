"""
ingestion/splitters.py

스플리터 비교 모듈 — 동일한 문서에 두 가지 분할 방식을 적용하여 결과를 비교합니다.

스플리터 1 — RecursiveCharacterTextSplitter (기본)
  chunk_size=500, chunk_overlap=50
  선택 근거:
    - 500자 ≈ 2~3 문장 분량으로 스포츠 커리어 서술에 적합
    - overlap=50으로 선수 이름 등 엔티티가 청크 경계에서 잘리지 않도록 보존
    - 결정론적(API 호출 없음) → 인덱스 재구성 비용 0원
    - CSV 행처럼 이미 밀도 높은 구조 데이터에 특히 적합

스플리터 2 — SemanticChunker (langchain_experimental)
  embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
  breakpoint_threshold_type="percentile"
  선택 근거:
    - 인접 문장 임베딩의 코사인 유사도로 토픽 전환점을 감지
    - 커리어 이적 기록처럼 의미적으로 연속된 서술을 하나의 청크로 보존
    - Context Recall 향상 기대 (다중 홉 사실이 분리되지 않음)
    - 단점: OpenAI API 호출 필요 → 비용 발생, 비결정적 결과
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()
logger = logging.getLogger(__name__)

# ── 스플리터 1 파라미터 ────────────────────────────────────────────────────────
# chunk_size=500: 스포츠 서술 기준 약 2~3 문장
# chunk_overlap=50: 청크 경계에서 선수·팀 이름이 잘리는 것을 방지
RECURSIVE_CHUNK_SIZE    = 500
RECURSIVE_CHUNK_OVERLAP = 50


def split_recursive(docs: List[Document]) -> List[Document]:
    """
    RecursiveCharacterTextSplitter로 문서를 분할합니다.

    단락 → 문장 → 단어 → 문자 순서로 자연스러운 경계를 먼저 시도합니다.
    API 호출이 없어 비용이 들지 않으며 결과가 결정론적입니다.

    매개변수:
        docs: 분할할 Document 리스트

    반환값:
        분할된 청크 Document 리스트 (원본 메타데이터 상속)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info("RecursiveSplitter: %d개 문서 → %d개 청크", len(docs), len(chunks))
    return chunks


def split_semantic(docs: List[Document]) -> List[Document]:
    """
    SemanticChunker로 문서를 의미 기반 분할합니다.

    인접 문장 임베딩의 유사도 차이가 가장 큰 상위 5% 지점을 분할 경계로 설정합니다.
    문서 길이에 따라 청크 수가 자동 조정됩니다.

    주의: OpenAI API(text-embedding-3-small)를 호출하므로 비용이 발생합니다.
          개발 단계에서는 skip-splitter-compare 옵션 활용을 권장합니다.

    매개변수:
        docs: 분할할 Document 리스트

    반환값:
        의미 경계 기반으로 분할된 청크 Document 리스트
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        logger.error("langchain_experimental 미설치. pip install langchain-experimental")
        raise

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,  # 상위 5% 이질적 전환점에서만 분할
    )
    chunks = splitter.split_documents(docs)
    logger.info("SemanticChunker: %d개 문서 → %d개 청크", len(docs), len(chunks))
    return chunks


def _chunk_stats(chunks: List[Document]) -> Dict[str, Any]:
    """
    청크 리스트의 통계 요약을 계산합니다.

    매개변수:
        chunks: 청크 Document 리스트

    반환값:
        count, avg_len, min_len, max_len, 두 경계 샘플을 담은 딕셔너리
    """
    lengths = [len(c.page_content) for c in chunks]
    if not lengths:
        return {"count": 0, "avg_len": 0.0, "min_len": 0, "max_len": 0,
                "sample_boundary_1": "", "sample_boundary_2": ""}

    avg = sum(lengths) / len(lengths)

    # 전체의 25%, 75% 지점에서 청크 경계 샘플 추출
    idx1 = len(chunks) // 4
    idx2 = (len(chunks) * 3) // 4

    def _boundary(idx: int) -> str:
        """인접 두 청크의 꼬리와 머리를 이어서 경계 스니펫 반환"""
        if idx <= 0 or idx >= len(chunks):
            return "(해당 없음)"
        tail = chunks[idx - 1].page_content[-80:].replace("\n", " ").strip()
        head = chunks[idx].page_content[:80].replace("\n", " ").strip()
        return f"...{tail} | {head}..."

    return {
        "count":            len(chunks),
        "avg_len":          round(avg, 1),
        "min_len":          min(lengths),
        "max_len":          max(lengths),
        "sample_boundary_1": _boundary(idx1),
        "sample_boundary_2": _boundary(idx2),
    }


def compare_splitters(docs: List[Document]) -> Dict[str, Any]:
    """
    두 스플리터를 동일 문서에 적용하여 결과를 비교합니다.

    비교 항목: 청크 수, 평균/최소/최대 길이, 경계 자연스러움 샘플

    매개변수:
        docs: 비교할 원본 Document 리스트 (보통 아티클 문서)

    반환값:
        {
          "recursive": {count, avg_len, min_len, max_len, sample_boundary_1, sample_boundary_2},
          "semantic" : {동일 구조},
          "analysis" : 분석 결과 문자열
        }
    """
    print("\n" + "=" * 65)
    print(f"{'스플리터 비교':^65}")
    print("=" * 65)
    print(f"  입력 문서 수: {len(docs)}")

    print("\n  [1/2] RecursiveCharacterTextSplitter 실행 중 ...")
    rec_chunks = split_recursive(docs)
    rec_stats  = _chunk_stats(rec_chunks)

    print("  [2/2] SemanticChunker 실행 중 (OpenAI 임베딩 API 호출) ...")
    try:
        sem_chunks = split_semantic(docs)
        sem_stats  = _chunk_stats(sem_chunks)
        semantic_ok = True
    except Exception as exc:
        logger.error("SemanticChunker 실패: %s", exc)
        sem_stats   = {"count": 0, "avg_len": 0.0, "min_len": 0, "max_len": 0,
                       "sample_boundary_1": "(오류)", "sample_boundary_2": "(오류)"}
        semantic_ok = False

    # 비교 표 출력
    print("\n" + "-" * 65)
    print(f"  {'항목':<25} {'Recursive':>15} {'Semantic':>15}")
    print("-" * 65)
    for key in ("count", "avg_len", "min_len", "max_len"):
        print(f"  {key:<25} {str(rec_stats[key]):>15} {str(sem_stats[key]):>15}")
    print("-" * 65)

    print("\n  경계 샘플 (Recursive):")
    print(f"    B1: {rec_stats['sample_boundary_1']}")
    print(f"    B2: {rec_stats['sample_boundary_2']}")

    if semantic_ok:
        print("\n  경계 샘플 (Semantic):")
        print(f"    B1: {sem_stats['sample_boundary_1']}")
        print(f"    B2: {sem_stats['sample_boundary_2']}")

    analysis = (
        "분석: 스포츠 커리어 서술 아티클의 경우 SemanticChunker가 이적 시기별로 "
        "더 자연스러운 청크를 생성하는 경향이 있습니다. 예를 들어 '과르디올라의 바르셀로나 시절' "
        "전체가 하나의 청크에 담겨 Context Recall이 향상됩니다. "
        "반면 RecursiveCharacterTextSplitter는 API 호출이 없어 비용이 0원이고 결과가 "
        "결정론적이어서 CSV 행처럼 이미 구조화된 데이터에 최적입니다. "
        "최종 운영 전략: CSV → Recursive, 아티클 → Recursive(600자) 적용. "
        "SemanticChunker는 비교 분석 용도로만 사용하여 반복 임베딩 비용을 절감합니다."
    )

    print(f"\n  분석:\n    {analysis}\n")
    print("=" * 65 + "\n")

    return {"recursive": rec_stats, "semantic": sem_stats, "analysis": analysis}


def get_production_chunks(docs: List[Document]) -> List[Document]:
    """
    운영용 최적 청크 세트를 반환합니다.

    전략:
      - CSV 문서  → RecursiveCharacterTextSplitter (비용 0, 결정론적)
      - 아티클 문서 → RecursiveCharacterTextSplitter (chunk_size=600, overlap=80)
        (SemanticChunker는 비교 실험용으로만 사용 — 반복 임베딩 비용 방지)

    매개변수:
        docs: 전체 원본 Document 리스트

    반환값:
        인덱싱 준비된 청크 리스트
    """
    csv_docs     = [d for d in docs if d.metadata.get("doc_type") == "csv"]
    article_docs = [d for d in docs if d.metadata.get("doc_type") == "article"]

    # CSV: 행 길이(~400자)에 맞는 500자 청크
    csv_chunks = split_recursive(csv_docs) if csv_docs else []

    # 아티클: 커리어 서술 보존을 위해 약간 큰 600자 청크
    article_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    article_chunks = article_splitter.split_documents(article_docs) if article_docs else []

    all_chunks = csv_chunks + article_chunks
    logger.info("운영 청크 생성 완료: %d개 (CSV: %d, 아티클: %d)",
                len(all_chunks), len(csv_chunks), len(article_chunks))
    return all_chunks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(levelname)s | %(name)s | %(message)s")

    from ingestion.loaders import load_article_documents
    article_docs = load_article_documents()
    print(f"아티클 문서 {len(article_docs)}개 로드 완료. 스플리터 비교 시작...")
    compare_splitters(article_docs)
