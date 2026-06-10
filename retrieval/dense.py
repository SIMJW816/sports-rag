"""
retrieval/dense.py

Dense(밀집) 검색 모듈 — Chroma 벡터 스토어 + OpenAI 임베딩 사용.

비용 절감 설계:
  - build_or_load_vectorstore: chroma.sqlite3 파일이 존재하면
    새 임베딩을 생성하지 않고 기존 인덱스를 그대로 로드합니다.
    → 재실행 시 임베딩 API 비용 0원
  - 임베딩 모델: text-embedding-3-small ($0.02/M tokens, 가장 경제적)

제공 기능:
  - 유사도(Similarity) 검색 리트리버
  - MMR(Maximal Marginal Relevance) 리트리버 (다양성 제어)
  - Similarity vs MMR 결과 비교 시각화
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
logger = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / "chroma_db")
EMBEDDING_MODEL    = "text-embedding-3-small"  # 경제적·고성능 임베딩 모델
COLLECTION_NAME    = "sports_rag"


def _get_embeddings() -> OpenAIEmbeddings:
    """
    OpenAI 임베딩 인스턴스를 반환합니다.

    반환값:
        text-embedding-3-small 모델의 OpenAIEmbeddings 인스턴스
    """
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_or_load_vectorstore(docs: List[Document]) -> Chroma:
    """
    Chroma 인덱스를 새로 구축하거나 기존 인덱스를 로드합니다 (멱등성 보장).

    chroma.sqlite3 파일이 있고 문서가 1개 이상이면 기존 인덱스를 로드합니다.
    새 임베딩을 생성하지 않으므로 반복 실행 시 API 비용이 발생하지 않습니다.

    매개변수:
        docs: 새로 구축할 경우 인덱싱할 Document 리스트

    반환값:
        로드 또는 새로 구축된 Chroma 인스턴스
    """
    embeddings   = _get_embeddings()
    persist_path = Path(CHROMA_PERSIST_DIR)

    # chroma.sqlite3 존재 여부로 기존 인덱스 확인
    chroma_db_file = persist_path / "chroma.sqlite3"
    if chroma_db_file.exists():
        logger.info("기존 Chroma 인덱스 발견 (%s) — 로드 중...", CHROMA_PERSIST_DIR)
        vs    = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        count = vs._collection.count()
        if count > 0:
            logger.info("Chroma 인덱스 로드 완료: %d개 벡터 (임베딩 비용 절감)", count)
            return vs
        logger.info("기존 인덱스가 비어 있음 — 재구축합니다.")

    logger.info("새 Chroma 인덱스 구축 중 (%d개 문서) ...", len(docs))
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    logger.info("Chroma 인덱스 구축 완료: %d개 문서 인덱싱", len(docs))
    return vs


def get_similarity_retriever(vectorstore: Chroma, k: int = 5) -> VectorStoreRetriever:
    """
    코사인 유사도 기반 리트리버를 생성합니다.

    매개변수:
        vectorstore: 초기화된 Chroma 인스턴스
        k: 반환할 문서 수

    반환값:
        similarity 검색 타입의 VectorStoreRetriever
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def get_mmr_retriever(
    vectorstore: Chroma,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> VectorStoreRetriever:
    """
    MMR(Maximal Marginal Relevance) 리트리버를 생성합니다.

    MMR은 관련성과 다양성을 균형 있게 고려합니다.
    먼저 fetch_k개를 유사도로 후보 선정 후, 이미 선택된 문서와 중복도가
    낮은 것을 우선 선택하여 k개를 반환합니다.

    lambda_mult 값의 의미:
      1.0 → 순수 유사도 (Similarity와 동일)
      0.5 → 균형 (기본값 권장)
      0.0 → 순수 다양성 (쿼리 관련성 무시)

    매개변수:
        vectorstore : 초기화된 Chroma 인스턴스
        k           : 최종 반환 문서 수
        fetch_k     : MMR 전 후보 풀 크기
        lambda_mult : 관련성-다양성 트레이드오프 (0.0~1.0)

    반환값:
        MMR 검색 타입의 VectorStoreRetriever
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           k,
            "fetch_k":     fetch_k,
            "lambda_mult": lambda_mult,
        },
    )


def compare_similarity_vs_mmr(vectorstore: Chroma, query: str) -> Dict[str, Any]:
    """
    동일 쿼리에 대해 유사도 검색과 MMR(lambda 3종)을 비교합니다.

    MMR이 중복 청크를 어떻게 줄이는지 보여줍니다.
    lambda_mult = 0.9, 0.5, 0.1 세 가지 설정으로 다양성 변화를 실험합니다.

    매개변수:
        vectorstore: 초기화된 Chroma 인스턴스
        query      : 검색 쿼리 문자열

    반환값:
        {검색_방식: [{snippet, source}, ...]} 형태의 결과 딕셔너리
    """
    results: Dict[str, Any] = {}

    # 기준: 순수 유사도 검색
    sim_ret  = get_similarity_retriever(vectorstore, k=5)
    sim_docs = sim_ret.invoke(query)
    results["similarity"] = [
        {"snippet": d.page_content[:120].replace("\n", " "),
         "source":  d.metadata.get("entity_name", "?")}
        for d in sim_docs
    ]

    # MMR: lambda = 0.9 / 0.5 / 0.1 비교
    for lm in [0.9, 0.5, 0.1]:
        ret  = get_mmr_retriever(vectorstore, k=5, fetch_k=20, lambda_mult=lm)
        docs = ret.invoke(query)
        results[f"mmr_lambda={lm}"] = [
            {"snippet": d.page_content[:120].replace("\n", " "),
             "source":  d.metadata.get("entity_name", "?")}
            for d in docs
        ]

    # 결과 출력
    print("\n" + "=" * 70)
    print(f"  유사도 vs MMR 비교")
    print(f"  쿼리: '{query}'")
    print("=" * 70)

    for label, items in results.items():
        sources    = [it["source"] for it in items]
        unique_cnt = len(set(sources))
        print(f"\n  [{label}]  고유 소스={unique_cnt}/{len(sources)}")
        for i, it in enumerate(items, 1):
            print(f"    {i}. [{it['source']}] {it['snippet']}...")

    # 중복도 분석
    sim_redundancy = len(sim_docs) - len({d.page_content[:80] for d in sim_docs})
    mmr_docs       = get_mmr_retriever(vectorstore, k=5, lambda_mult=0.5).invoke(query)
    mmr_redundancy = len(mmr_docs) - len({d.page_content[:80] for d in mmr_docs})

    print(f"\n  중복도 분석:")
    print(f"    유사도 검색   — 중복 소스: {sim_redundancy}개")
    print(f"    MMR λ=0.5    — 중복 소스: {mmr_redundancy}개")
    print(f"    → MMR이 중복을 {max(0, sim_redundancy - mmr_redundancy)}개 줄였습니다.")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(levelname)s | %(name)s | %(message)s")

    from ingestion.loaders import load_all_documents
    from ingestion.splitters import get_production_chunks

    docs   = load_all_documents()
    chunks = get_production_chunks(docs)
    vs     = build_or_load_vectorstore(chunks)

    compare_similarity_vs_mmr(vs, "Ronaldo Champions League goals")
    compare_similarity_vs_mmr(vs, "Messi Barcelona career trophies")
