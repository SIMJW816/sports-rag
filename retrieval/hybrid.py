"""
retrieval/hybrid.py

하이브리드 검색 모듈 — Dense(Chroma) + Sparse(BM25)를
EnsembleRetriever로 결합합니다.

가중치 설정: dense=0.6, sparse=0.4
  선택 근거:
    - 서술형 위키 아티클은 의미 기반 Dense 검색이 더 효과적
      ("elegant playmaker"로 Xavi를 찾으려면 의미 이해 필요)
    - 선수명·연도·대회명처럼 정확한 키워드가 있는 쿼리에서는
      BM25가 누락 없이 직접 매칭
    - 0.4 BM25 가중치로 "Cristiano Ronaldo 2009" 같은 정확 키워드도 보장

결합 방식: Reciprocal Rank Fusion (RRF)
  두 리트리버의 순위를 역수 합산하여 최종 순위를 결정합니다.

비용: Dense(임베딩 쿼리 1회) + BM25(로컬 연산) = 매우 낮은 비용
"""

import logging
from typing import List, Dict

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

# EnsembleRetriever: langchain 버전에 따라 위치가 다름
# langchain <0.3  → langchain.retrievers
# langchain 0.3+  → langchain_classic.retrievers (또는 langchain_community)
# 순서대로 시도하여 첫 번째 성공한 경로 사용
try:
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever
    except ImportError:
        try:
            from langchain_community.retrievers.ensemble import EnsembleRetriever
        except ImportError:
            from langchain.retrievers import EnsembleRetriever

logger = logging.getLogger(__name__)

# Dense 0.6 / Sparse 0.4
# 의미 검색이 더 중요하지만 정확 키워드 누락 방지를 위해 BM25도 40% 유지
DEFAULT_WEIGHTS = [0.6, 0.4]


def build_hybrid_retriever(
    vectorstore: Chroma,
    bm25_retriever: BM25Retriever,
    weights: List[float] = None,
    k: int = 5,
) -> EnsembleRetriever:
    """
    Dense + Sparse EnsembleRetriever를 구성합니다.

    EnsembleRetriever는 두 리트리버를 독립적으로 실행한 뒤
    Reciprocal Rank Fusion(RRF)으로 결과를 병합합니다.

    매개변수:
        vectorstore   : 초기화된 Chroma 인스턴스
        bm25_retriever: 구축된 BM25Retriever 인스턴스
        weights       : [dense_가중치, sparse_가중치] (기본: [0.6, 0.4])
        k             : 최종 반환 문서 수

    반환값:
        설정된 EnsembleRetriever 인스턴스
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    # pydantic v2에서 .k 직접 설정 불가 → 새 인스턴스로 k 적용
    bm25_retriever = BM25Retriever.from_documents(
        bm25_retriever.docs if hasattr(bm25_retriever, "docs") else [],
        k=k,
    ) if not hasattr(bm25_retriever, "docs") else bm25_retriever
    try:
        object.__setattr__(bm25_retriever, "k", k)
    except Exception:
        pass

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=weights,
    )
    logger.info(
        "하이브리드 리트리버 구성 완료 (Dense=%.1f, Sparse=%.1f, k=%d)",
        weights[0], weights[1], k,
    )
    return ensemble


def compare_three_retrievers(
    vectorstore: Chroma,
    bm25_retriever: BM25Retriever,
    queries: List[str] = None,
) -> None:
    """
    Dense, Sparse, Hybrid 세 리트리버를 6개 쿼리로 비교합니다.

    각 리트리버의 강점을 보여주는 쿼리 패턴:
      - Dense 우위  : 의미·개념 쿼리 ("elegant dribbling")
      - Sparse 우위 : 정확 키워드 쿼리 ("Ronaldo hat-trick 2009")
      - Hybrid 우위 : 의미와 키워드가 혼합된 쿼리

    매개변수:
        vectorstore   : 초기화된 Chroma 인스턴스
        bm25_retriever: 구축된 BM25Retriever 인스턴스
        queries       : 테스트할 쿼리 리스트 (None이면 내장 테스트셋 사용)
    """
    # (쿼리 문자열, 예상 우위 시스템, 분석 설명) 튜플
    default_queries = [
        (
            "players known for elegant dribbling and creative flair",
            "dense",
            "의미 쿼리 — 'elegant flair'가 문서에 그대로 없음; "
            "Dense가 Messi·Ronaldinho 관련 문장을 의미적으로 탐지",
        ),
        (
            "Cristiano Ronaldo hat-trick 2009 Champions League",
            "sparse",
            "정확 키워드 쿼리 — BM25가 모든 토큰을 직접 매칭; "
            "Dense는 다른 Ronaldo 콘텐츠로 희석될 수 있음",
        ),
        (
            "Pep Guardiola Barcelona tactics possession football",
            "hybrid",
            "혼합 쿼리 — 'tactics possession'은 Dense, "
            "'Guardiola Barcelona'는 BM25; Hybrid가 최적",
        ),
        (
            "Brazilian players who won FIFA World Cup",
            "hybrid",
            "두 방식 모두 기여: 'Brazilian'은 정확 키워드, "
            "'won World Cup'은 의미 변형 가능",
        ),
        (
            "goalkeeper legends Premier League saves",
            "dense",
            "의미 개념 — 'legends saves'가 문자 그대로 없음; "
            "Dense가 Schmeichel·Buffon 관련 텍스트를 찾아냄",
        ),
        (
            "Messi Ballon d'Or 2012 record",
            "sparse",
            "정확 연도+수상명 — BM25가 '2012'와 'Ballon dOr'를 직접 매칭; "
            "Dense는 일반 Messi 콘텐츠로 순위 분산 가능",
        ),
    ]

    query_configs = default_queries if queries is None else [
        (q, "?", "사용자 정의 쿼리") for q in queries
    ]

    # 각 리트리버 k=3으로 설정
    dense_ret  = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    try:
        object.__setattr__(bm25_retriever, "k", 3)
    except Exception:
        pass
    hybrid_ret = build_hybrid_retriever(
        vectorstore, bm25_retriever, weights=DEFAULT_WEIGHTS, k=3
    )

    print("\n" + "=" * 90)
    print(f"{'세 가지 리트리버 비교':^90}")
    print("=" * 90)

    for idx, item in enumerate(query_configs, 1):
        query, expected, rationale = item if len(item) == 3 else (item[0], "?", "")

        print(f"\n  [{idx}] 쿼리: \"{query}\"")
        print(f"       예상 우위: {expected.upper()}")
        print(f"       근거: {rationale}")
        print()

        def _fmt(docs: List[Document]) -> List[str]:
            return [
                f"[{d.metadata.get('entity_name', '?')[:18]}] "
                f"{d.page_content[:70].replace(chr(10), ' ')}..."
                for d in docs
            ]

        try:
            dense_results  = dense_ret.invoke(query)
        except Exception as e:
            dense_results  = []
            logger.error("Dense 검색 실패: %s", e)

        try:
            sparse_results = bm25_retriever.invoke(query)
        except Exception as e:
            sparse_results = []
            logger.error("Sparse 검색 실패: %s", e)

        try:
            hybrid_results = hybrid_ret.invoke(query)
        except Exception as e:
            hybrid_results = []
            logger.error("Hybrid 검색 실패: %s", e)

        dense_fmt  = _fmt(dense_results)
        sparse_fmt = _fmt(sparse_results)
        hybrid_fmt = _fmt(hybrid_results)
        max_rows   = max(len(dense_fmt), len(sparse_fmt), len(hybrid_fmt))

        col_w  = 28
        header = f"  {'DENSE':^{col_w}} | {'SPARSE':^{col_w}} | {'HYBRID':^{col_w}}"
        print(header)
        print("  " + "-" * (col_w * 3 + 6))
        for row in range(max_rows):
            d = dense_fmt[row][:col_w]  if row < len(dense_fmt)  else ""
            s = sparse_fmt[row][:col_w] if row < len(sparse_fmt) else ""
            h = hybrid_fmt[row][:col_w] if row < len(hybrid_fmt) else ""
            print(f"  {d:<{col_w}} | {s:<{col_w}} | {h:<{col_w}}")

    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(levelname)s | %(name)s | %(message)s")

    from ingestion.loaders import load_all_documents
    from ingestion.splitters import get_production_chunks
    from retrieval.dense import build_or_load_vectorstore
    from retrieval.sparse import build_bm25_retriever

    docs   = load_all_documents()
    chunks = get_production_chunks(docs)
    vs     = build_or_load_vectorstore(chunks)
    bm25   = build_bm25_retriever(chunks)
    compare_three_retrievers(vs, bm25)
