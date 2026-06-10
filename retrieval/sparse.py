"""
retrieval/sparse.py

희소(Sparse) 검색 모듈 — BM25(Best Match 25) 알고리즘 사용.

BM25는 단어 빈도(TF-IDF 변형) 기반의 키워드 검색 알고리즘으로,
정확한 단어 일치에 강합니다. 선수 이름, 연도, 대회명처럼 특정
키워드가 포함된 쿼리에서 Dense 검색보다 우수합니다.

예: "Ronaldo hat-trick 2009 Champions League" →
    BM25가 모든 토큰을 직접 매칭하여 Dense보다 정확한 결과 반환

단점: "forward"와 "striker"처럼 의미는 같지만 단어가 다른 경우 인식 불가.
     → hybrid.py의 EnsembleRetriever가 Dense와 결합하여 보완

비용: API 호출 없음 (순수 로컬 연산)
"""

import logging
import pickle
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)

# BM25 인덱스 직렬화 저장 경로
BM25_PICKLE_PATH = Path(__file__).parent.parent / "bm25_index.pkl"


def build_bm25_retriever(docs: List[Document], k: int = 5) -> BM25Retriever:
    """
    문서 리스트로 BM25Retriever를 구축합니다.

    빌드 시 문서를 토큰화하고 TF-IDF 통계를 계산합니다.
    외부 API 호출이 없으므로 비용이 발생하지 않습니다.

    매개변수:
        docs: 인덱싱할 청크 Document 리스트
        k   : 쿼리당 반환할 문서 수

    반환값:
        설정된 BM25Retriever 인스턴스
    """
    # k는 from_documents() 생성 시 직접 전달 (pydantic v2 호환)
    retriever   = BM25Retriever.from_documents(docs, k=k)
    logger.info("BM25 리트리버 구축 완료: %d개 문서, k=%d", len(docs), k)
    return retriever


def save_bm25_retriever(
    retriever: BM25Retriever,
    path: Path = BM25_PICKLE_PATH,
) -> None:
    """
    BM25Retriever를 pickle로 직렬화하여 저장합니다.

    다음 실행 시 load_bm25_retriever()로 불러오면
    재구축 없이 즉시 사용할 수 있습니다.

    매개변수:
        retriever: 구축된 BM25Retriever 인스턴스
        path     : 저장 파일 경로
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(retriever, fh)
    logger.info("BM25 인덱스 저장 완료: %s", path)


def load_bm25_retriever(path: Path = BM25_PICKLE_PATH) -> BM25Retriever:
    """
    저장된 BM25Retriever를 디스크에서 불러옵니다.

    scripts/build_index.py 실행 후 저장된 인덱스를 재사용합니다.

    매개변수:
        path: pickle 파일 경로

    반환값:
        역직렬화된 BM25Retriever 인스턴스

    예외:
        FileNotFoundError: pickle 파일이 없을 때
    """
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 인덱스 없음: {path}\n"
            "먼저 scripts/build_index.py를 실행하세요."
        )
    with open(path, "rb") as fh:
        retriever = pickle.load(fh)
    logger.info("BM25 인덱스 로드 완료: %s", path)
    return retriever


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(levelname)s | %(name)s | %(message)s")

    from ingestion.loaders import load_all_documents
    from ingestion.splitters import get_production_chunks

    docs      = load_all_documents()
    chunks    = get_production_chunks(docs)
    retriever = build_bm25_retriever(chunks, k=5)

    test_query = "Cristiano Ronaldo hat-trick 2009 Champions League"
    results    = retriever.invoke(test_query)
    print(f"\nBM25 결과 쿼리: '{test_query}'")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata.get('entity_name', '?')}] "
              f"{doc.page_content[:100].replace(chr(10), ' ')}...")
