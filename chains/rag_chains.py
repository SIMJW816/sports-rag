"""
chains/rag_chains.py

RAG 체인 구현 모듈 — 통합 인터페이스로 세 가지 검색 전략을 제공합니다.

클래스 구조:
  BaseRAGChain    — 공통 invoke() 인터페이스를 정의하는 추상 기반 클래스
  VectorRAGChain  — Dense MMR 검색 + LLM 답변 생성
  HybridRAGChain  — EnsembleRetriever(Dense+Sparse) + LLM 답변 생성
  GraphRAGChain   — GraphCypherQAChain을 통한 Neo4j 그래프 쿼리
  RouterRAGChain  — LLM 기반 3-class 분류 후 최적 체인으로 라우팅

비용 설정:
  - 답변 생성 LLM: gpt-4.1-mini (경제적)
  - 그래프 Cypher 생성: gpt-4o (복잡한 다중 홉 추론 필요)
  - 라우터 분류: gpt-4.1-mini (단순 분류 작업, 비용 최소화)
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

# EnsembleRetriever: langchain 버전에 따라 위치가 다름
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

load_dotenv()
logger = logging.getLogger(__name__)

# 답변 생성 기본 모델 — gpt-4.1-mini: 충분한 성능 + 낮은 비용
CHAT_MODEL = "gpt-4.1-mini"

# RAG 시스템 프롬프트
RAG_SYSTEM_PROMPT = """당신은 축구(풋볼)에 관한 전문 스포츠 애널리스트입니다.
아래 제공된 컨텍스트 정보만을 사용하여 질문에 답하세요.
컨텍스트에 충분한 정보가 없다면 "제 지식 베이스에 해당 내용이 없어 정확한 답변이 어렵습니다"라고 말하세요.
컨텍스트에 없는 선수 통계, 이적 세부 사항, 날짜 등을 절대 지어내지 마세요.
반드시 한국어로 답변하세요.

컨텍스트:
{context}"""



# 검색용 영어 번역 프롬프트
# 데이터(wiki_articles)가 영어이므로 한국어 쿼리를 영어로 번역해야
# 임베딩 공간에서 정확한 검색이 가능합니다
TRANSLATE_PROMPT = (
    "다음 질문을 영어로 번역하세요. 번역문만 출력하고 다른 말은 하지 마세요.\n"
    "질문: {question}\n"
    "영어 번역:"
)


def _translate_to_english(question: str, llm) -> str:
    """
    한국어 질문을 영어로 번역합니다.

    영어 문서(wiki_articles)를 대상으로 임베딩 검색하기 때문에
    한국어 쿼리는 임베딩 공간에서 영어 문서와 거리가 멀어 검색 품질이 저하됩니다.
    영어로 번역하면 동일한 임베딩 공간에서 더 정확한 검색이 가능합니다.

    ASCII 비율 기반으로 이미 영어인 질문은 번역을 생략합니다 (API 비용 절감).

    매개변수:
        question: 사용자 질문 (한국어 또는 영어)
        llm: 번역에 사용할 ChatOpenAI 인스턴스

    반환값:
        영어로 번역된 질문 문자열
    """
    ascii_ratio = sum(1 for c in question if ord(c) < 128) / max(len(question), 1)
    if ascii_ratio > 0.8:
        return question  # 이미 영어 → 번역 생략

    try:
        prompt = ChatPromptTemplate.from_template(TRANSLATE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        translated = chain.invoke({"question": question}).strip()
        logger.info("쿼리 번역: '%s' → '%s'", question[:40], translated[:60])
        return translated
    except Exception as exc:
        logger.warning("번역 실패, 원본 사용: %s", exc)
        return question


def _format_docs(docs: List[Document]) -> str:
    """
    Document 리스트를 LLM 컨텍스트 문자열로 변환합니다.

    매개변수:
        docs: 검색된 Document 리스트

    반환값:
        출처 태그가 포함된 연결 문자열
    """
    return "\n\n---\n\n".join(
        f"[출처: {d.metadata.get('entity_name', '알 수 없음')}]\n{d.page_content}"
        for d in docs
    )


class BaseRAGChain(ABC):
    """
    모든 RAG 체인의 추상 기반 클래스.

    모든 서브클래스는 invoke()를 구현하고 source_type 문자열을 노출해야 합니다.
    """

    source_type: str = "base"

    @abstractmethod
    def invoke(self, question: str) -> Dict[str, Any]:
        """
        질문을 받아 답변과 지원 컨텍스트를 반환합니다.

        매개변수:
            question: 자연어 질문 문자열

        반환값:
            {
              "answer"             : str,         # 생성된 답변
              "contexts"           : List[str],   # 검색된 컨텍스트
              "source_type"        : str,         # 체인 유형 식별자
              "num_docs_retrieved" : int,         # 검색된 문서 수
            }
        """

    def batch_invoke(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        여러 질문을 일괄 처리합니다.

        매개변수:
            questions: 질문 문자열 리스트

        반환값:
            각 질문의 결과 딕셔너리 리스트
        """
        return [self.invoke(q) for q in questions]


class VectorRAGChain(BaseRAGChain):
    """
    Dense 벡터 검색(MMR) + LLM 답변 생성 체인.

    Chroma MMR 검색으로 관련성과 다양성을 균형 있게 고려하여
    문서를 검색한 후 LLM으로 답변을 생성합니다.
    """

    source_type = "vector"

    def __init__(
        self,
        vectorstore: Chroma,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        model: str = CHAT_MODEL,
    ):
        """
        매개변수:
            vectorstore : 초기화된 Chroma 인스턴스
            k           : 검색할 문서 수
            fetch_k     : MMR 후보 풀 크기
            lambda_mult : MMR 관련성-다양성 트레이드오프
            model       : OpenAI 채팅 모델명
        """
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )
        self.llm = ChatOpenAI(model=model, temperature=0)
        self._build_chain()
        logger.info("VectorRAGChain 초기화 (model=%s, k=%d, λ=%.1f)", model, k, lambda_mult)

    def _build_chain(self) -> None:
        """답변 생성용 LLM 체인을 구성합니다.
        검색(retriever)은 invoke()에서 번역된 쿼리로 직접 수행합니다."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        # retriever를 LCEL에서 분리 — invoke()에서 번역 쿼리로 직접 검색
        self._answer_chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, question: str) -> Dict[str, Any]:
        """
        한국어 질문을 영어로 번역 후 검색하고, 한국어로 답변을 생성합니다.

        흐름:
          1. 한국어 질문 → 영어 번역 (ASCII 비율로 이미 영어면 생략)
          2. 번역된 영어 쿼리로 벡터 검색 (임베딩 공간 일치)
          3. 검색된 컨텍스트 + 원본 한국어 질문 → LLM → 한국어 답변

        매개변수:
            question: 자연어 질문 (한국어/영어 모두 가능)

        반환값:
            answer, contexts, source_type="vector" 포함 딕셔너리
        """
        try:
            # 1. 영어 번역 (검색 품질 향상)
            search_query = _translate_to_english(question, self.llm)
            # 2. 번역된 쿼리로 검색
            retrieved = self.retriever.invoke(search_query)
            # 3. 검색 결과 + 원본 질문으로 한국어 답변 생성
            context = _format_docs(retrieved)
            answer  = self._answer_chain.invoke(
                {"context": context, "question": question}
            )
            return {
                "answer":             answer,
                "contexts":           [d.page_content for d in retrieved],
                "source_type":        self.source_type,
                "num_docs_retrieved": len(retrieved),
            }
        except Exception as exc:
            logger.error("VectorRAGChain 오류: %s", exc)
            return {"answer": f"오류: {exc}", "contexts": [],
                    "source_type": self.source_type, "num_docs_retrieved": 0}


class HybridRAGChain(BaseRAGChain):
    """
    하이브리드 검색(Dense+Sparse) + LLM 답변 생성 체인.

    EnsembleRetriever로 의미 벡터 검색과 BM25 키워드 검색을 결합하여
    단일 방식의 약점을 보완합니다.
    """

    source_type = "hybrid"

    def __init__(
        self,
        ensemble_retriever: EnsembleRetriever,
        model: str = CHAT_MODEL,
    ):
        """
        매개변수:
            ensemble_retriever: 미리 구성된 EnsembleRetriever
            model             : OpenAI 채팅 모델명
        """
        self.retriever = ensemble_retriever
        self.llm       = ChatOpenAI(model=model, temperature=0)
        self._build_chain()
        logger.info("HybridRAGChain 초기화 (model=%s)", model)

    def _build_chain(self) -> None:
        """LCEL 파이프라인을 구성합니다."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        # LCEL에서 retriever를 제거: invoke()에서 번역된 쿼리로
        # 직접 검색한 docs를 컨텍스트로 주입하는 방식 사용
        self._answer_chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, question: str) -> Dict[str, Any]:
        """
        한국어 질문을 영어로 번역 후 하이브리드 검색하고, 한국어로 답변을 생성합니다.

        매개변수:
            question: 자연어 질문 (한국어/영어 모두 가능)

        반환값:
            answer, contexts, source_type="hybrid" 포함 딕셔너리
        """
        try:
            # 1. 한국어→영어 번역
            search_query = _translate_to_english(question, self.llm)
            # 2. 번역된 쿼리로 검색
            retrieved = self.retriever.invoke(search_query)
            # 3. 검색된 컨텍스트 + 원본 질문으로 한국어 답변 생성
            context = _format_docs(retrieved)
            answer  = self._answer_chain.invoke(
                {"context": context, "question": question}
            )
            return {
                "answer":             answer,
                "contexts":           [d.page_content for d in retrieved],
                "source_type":        self.source_type,
                "num_docs_retrieved": len(retrieved),
            }
        except Exception as exc:
            logger.error("HybridRAGChain 오류: %s", exc)
            return {"answer": f"오류: {exc}", "contexts": [],
                    "source_type": self.source_type, "num_docs_retrieved": 0}


class GraphRAGChain(BaseRAGChain):
    """
    Neo4j 지식 그래프 기반 RAG 체인 (GraphCypherQAChain 래퍼).

    벡터 검색이 어려운 질문 유형에서 우수한 성능을 발휘합니다:
      - 관계 추적: "A와 B의 관계는?"
      - 다중 홉 추론: "X의 감독이 이전에 코치한 선수는?"
      - 집계: "Y가 뛴 팀은 총 몇 개인가?"

    생성된 Cypher 쿼리가 결과 딕셔너리에 포함되어 투명성을 제공합니다.
    """

    source_type = "graph"

    def __init__(self, neo4j_graph, model: str = "gpt-4o"):
        """
        매개변수:
            neo4j_graph: 초기화된 Neo4jGraph 인스턴스
            model      : Cypher 생성 LLM (gpt-4o 권장 — 복잡한 추론 필요)
        """
        try:
            from langchain_neo4j import GraphCypherQAChain
        except ImportError:
            from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

        self.neo4j_graph = neo4j_graph
        self.llm         = ChatOpenAI(model=model, temperature=0)

        # Few-shot Cypher 프롬프트 적용 (graph/graph_qa.py)
        from graph.graph_qa import build_cypher_prompt
        cypher_prompt = build_cypher_prompt()

        self._qa_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=neo4j_graph,
            verbose=True,
            return_intermediate_steps=True,
            cypher_prompt=cypher_prompt,
            allow_dangerous_requests=True,
        )
        logger.info("GraphRAGChain 초기화 (model=%s)", model)

    def invoke(self, question: str) -> Dict[str, Any]:
        """
        자연어 질문 → Cypher 변환 → Neo4j 실행 → 답변 생성.

        매개변수:
            question: 자연어 질문

        반환값:
            answer, contexts(그래프 결과), cypher_query,
            source_type="graph" 포함 딕셔너리
        """
        try:
            result       = self._qa_chain.invoke({"query": question})
            answer       = result.get("result", "답변을 생성하지 못했습니다.")
            cypher_query = ""
            graph_ctx    = []

            # 중간 단계에서 Cypher 쿼리와 그래프 결과 추출
            for step in result.get("intermediate_steps", []):
                if isinstance(step, dict):
                    if "query" in step and not cypher_query:
                        cypher_query = step["query"]
                    if "context" in step:
                        graph_ctx = [str(r) for r in step["context"]]

            return {
                "answer":             answer,
                "contexts":           graph_ctx,
                "cypher_query":       cypher_query,
                "source_type":        self.source_type,
                "num_docs_retrieved": len(graph_ctx),
            }
        except Exception as exc:
            logger.error("GraphRAGChain 오류: %s", exc)
            return {
                "answer":             f"그래프 쿼리 실패: {exc}",
                "contexts":           [],
                "cypher_query":       "",
                "source_type":        self.source_type,
                "num_docs_retrieved": 0,
            }


class RouterRAGChain(BaseRAGChain):
    """
    LLM 기반 질문 분류 후 최적 체인으로 라우팅합니다.

    분류 기준:
      graph  — 관계/연결/다중 홉/집계 질문
      hybrid — 정확한 키워드·연도·통계가 포함된 질문
      vector — 일반적인 의미/서술 질문

    비용 최적화:
      - 분류기 LLM: gpt-4.1-mini (단순 3-class 분류, 저렴)
      - 분류 실패 시 키워드 기반 휴리스틱으로 폴백 (API 호출 없음)
    """

    source_type = "router"

    # 분류기 프롬프트 (영어 유지 — 분류 정확도가 높음)
    CLASSIFIER_PROMPT = """You are a RAG routing expert for a sports knowledge base.
Classify the following question into exactly ONE of three categories:

  graph   — Asks about RELATIONSHIPS, multi-hop reasoning, or aggregations.
             Examples: "Which players played for both X and Y?",
                       "How many clubs has Ronaldo played for?"

  hybrid  — Contains SPECIFIC keywords, names, exact years, or statistics.
             Examples: "Ronaldo hat-trick 2009", "Messi Ballon d'Or 2012 record"

  vector  — General semantic or descriptive questions.
             Examples: "What makes a great goalkeeper?",
                       "Describe Barcelona's playing style"

Respond with ONLY one word: graph, hybrid, or vector.

Question: {question}"""

    def __init__(
        self,
        vector_chain: VectorRAGChain,
        hybrid_chain: HybridRAGChain,
        graph_chain: Optional[GraphRAGChain] = None,
        model: str = CHAT_MODEL,
    ):
        """
        매개변수:
            vector_chain: 초기화된 VectorRAGChain
            hybrid_chain: 초기화된 HybridRAGChain
            graph_chain : 선택적 GraphRAGChain (없으면 hybrid로 폴백)
            model       : 분류기 LLM 모델명
        """
        self.chains = {
            "vector": vector_chain,
            "hybrid": hybrid_chain,
            # graph_chain이 None이면 hybrid로 폴백
            "graph":  graph_chain if graph_chain else hybrid_chain,
        }
        from langchain_core.prompts import ChatPromptTemplate as CPT
        classifier_llm = ChatOpenAI(model=model, temperature=0)
        self._classifier = (
            CPT.from_template(self.CLASSIFIER_PROMPT)
            | classifier_llm
            | StrOutputParser()
        )
        logger.info("RouterRAGChain 초기화 (분류기 모델: %s)", model)

    def classify_question(self, question: str) -> str:
        """
        질문을 'graph', 'hybrid', 'vector' 중 하나로 분류합니다.

        LLM 분류 → 예상 외 레이블이면 키워드 휴리스틱으로 폴백.

        매개변수:
            question: 분류할 질문 문자열

        반환값:
            'graph', 'hybrid', 'vector' 중 하나
        """
        try:
            label = self._classifier.invoke({"question": question}).strip().lower()
            if label in ("graph", "hybrid", "vector"):
                return label
            logger.warning("분류기가 예상 외 레이블 '%s' 반환 → 휴리스틱 사용", label)
        except Exception as exc:
            logger.error("분류기 LLM 실패: %s → 휴리스틱 폴백", exc)

        # 키워드 기반 휴리스틱 폴백 (API 호출 없음)
        q_lower = question.lower()
        graph_kw  = {"relationship", "connection", "managed", "transferred",
                     "played for", "how many", "list all", "which players",
                     "coached", "connected", "관계", "연결", "몇 개"}
        hybrid_kw = {"hat-trick", "ballon", "record", "2009", "2010",
                     "2011", "2012", "2013", "2014", "2015", "2016",
                     "2017", "2018", "2019", "2020", "2021", "2022",
                     "2023", "2024", "goal", "assist"}

        if any(kw in q_lower for kw in graph_kw):
            return "graph"
        if any(kw in q_lower for kw in hybrid_kw):
            return "hybrid"
        return "vector"

    def invoke(self, question: str) -> Dict[str, Any]:
        """
        질문을 분류하고 해당 체인에 위임합니다.

        매개변수:
            question: 자연어 질문

        반환값:
            선택된 체인의 결과 딕셔너리 + "routed_to" 키
        """
        route  = self.classify_question(question)
        logger.info("라우터: '%s' → %s", question[:60], route)
        result = self.chains[route].invoke(question)
        result["routed_to"]  = route
        result["source_type"] = f"router→{route}"
        return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(levelname)s | %(name)s | %(message)s")

    from ingestion.loaders import load_all_documents
    from ingestion.splitters import get_production_chunks
    from retrieval.dense import build_or_load_vectorstore
    from retrieval.sparse import build_bm25_retriever
    from retrieval.hybrid import build_hybrid_retriever

    docs     = load_all_documents()
    chunks   = get_production_chunks(docs)
    vs       = build_or_load_vectorstore(chunks)
    bm25     = build_bm25_retriever(chunks)
    ensemble = build_hybrid_retriever(vs, bm25)

    v_chain  = VectorRAGChain(vs)
    h_chain  = HybridRAGChain(ensemble)
    router   = RouterRAGChain(v_chain, h_chain)

    test_qs = [
        "Who won the Champions League in 2022?",
        "Messi Ballon d'Or 2012 record",
        "Which players have played for both Real Madrid and Barcelona?",
    ]
    for q in test_qs:
        res = router.invoke(q)
        print(f"\nQ: {q}")
        print(f"  라우팅: {res.get('routed_to')}")
        print(f"  답변: {res['answer'][:200]}")