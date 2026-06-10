"""
evaluation/compare.py

세 가지 RAG 시스템(Vector, Hybrid, Graph)을 8개 대표 쿼리로 비교합니다.

핵심 결론 시연:
  "단일 RAG 아키텍처로는 모든 질문 유형을 커버할 수 없다 — 라우팅이 필요하다."

질문 유형별 예상 우위 시스템:
  factual        → Hybrid  (키워드 + 의미 균형)
  keyword_exact  → Hybrid  (BM25 정확 매칭)
  relational     → Graph   (관계 엣지 직접 탐색)
  multi_hop      → Graph   (2개 이상 노드 체인 탐색)
  aggregation    → Graph   (PLAYED_FOR 엣지 COUNT)

사용법:
  python evaluation/compare.py
  python evaluation/compare.py --no-graph
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# ── 8개 비교 쿼리 ─────────────────────────────────────────────────────────────
# (query_id, question, question_type) 형태의 튜플
COMPARISON_QUERIES: List[Tuple[str, str, str]] = [
    # 사실 질문 — vector/hybrid 우위 예상
    ("q_cmp_01", "Who is the all-time top scorer in the UEFA Champions League?",  "factual"),
    ("q_cmp_02", "What year was the Premier League founded?",                     "factual"),
    # 정확 키워드 — hybrid/sparse 우위 예상
    ("q_cmp_03", "Cristiano Ronaldo hat-trick 2009 Champions League",             "keyword_exact"),
    ("q_cmp_04", "Messi Ballon d'Or 2012 record goals scored",                   "keyword_exact"),
    # 관계 질문 — graph 우위 예상
    ("q_cmp_05", "What is the connection between Pep Guardiola and Lionel Messi?","relational"),
    ("q_cmp_06", "Which managers have coached both Real Madrid and Barcelona?",   "relational"),
    # 다중 홉 — graph 명확 우위
    ("q_cmp_07",
     "Which players won the World Cup and then went on to win the Champions League?",
     "multi_hop"),
    # 집계 — graph 우위
    ("q_cmp_08",
     "How many different clubs has Cristiano Ronaldo played for in total?",
     "aggregation"),
]

# 질문별 분석 메모
QUERY_ANALYSIS: Dict[str, Dict[str, str]] = {
    "q_cmp_01": {
        "expected_winner": "hybrid",
        "reason": "사실 질문 + 특정 엔티티명. Hybrid는 'Champions League' 키워드와 "
                  "득점 기록 관련 의미를 모두 포착합니다.",
    },
    "q_cmp_02": {
        "expected_winner": "hybrid",
        "reason": "짧은 사실 질문 + 정확한 연도. BM25 컴포넌트가 설립 연도 언급을 직접 찾습니다.",
    },
    "q_cmp_03": {
        "expected_winner": "hybrid",
        "reason": "정확 키워드 쿼리 (연도 포함). BM25가 모든 토큰을 직접 매칭합니다.",
    },
    "q_cmp_04": {
        "expected_winner": "hybrid",
        "reason": "정확 수상명+연도. Sparse 검색이 '2012'와 'Ballon dOr'를 직접 매칭합니다.",
    },
    "q_cmp_05": {
        "expected_winner": "graph",
        "reason": "관계 쿼리. 그래프가 MANAGED_BY·PLAYED_FOR 엣지로 "
                  "Guardiola→Barcelona←Messi를 직접 탐색합니다.",
    },
    "q_cmp_06": {
        "expected_winner": "graph",
        "reason": "다중 엔티티 관계 쿼리. 두 팀 모두에 MANAGED_BY 엣지를 가진 Manager 노드를 탐색합니다.",
    },
    "q_cmp_07": {
        "expected_winner": "graph",
        "reason": "다중 홉 쿼리 (Player→WON→WorldCup AND Player→WON→UCL). "
                  "두 토너먼트 관계를 동시에 탐색해야 합니다.",
    },
    "q_cmp_08": {
        "expected_winner": "graph",
        "reason": "집계 쿼리. 그래프가 PLAYED_FOR 엣지를 COUNT()로 집계합니다. "
                  "벡터 검색은 정확한 수를 반환하지 못합니다.",
    },
}


def _truncate(text: str, max_len: int = 200) -> str:
    """텍스트를 max_len자로 잘라 반환합니다."""
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def run_full_comparison(
    vector_chain,
    hybrid_chain,
    graph_chain=None,
    queries: List[Tuple[str, str, str]] = None,
) -> pd.DataFrame:
    """
    8개 비교 쿼리를 세 RAG 시스템 모두에 실행합니다.

    결과를 형식화된 표로 출력하고 CSV로 저장합니다.
    질문 유형별로 어느 시스템이 최적인지 정성적 분석을 포함합니다.

    매개변수:
        vector_chain : VectorRAGChain 인스턴스
        hybrid_chain : HybridRAGChain 인스턴스
        graph_chain  : GraphRAGChain 인스턴스 (없으면 생략)
        queries      : 쿼리 튜플 리스트 (None이면 COMPARISON_QUERIES 사용)

    반환값:
        비교 결과 DataFrame
    """
    if queries is None:
        queries = COMPARISON_QUERIES

    rows = []

    print("\n" + "=" * 100)
    print(f"{'세 가지 RAG 시스템 비교':^100}")
    print("=" * 100)

    for q_id, question, q_type in queries:
        analysis = QUERY_ANALYSIS.get(q_id, {})
        print(f"\n  ┌─ [{q_id}] 유형: {q_type.upper()}")
        print(f"  │  Q: {question}")
        print(f"  ├─ 예상 우위: {analysis.get('expected_winner', '?').upper()}")
        print(f"  │  근거: {analysis.get('reason', '')}")

        # Vector RAG
        try:
            v_result = vector_chain.invoke(question)
            v_answer = _truncate(v_result.get("answer", ""), 200)
        except Exception as e:
            v_answer = f"오류: {e}"
            logger.error("VectorRAG 실패 [%s]: %s", q_id, e)

        # Hybrid RAG
        try:
            h_result = hybrid_chain.invoke(question)
            h_answer = _truncate(h_result.get("answer", ""), 200)
        except Exception as e:
            h_answer = f"오류: {e}"
            logger.error("HybridRAG 실패 [%s]: %s", q_id, e)

        # Graph RAG
        if graph_chain is not None:
            try:
                g_result  = graph_chain.invoke(question)
                g_answer  = _truncate(g_result.get("answer", ""), 200)
                g_cypher  = g_result.get("cypher_query", "")
            except Exception as e:
                g_answer = f"오류: {e}"
                g_cypher = ""
                logger.error("GraphRAG 실패 [%s]: %s", q_id, e)
        else:
            g_answer = "(Graph RAG 미연결)"
            g_cypher = ""

        print(f"  ├─ Vector : {v_answer}")
        print(f"  ├─ Hybrid : {h_answer}")
        print(f"  └─ Graph  : {g_answer}")
        if g_cypher:
            print(f"     Cypher : {g_cypher[:120]}")

        rows.append({
            "query_id":        q_id,
            "question_type":   q_type,
            "question":        question,
            "vector_answer":   v_answer,
            "hybrid_answer":   h_answer,
            "graph_answer":    g_answer,
            "graph_cypher":    g_cypher,
            "expected_winner": analysis.get("expected_winner", "?"),
            "analysis":        analysis.get("reason", ""),
        })

    print("\n" + "=" * 100)

    df = pd.DataFrame(rows)

    # 질문 유형별 요약
    print("\n  질문 유형별 우위 시스템 요약")
    print(f"  {'유형':<18} {'예상 우위':<12} {'설명'}")
    print("  " + "-" * 75)
    type_notes = {
        "factual":       "Vector/Hybrid 충분 — 말뭉치에 사실이 존재",
        "keyword_exact": "Hybrid 우위 — BM25 정확 키워드 매칭",
        "relational":    "Graph 우위 — 명시적 관계 엣지 탐색",
        "multi_hop":     "Graph 우위 — 벡터 검색은 다단계 연결 불가",
        "aggregation":   "Graph 우위 — PLAYED_FOR 엣지 COUNT 집계",
    }
    for qtype in df["question_type"].unique():
        winner = QUERY_ANALYSIS.get(
            df[df["question_type"] == qtype]["query_id"].iloc[0], {}
        ).get("expected_winner", "?")
        note = type_notes.get(qtype, "")
        print(f"  {qtype:<18} {winner.upper():<12} {note}")

    # CSV 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "system_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  비교 결과 저장 → {out_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Vector, Hybrid, Graph RAG 시스템을 비교합니다."
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Graph RAG 생략 (Neo4j 미사용 시).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from ingestion.loaders import load_all_documents
    from ingestion.splitters import get_production_chunks
    from retrieval.dense import build_or_load_vectorstore
    from retrieval.sparse import build_bm25_retriever, load_bm25_retriever, BM25_PICKLE_PATH
    from retrieval.hybrid import build_hybrid_retriever
    from chains.rag_chains import VectorRAGChain, HybridRAGChain

    docs   = load_all_documents()
    chunks = get_production_chunks(docs)
    vs     = build_or_load_vectorstore(chunks)

    try:
        bm25 = load_bm25_retriever()
    except FileNotFoundError:
        from retrieval.sparse import save_bm25_retriever
        bm25 = build_bm25_retriever(chunks)
        save_bm25_retriever(bm25)

    ensemble     = build_hybrid_retriever(vs, bm25)
    vector_chain = VectorRAGChain(vs)
    hybrid_chain = HybridRAGChain(ensemble)

    graph_chain = None
    if not args.no_graph:
        try:
            from graph.build_graph import get_neo4j_graph
            from chains.rag_chains import GraphRAGChain
            neo4j_graph = get_neo4j_graph()
            graph_chain = GraphRAGChain(neo4j_graph)
        except Exception as e:
            logger.warning("Graph RAG 불가 (%s). --no-graph 옵션으로 경고를 숨길 수 있습니다.", e)

    run_full_comparison(vector_chain, hybrid_chain, graph_chain)


if __name__ == "__main__":
    main()
