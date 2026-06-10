"""
ui/app.py

스포츠 RAG 시스템 Streamlit UI

탭 구성:
  1. 무엇이든 물어보세요  — 3개 RAG 시스템 또는 동시 비교
  2. 리트리버 비교       — Dense / Sparse / Hybrid 청크 나란히 비교 + MMR 슬라이더
  3. Ragas 대시보드      — 사전 계산된 평가 결과 시각화
  4. 그래프 탐색기       — 자연어 → Cypher + pyvis 서브그래프 시각화

실행 방법:
  streamlit run ui/app.py
"""

import logging
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
logger = logging.getLogger(__name__)

# ── 페이지 기본 설정 ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Sports RAG System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* ── 답변 박스: 밝은 흰색 배경 ── */
  .answer-box {
    background: #ffffff;
    color: #1a1a1a;
    padding: 18px 20px;
    border-radius: 10px;
    border-left: 5px solid #4A90D9;
    border: 1px solid #d0e4f7;
    margin-top: 10px;
    line-height: 1.8;
    font-size: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
  /* ── Graph RAG 오류 메시지 박스 ── */
  .answer-box-error {
    background: #fff8f0;
    color: #7a3500;
    padding: 14px 18px;
    border-radius: 10px;
    border-left: 5px solid #f0a000;
    border: 1px solid #f5d5a0;
    margin-top: 10px;
    font-size: 14px;
  }
  /* ── Cypher 코드 블록 ── */
  .cypher-block {
    background: #f6f8fa;
    color: #0550ae;
    padding: 14px;
    border-radius: 8px;
    border: 1px solid #d0d7de;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    white-space: pre-wrap;
  }
  /* ── 예시 질문 버튼 ── */
  .example-label {
    font-size: 13px;
    color: #666;
    margin-bottom: 6px;
    font-weight: 600;
  }
  /* ── 청크 카드 ── */
  .chunk-card {
    background: #f9fafb;
    color: #1a1a1a;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    margin-bottom: 8px;
    font-size: 13px;
  }
  .overlap-card {
    background: #eff6ff;
    border-left: 4px solid #4A90D9;
  }
  .unique-card {
    background: #f0fdf4;
    border-left: 4px solid #2ECC71;
  }
</style>
""", unsafe_allow_html=True)


# ── 리소스 로드 (캐시) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="문서 로드 및 인덱스 초기화 중...")
def load_resources():
    """
    RAG 컴포넌트를 로드하고 캐싱합니다.

    Chroma 인덱스가 이미 있으면 임베딩 없이 로드합니다 (비용 절감).
    BM25 인덱스가 없으면 새로 구축합니다.

    반환값:
        docs, chunks, vectorstore, bm25, ensemble, vector_chain, hybrid_chain 포함 딕셔너리
    """
    from ingestion.loaders import load_all_documents
    from ingestion.splitters import get_production_chunks
    from retrieval.dense import build_or_load_vectorstore
    from retrieval.sparse import build_bm25_retriever, load_bm25_retriever, BM25_PICKLE_PATH
    from retrieval.hybrid import build_hybrid_retriever
    from chains.rag_chains import VectorRAGChain, HybridRAGChain

    docs   = load_all_documents()
    chunks = get_production_chunks(docs)
    vs     = build_or_load_vectorstore(chunks)

    # BM25: 저장된 인덱스 우선 로드, 없으면 구축
    try:
        bm25 = load_bm25_retriever()
    except FileNotFoundError:
        from retrieval.sparse import save_bm25_retriever
        bm25 = build_bm25_retriever(chunks)
        save_bm25_retriever(bm25)

    ensemble     = build_hybrid_retriever(vs, bm25)
    vector_chain = VectorRAGChain(vs)
    hybrid_chain = HybridRAGChain(ensemble)

    return {
        "docs":         docs,
        "chunks":       chunks,
        "vectorstore":  vs,
        "bm25":         bm25,
        "ensemble":     ensemble,
        "vector_chain": vector_chain,
        "hybrid_chain": hybrid_chain,
    }


@st.cache_resource(show_spinner="Neo4j 연결 중...")
def load_graph_resources():
    """
    Neo4j 그래프 리소스를 로드하고 캐싱합니다.

    연결 실패 시 {"available": False, "error": ...}를 반환합니다.

    반환값:
        neo4j_graph, graph_chain, available 포함 딕셔너리
    """
    from graph.build_graph import get_neo4j_graph
    from graph.graph_qa import get_graph_qa_chain
    from chains.rag_chains import GraphRAGChain

    try:
        neo4j_graph = get_neo4j_graph()
        graph_chain = GraphRAGChain(neo4j_graph)
        return {
            "neo4j_graph": neo4j_graph,
            "graph_chain": graph_chain,
            "available":   True,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ Sports RAG")
    st.markdown("**도메인:** 축구 / 풋볼")
    st.markdown("**데이터:** 프리미어리그, 라리가, UCL, 월드컵")
    st.divider()

    st.markdown("### 시스템 상태")
    with st.spinner("초기화 중..."):
        resources = load_resources()

    st.success(f"✓ {len(resources['chunks'])}개 청크 인덱싱 완료")
    st.success(f"✓ {len(resources['docs'])}개 원본 문서")

    graph_resources = load_graph_resources()
    if graph_resources["available"]:
        st.success("✓ Neo4j 그래프 연결됨")
    else:
        st.warning(f"⚠ Neo4j 미연결\n{graph_resources.get('error','')[:80]}")

    st.divider()
    st.markdown("### 모델 설정")
    st.code(
        "답변 생성 : gpt-4.1-mini\n"
        "그래프 QA : gpt-4o\n"
        "임베딩    : text-embedding-3-small"
    )
    st.divider()
    st.markdown("### 비용 절감 기능")
    st.info(
        "✓ Chroma 인덱스 영속화\n"
        "✓ BM25 pickle 재사용\n"
        "✓ 그래프 추출 JSON 캐시\n"
        "✓ Ragas 답변 캐시"
    )


# ── 탭 구성 ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 무엇이든 물어보세요",
    "🔍 리트리버 비교",
    "📊 Ragas 대시보드",
    "🕸️ 그래프 탐색기",
])


# ═══════════════════════════════════════════════════════════════════════════════
# 탭 1: 무엇이든 물어보세요
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("💬 축구에 대해 무엇이든 물어보세요")

    # ── 예시 질문 버튼 ──────────────────────────────────────────────────────
    # Streamlit에서 버튼 클릭으로 text_input 값을 바꾸는 올바른 방법:
    # text_input의 key를 session_state로 직접 제어한다.
    # (value= 인자는 key가 session_state에 이미 있으면 무시되므로 사용 불가)
    EXAMPLE_QUESTIONS = [
        "2022년 UEFA 챔피언스리그 우승팀은?",
        "Cristiano Ronaldo hat-trick 2009 Champions League",
        "펩 과르디올라와 리오넬 메시의 관계는?",
        "월드컵과 챔피언스리그를 모두 우승한 선수는?",
        "크리스티아누 호날두가 커리어에서 뛴 팀은 몇 개인가?",
        "FC 바르셀로나의 티키타카 전술을 설명해줘",
    ]

    # text_input key를 session_state와 직접 연결
    # 버튼 클릭 시 이 key 값을 변경하면 입력창이 즉시 업데이트됨
    if "tab1_q" not in st.session_state:
        st.session_state["tab1_q"] = ""

    st.markdown("💡 **예시 질문** (클릭하면 입력됩니다)")
    for i, q_text in enumerate(EXAMPLE_QUESTIONS):
        if st.button(q_text, key=f"ex_btn_{i}", use_container_width=True):
            st.session_state["tab1_q"] = q_text

    st.divider()

    question = st.text_input(
        "질문을 입력하세요:",
        placeholder="예: 2022년 UEFA 챔피언스리그 우승팀은?",
        key="tab1_q",
    )

    system_choice = st.radio(
        "검색 시스템 선택:",
        ["Vector RAG", "Hybrid RAG", "Graph RAG", "🔀 세 시스템 동시 비교"],
        horizontal=True,
    )

    if st.button("질문하기", key="tab1_ask", type="primary"):
        if not question.strip():
            st.warning("질문을 입력해 주세요.")
        else:
            if system_choice == "Vector RAG":
                with st.spinner("Vector RAG 처리 중..."):
                    result = resources["vector_chain"].invoke(question)
                st.markdown("### 답변 (Vector RAG — MMR 검색)")
                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander(f"검색된 컨텍스트 ({result.get('num_docs_retrieved',0)}개)"):
                    for i, ctx in enumerate(result.get("contexts", []), 1):
                        st.markdown(f"**청크 {i}:** {ctx[:300]}...")

            elif system_choice == "Hybrid RAG":
                with st.spinner("Hybrid RAG 처리 중 (Dense + BM25)..."):
                    result = resources["hybrid_chain"].invoke(question)
                st.markdown("### 답변 (Hybrid RAG — Dense+Sparse 결합)")
                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander(f"검색된 컨텍스트 ({result.get('num_docs_retrieved',0)}개)"):
                    for i, ctx in enumerate(result.get("contexts", []), 1):
                        st.markdown(f"**청크 {i}:** {ctx[:300]}...")

            elif system_choice == "Graph RAG":
                if not graph_resources["available"]:
                    st.error("Graph RAG를 사용할 수 없습니다. Neo4j 연결을 확인하세요.")
                else:
                    with st.spinner("Graph RAG 처리 중 (Cypher 쿼리 생성)..."):
                        result = graph_resources["graph_chain"].invoke(question)
                    st.markdown("### 답변 (Graph RAG — 지식 그래프 탐색)")
                    st.markdown(
                        f'<div class="answer-box">{result["answer"]}</div>',
                        unsafe_allow_html=True,
                    )
                    if result.get("cypher_query"):
                        with st.expander("생성된 Cypher 쿼리"):
                            st.markdown(
                                f'<div class="cypher-block">{result["cypher_query"]}</div>',
                                unsafe_allow_html=True,
                            )

            else:  # 세 시스템 동시 비교
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("#### 🔵 Vector RAG")
                    with st.spinner("처리 중..."):
                        v = resources["vector_chain"].invoke(question)
                    st.markdown(
                        f'<div class="answer-box">{v["answer"]}</div>',
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown("#### 🔴 Hybrid RAG")
                    with st.spinner("처리 중..."):
                        h = resources["hybrid_chain"].invoke(question)
                    st.markdown(
                        f'<div class="answer-box">{h["answer"]}</div>',
                        unsafe_allow_html=True,
                    )

                with col3:
                    st.markdown("#### 🟢 Graph RAG")
                    if graph_resources["available"]:
                        with st.spinner("처리 중..."):
                            g = graph_resources["graph_chain"].invoke(question)
                        answer_text = g["answer"]
                        is_error = ("neo4j_code" in answer_text.lower() or
                                    "syntaxerror" in answer_text.lower() or
                                    "ClientError" in answer_text)
                        box_cls = "answer-box-error" if is_error else "answer-box"
                        st.markdown(
                            f'<div class="{box_cls}">{answer_text}</div>',
                            unsafe_allow_html=True,
                        )
                        if g.get("cypher_query"):
                            with st.expander("Cypher"):
                                st.code(g["cypher_query"], language="cypher")
                    else:
                        st.warning("Neo4j 미연결")


# ═══════════════════════════════════════════════════════════════════════════════
# 탭 2: 리트리버 비교
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 리트리버 비교")
    st.markdown(
        "동일 쿼리에 대한 **Dense**, **Sparse(BM25)**, **Hybrid** 검색 결과를 나란히 비교합니다.\n\n"
        "🔵 파란 테두리 = 두 리트리버 모두 검색 (중복) / 🟢 초록 테두리 = 고유 결과"
    )

    query    = st.text_input(
        "검색 쿼리:", placeholder="예: Messi Barcelona Champions League",
        key="tab2_query",
    )
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        lambda_mult = st.slider(
            "MMR lambda_mult (관련성 ↔ 다양성)",
            0.0, 1.0, 0.5, 0.1,
            help="1.0 = 순수 관련성(유사도와 동일), 0.0 = 최대 다양성",
        )
    with col_s2:
        k_results = st.slider("리트리버당 결과 수(k)", 3, 8, 5)

    if st.button("검색", key="tab2_search", type="primary"):
        if not query.strip():
            st.warning("쿼리를 입력해 주세요.")
        else:
            vs   = resources["vectorstore"]
            bm25 = resources["bm25"]
            from retrieval.hybrid import build_hybrid_retriever

            # 각 리트리버 구성
            dense_ret = vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k_results, "fetch_k": 20, "lambda_mult": lambda_mult},
            )
            # pydantic v2 호환: 새 BM25Retriever로 k 적용
            try:
                object.__setattr__(bm25, 'k', k_results)
            except Exception:
                pass
            hybrid_ret = build_hybrid_retriever(vs, bm25, k=k_results)

            with st.spinner("검색 중..."):
                dense_docs  = dense_ret.invoke(query)
                sparse_docs = bm25.invoke(query)
                hybrid_docs = hybrid_ret.invoke(query)

            # 중복 청크 감지 (앞 80자 기준)
            dense_keys  = {d.page_content[:80] for d in dense_docs}
            sparse_keys = {d.page_content[:80] for d in sparse_docs}
            overlap     = dense_keys & sparse_keys

            col1, col2, col3 = st.columns(3)

            def _render_chunk(doc, overlap_set):
                """청크 카드 HTML을 렌더링합니다. 중복 여부에 따라 색상을 다르게 표시합니다."""
                key     = doc.page_content[:80]
                label   = doc.metadata.get("entity_name", "?")
                is_dup  = key in overlap_set
                badge   = "🔁 중복" if is_dup else "✨ 고유"
                css     = "overlap-card" if is_dup else "unique-card"
                snippet = doc.page_content[:200].replace("\n", " ")
                return (
                    f'<div class="chunk-card {css}">'
                    f'<b>[{label}]</b> {badge}<br>'
                    f'<small>{snippet}...</small>'
                    f'</div>'
                )

            with col1:
                st.markdown(f"#### 🔵 Dense (MMR λ={lambda_mult})")
                st.caption(f"{len(dense_docs)}개 결과")
                for doc in dense_docs:
                    st.markdown(_render_chunk(doc, overlap), unsafe_allow_html=True)

            with col2:
                st.markdown("#### 🟡 Sparse (BM25)")
                st.caption(f"{len(sparse_docs)}개 결과")
                for doc in sparse_docs:
                    st.markdown(_render_chunk(doc, overlap), unsafe_allow_html=True)

            with col3:
                st.markdown("#### 🔴 Hybrid (Dense+Sparse 결합)")
                st.caption(f"{len(hybrid_docs)}개 결과")
                for doc in hybrid_docs:
                    st.markdown(_render_chunk(doc, overlap), unsafe_allow_html=True)

            st.divider()
            st.markdown(
                f"**중복 분석:** Dense와 Sparse 결과 모두에 포함된 청크: **{len(overlap)}개**\n\n"
                f"Hybrid는 두 결과를 RRF(Reciprocal Rank Fusion)로 병합하여 "
                f"최종 **{len(hybrid_docs)}개** 결과를 반환합니다."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 탭 3: Ragas 대시보드
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Ragas 평가 대시보드")

    RESULTS_DIR = Path(__file__).parent.parent / "evaluation" / "results"
    v_path = RESULTS_DIR / "ragas_vectorrag.csv"
    h_path = RESULTS_DIR / "ragas_hybridrag.csv"

    if not v_path.exists() or not h_path.exists():
        st.info(
            "아직 평가 결과가 없습니다.\n\n"
            "아래 명령어로 평가를 먼저 실행하세요:\n\n"
            "```bash\n"
            "# 개발용 드라이런 (5개 질문, 저렴)\n"
            "python evaluation/ragas_eval.py --dry-run\n\n"
            "# 전체 평가 (최종 제출용)\n"
            "python evaluation/ragas_eval.py --judge-model gpt-4o\n"
            "```"
        )
    else:
        import pandas as pd
        import plotly.graph_objects as go

        v_df = pd.read_csv(v_path)
        h_df = pd.read_csv(h_path)

        metric_cols = ["faithfulness", "answer_relevancy",
                       "context_precision", "context_recall"]

        v_mean = v_df[v_df["question_id"] == "MEAN"][metric_cols].iloc[0]
        h_mean = h_df[h_df["question_id"] == "MEAN"][metric_cols].iloc[0]

        # 지표 카드
        st.markdown("#### 집계 점수 (전체 평균)")
        cols = st.columns(4)
        metric_labels = {
            "faithfulness":      "Faithfulness\n(환각 없음)",
            "answer_relevancy":  "Answer Relevancy\n(질문 관련성)",
            "context_precision": "Context Precision\n(검색 정밀도)",
            "context_recall":    "Context Recall\n(검색 재현율)",
        }
        for i, metric in enumerate(metric_cols):
            with cols[i]:
                delta     = h_mean[metric] - v_mean[metric]
                delta_str = f"Hybrid vs Vector {delta:+.3f}"
                st.metric(
                    label=metric_labels[metric],
                    value=f"{h_mean[metric]:.3f}",
                    delta=delta_str,
                    delta_color="normal" if delta >= 0 else "inverse",
                )

        st.divider()

        # Plotly 그룹 막대 그래프
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="VectorRAG", x=metric_cols,
            y=[v_mean[m] for m in metric_cols],
            marker_color="#4A90D9",
            text=[f"{v_mean[m]:.3f}" for m in metric_cols],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="HybridRAG", x=metric_cols,
            y=[h_mean[m] for m in metric_cols],
            marker_color="#E74C3C",
            text=[f"{h_mean[m]:.3f}" for m in metric_cols],
            textposition="outside",
        ))
        fig.update_layout(
            title="Ragas 4대 지표 비교: VectorRAG vs HybridRAG",
            barmode="group",
            yaxis=dict(range=[0, 1.15], title="점수 (0–1)"),
            legend=dict(orientation="h", x=0.3, y=1.12),
            height=420,
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

        # 질문별 점수 필터
        st.markdown("#### 질문별 점수")
        q_types = ["전체"] + sorted(
            v_df[v_df["question_id"] != "MEAN"]["question_type"].dropna().unique().tolist()
        )
        selected = st.selectbox("질문 유형 필터:", q_types)

        display = v_df[v_df["question_id"] != "MEAN"].copy()
        if selected != "전체":
            display = display[display["question_type"] == selected]

        if not display.empty:
            # Hybrid 점수 병합
            h_scores = h_df[h_df["question_id"] != "MEAN"][
                ["question_id"] + metric_cols
            ].rename(columns={m: f"hybrid_{m}" for m in metric_cols})
            merged = display.merge(h_scores, on="question_id", how="left")

            # 색상 히트맵으로 표시
            show_cols = ["question_id", "question_type", "difficulty", "question"] + metric_cols
            available = [c for c in show_cols if c in merged.columns]
            styled = merged[available].style.background_gradient(
                subset=[c for c in metric_cols if c in available],
                cmap="RdYlGn", vmin=0, vmax=1,
            )
            st.dataframe(styled, use_container_width=True)

        # 저장된 비교 PNG 표시
        comp_img = RESULTS_DIR / "ragas_comparison.png"
        if comp_img.exists():
            st.image(str(comp_img), caption="Ragas 비교 차트 (matplotlib)")


# ═══════════════════════════════════════════════════════════════════════════════
# 탭 4: 그래프 탐색기
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🕸️ 그래프 탐색기")
    st.markdown(
        "관계·다중 홉·집계 질문을 입력하면 **Cypher 쿼리**를 생성하여 "
        "Neo4j 지식 그래프를 탐색합니다."
    )

    if not graph_resources["available"]:
        st.error(
            f"Neo4j가 연결되지 않았습니다: {graph_resources.get('error', '알 수 없는 오류')}\n\n"
            "Neo4j를 먼저 실행하세요:\n\n"
            "```bash\n"
            "# Docker\n"
            "docker run -p 7474:7474 -p 7687:7687 \\\n"
            "  -e NEO4J_AUTH=neo4j/password123 neo4j:latest\n\n"
            "# 그래프 구축\n"
            "python scripts/build_graph.py\n"
            "```"
        )
    else:
        # 질문 입력
        graph_q = st.text_input(
            "그래프 질문:",
            placeholder="예: 크리스티아누 호날두가 뛴 팀은 몇 개인가?",
            key="tab4_graph_q",
        )

        # 예시 질문 선택
        example_queries = [
            "What is the relationship between Zinedine Zidane and Real Madrid?",
            "Which players were managed by Pep Guardiola at Barcelona?",
            "How many Champions League titles has Real Madrid won?",
            "Which managers have won the Champions League as both player and manager?",
            "List all players who won the FIFA World Cup with Brazil.",
            "How many different teams has Cristiano Ronaldo played for?",
        ]
        selected_ex = st.selectbox(
            "또는 예시 선택:",
            ["— 선택 —"] + example_queries,
            key="tab4_example",
        )
        if selected_ex != "— 선택 —":
            graph_q = selected_ex

        if st.button("그래프 쿼리 실행", key="tab4_go", type="primary"):
            if not graph_q.strip():
                st.warning("질문을 입력해 주세요.")
            else:
                with st.spinner("Cypher 쿼리 생성 및 Neo4j 실행 중 (gpt-4o)..."):
                    g_result = graph_resources["graph_chain"].invoke(graph_q)

                st.markdown("#### 답변")
                st.markdown(
                    f'<div class="answer-box">{g_result["answer"]}</div>',
                    unsafe_allow_html=True,
                )

                if g_result.get("cypher_query"):
                    st.markdown("#### 생성된 Cypher 쿼리")
                    st.code(g_result["cypher_query"], language="cypher")
                    st.caption(
                        "※ Few-Shot 프롬프팅 적용 (5개 예시) — Zero-shot 대비 "
                        "Cypher 성공률 15~25% 향상"
                    )

                if g_result.get("contexts"):
                    with st.expander("그래프 원본 결과"):
                        for ctx in g_result["contexts"][:10]:
                            st.write(ctx)

                # pyvis 그래프 시각화
                try:
                    from pyvis.network import Network
                    import tempfile, os

                    neo4j_graph = graph_resources["neo4j_graph"]
                    # id 속성 기준으로 서브그래프 조회
                    # (LLMGraphTransformer는 name 대신 id 속성으로 저장)
                    viz_results = neo4j_graph.query(
                        "MATCH (n)-[r]->(m) "
                        "WHERE n.id IS NOT NULL AND m.id IS NOT NULL "
                        "AND NOT n:Document AND NOT m:Document "
                        "RETURN n.id AS src_id, labels(n) AS src_labels, "
                        "type(r) AS rel_type, "
                        "m.id AS tgt_id, labels(m) AS tgt_labels "
                        "LIMIT 80"
                    )

                    if viz_results:
                        net = Network(
                            height="520px", width="100%",
                            bgcolor="#1a1a2e", font_color="white",
                            directed=True,
                        )
                        net.barnes_hut()

                        NODE_COLORS = {
                            "Player":     "#4A90D9",
                            "Team":       "#E74C3C",
                            "Tournament": "#F0C040",
                            "Manager":    "#2ECC71",
                            "Country":    "#9B59B6",
                            "Season":     "#F39C12",
                        }
                        added = set()

                        def _add_node(nid, labels):
                            if nid in added or not nid:
                                return
                            # __Entity__, Document 제외
                            clean = [l for l in (labels or []) if l not in ("__Entity__", "Document")]
                            ntype = clean[0] if clean else "Unknown"
                            color = NODE_COLORS.get(ntype, "#95A5A6")
                            net.add_node(
                                nid, label=str(nid)[:20],
                                color=color, size=18,
                                title=f"{ntype}\n{nid}",
                            )
                            added.add(nid)

                        for record in viz_results:
                            src_id  = record.get("src_id", "")
                            tgt_id  = record.get("tgt_id", "")
                            src_lbl = record.get("src_labels", [])
                            tgt_lbl = record.get("tgt_labels", [])
                            rel     = record.get("rel_type", "")
                            _add_node(src_id, src_lbl)
                            _add_node(tgt_id, tgt_lbl)
                            if src_id and tgt_id:
                                net.add_edge(
                                    src_id, tgt_id,
                                    label=rel, arrows="to",
                                    font={"size": 9},
                                )

                        # HTML 생성 후 Streamlit에 임베드
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".html", mode="w", encoding="utf-8"
                        ) as f:
                            net.save_graph(f.name)
                            html_content = open(f.name, encoding="utf-8").read()
                        os.unlink(f.name)

                        st.markdown("#### 지식 그래프 시각화 (pyvis)")
                        st.caption("노드 색상: 🔵 선수 / 🔴 팀 / 🟡 대회 / 🟢 감독 / 🟣 국가")
                        st.components.v1.html(html_content, height=540)
                    else:
                        st.info("시각화할 그래프 데이터가 없습니다. 먼저 그래프를 구축하세요.")

                except ImportError:
                    st.info("그래프 시각화를 사용하려면 pyvis를 설치하세요: `pip install pyvis`")
                except Exception as e:
                    logger.warning("시각화 생성 실패: %s", e)

        # Graph RAG 우위 시연
        st.divider()
        st.markdown("#### Graph RAG 우위 시연 질문")
        st.caption("벡터/하이브리드 RAG가 풀지 못하는 질문 유형입니다.")

        demo_items = [
            ("🔗 관계 추적", "What is the relationship between Zinedine Zidane and Real Madrid?"),
            ("🔀 다중 홉 추론", "Which players were managed by Pep Guardiola at Barcelona and later won the Champions League with a different club?"),
            ("🔢 집계", "How many different teams has Cristiano Ronaldo played for in his career?"),
        ]
        for label, demo_q in demo_items:
            with st.expander(f"🏷️ {label}"):
                st.markdown(f"**Q:** {demo_q}")
                if st.button(f"이 질문 실행", key=f"demo_{label}"):
                    with st.spinner("Graph RAG 처리 중..."):
                        res = graph_resources["graph_chain"].invoke(demo_q)
                    st.markdown(f"**A:** {res['answer']}")
                    if res.get("cypher_query"):
                        st.code(res["cypher_query"], language="cypher")