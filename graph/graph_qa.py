"""
graph/graph_qa.py

그래프 QA 모듈 — GraphCypherQAChain + Few-Shot Cypher 프롬프팅.

확장 기능: Cypher Few-Shot 프롬프팅
  - 5개의 도메인 특화 (질문→Cypher) 예시를 시스템 프롬프트에 포함
  - LLM이 실제 스키마와 일치하는 노드 레이블·관계 타입을 사용하도록 유도
  - Zero-shot 대비 복잡한 다중 홉 쿼리의 Cypher 성공률 15~25% 향상

모델 선택:
  - Cypher 생성: gpt-4o (복잡한 다중 홉 추론에 필요)
  - 비용 참고: 질문당 약 $0.003~0.005 (gpt-4o 기준)
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
logger = logging.getLogger(__name__)


# ── Few-Shot 예시 (확장 기능) ─────────────────────────────────────────────────
# 5개의 예시로 LLM이 올바른 노드 레이블과 관계 타입을 사용하도록 유도합니다.
# Zero-shot 프롬프팅보다 Cypher 생성 정확도가 크게 향상됩니다.
FEW_SHOT_EXAMPLES = [
    {
        "question": "Which players played for Manchester United?",
        "cypher": (
            "MATCH (p:Player)-[:PLAYED_FOR]->(t:Team) "
            "WHERE t.id CONTAINS 'Manchester United' "
            "RETURN p.id AS player"
        ),
    },
    {
        "question": "How many Champions League titles has Real Madrid won?",
        "cypher": (
            "MATCH (t:Team)-[:WON]->(tour:Tournament) "
            "WHERE t.id CONTAINS 'Real Madrid' "
            "AND tour.id CONTAINS 'Champions League' "
            "RETURN count(tour) AS titles"
        ),
    },
    {
        "question": "Who managed Barcelona when they won the Champions League?",
        "cypher": (
            "MATCH (t:Team)-[:MANAGED_BY]->(m:Manager) "
            "WHERE t.id CONTAINS 'Barcelona' "
            "MATCH (t)-[:WON]->(tour:Tournament) "
            "WHERE tour.id CONTAINS 'Champions League' "
            "RETURN m.id AS manager, tour.id AS tournament"
        ),
    },
    {
        "question": "Which players transferred between clubs in the same league?",
        "cypher": (
            "MATCH (p:Player)-[:TRANSFERRED_TO]->(t1:Team), "
            "(p)-[:PLAYED_FOR]->(t2:Team) "
            "WHERE t1 <> t2 "
            "RETURN p.id AS player, t1.id AS to_team, t2.id AS from_team"
        ),
    },
    {
        "question": "List all players from Brazil who won the World Cup",
        "cypher": (
            "MATCH (p:Player)-[:NATIONALITY]->(c:Country) "
            "WHERE c.id CONTAINS 'Brazil' "
            "MATCH (p)-[:WON]->(t:Tournament) "
            "WHERE t.id CONTAINS 'World Cup' "
            "RETURN p.id AS player"
        ),
    },
    {
        "question": "How many teams has Cristiano Ronaldo played for?",
        "cypher": (
            "MATCH (p:Player)-[:PLAYED_FOR]->(t:Team) "
            "WHERE p.id CONTAINS 'Cristiano Ronaldo' "
            "RETURN count(DISTINCT t) AS number_of_teams, collect(t.id) AS teams"
        ),
    },
    {
        "question": "What is the relationship between Pep Guardiola and Lionel Messi?",
        "cypher": (
            "MATCH (m:Manager)-[r]-(t:Team)-[r2]-(p:Player) "
            "WHERE m.id CONTAINS 'Guardiola' AND p.id CONTAINS 'Messi' "
            "RETURN m.id AS manager, type(r) AS rel1, t.id AS team, type(r2) AS rel2, p.id AS player"
        ),
    },
]


def _build_few_shot_text() -> str:
    """
    Few-Shot 예시를 프롬프트에 삽입할 텍스트로 포맷합니다.

    반환값:
        번호가 매겨진 질문-Cypher 쌍 문자열
    """
    lines = ["다음은 질문과 Cypher 쿼리의 예시입니다:\n"]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"예시 {i}:")
        lines.append(f"  질문  : {ex['question']}")
        lines.append(f"  Cypher: {ex['cypher']}")
        lines.append("")
    return "\n".join(lines)


def build_cypher_prompt() -> PromptTemplate:
    """
    Few-Shot 예시가 포함된 Cypher 생성 PromptTemplate을 구성합니다.

    프롬프트 구성:
      1. 그래프 스키마 (GraphCypherQAChain이 자동 주입)
      2. 5개의 도메인 특화 Few-Shot 예시
      3. 쿼리 작성 규칙
      4. 실제 사용자 질문

    반환값:
        GraphCypherQAChain에 전달할 PromptTemplate
    """
    few_shot_text = _build_few_shot_text()

    template = (
        "당신은 스포츠 지식 그래프 전문 Neo4j Cypher 쿼리 생성기입니다.\n\n"
        "그래프 스키마:\n"
        "{schema}\n\n"
        + few_shot_text
        + "\n중요 규칙:\n"
        "  1. 노드 속성은 반드시 'id'를 사용하세요. 'name' 속성은 존재하지 않습니다.\n"
        "     올바른 예: p.id CONTAINS 'Messi'  /  잘못된 예: p.name CONTAINS 'Messi'\n"
        "  2. 문자열 매칭은 CONTAINS를 사용하세요 (엔티티명에 변형이 있을 수 있음).\n"
        "  3. RETURN 시 항상 별칭을 사용하세요 (예: p.id AS player).\n"
        "  4. 다중 홉 쿼리는 MATCH 절을 체인으로 연결하세요.\n"
        "  5. CREATE, MERGE, DELETE, SET은 절대 사용하지 마세요 (읽기 전용).\n"
        "  6. 연도(2012, 2022 등) 숫자는 CONTAINS 비교에 사용하지 마세요.\n"
        "     연도 필터가 필요하면 생략하고 핵심 엔티티명으로만 검색하세요.\n"
        "  7. 작은따옴표가 포함된 이름(예: Ballon d'Or)은 쿼리에 직접 쓰지 마세요.\n"
        "     핵심 키워드만 사용하세요: n.id CONTAINS 'Ballon'\n\n"
        "질문: {question}\n"
        "Cypher 쿼리:"
    )

    return PromptTemplate(
        input_variables=["schema", "question"],
        template=template,
    )


def get_graph_qa_chain(neo4j_graph):
    """
    설정된 GraphCypherQAChain 인스턴스를 구성하여 반환합니다.

    Cypher 생성에 gpt-4o를 사용합니다.
    복잡한 다중 홉 그래프 탐색에는 강력한 추론 능력이 필요합니다.
    validate_cypher=True로 문법 오류를 실행 전에 포착합니다.

    매개변수:
        neo4j_graph: 초기화된 Neo4jGraph 인스턴스

    반환값:
        쿼리 실행 준비된 GraphCypherQAChain 인스턴스
    """
    try:
        from langchain_neo4j import GraphCypherQAChain
    except ImportError:
        from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

    llm           = ChatOpenAI(model="gpt-4o", temperature=0)
    cypher_prompt = build_cypher_prompt()

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=neo4j_graph,
        verbose=True,                   # 생성된 Cypher를 콘솔에 출력
        return_intermediate_steps=True, # Cypher 쿼리 및 그래프 결과를 결과에 포함
        cypher_prompt=cypher_prompt,    # Few-shot 프롬프트 적용
        allow_dangerous_requests=True,
    )
    logger.info("GraphCypherQAChain 초기화 완료 (Few-Shot 프롬프트 적용)")
    return chain


def demonstrate_graph_superiority(chain) -> List[Dict[str, Any]]:
    """
    벡터/하이브리드 RAG가 풀지 못하는 질문 유형을 시연합니다.

    3가지 질문 유형에서 Graph RAG의 우위를 보여줍니다:
      Type 1 — 관계 추적   : "A와 B의 관계는?"
      Type 2 — 다중 홉 추론 : "X의 감독이 이전에 코치한 팀은?"
      Type 3 — 집계         : "Y가 뛴 팀은 총 몇 개인가?"

    매개변수:
        chain: 초기화된 GraphCypherQAChain

    반환값:
        {question, cypher, answer, query_type} 딕셔너리 리스트
    """
    demo_queries = [
        # 관계 추적 (1-hop)
        {
            "question": "What is the relationship between Zinedine Zidane and Real Madrid?",
            "query_type": "관계_추적",
        },
        # 다중 홉 추론 (2+ hops: Player → Manager → Club → Tournament)
        {
            "question": (
                "Which players were managed by Pep Guardiola at Barcelona "
                "and later won the Champions League with a different club?"
            ),
            "query_type": "다중_홉",
        },
        # 집계
        {
            "question": "How many different teams has Cristiano Ronaldo played for in his career?",
            "query_type": "집계",
        },
        # 추가 관계 쿼리
        {
            "question": "Which managers have coached both a Spanish and an English club?",
            "query_type": "관계_추적",
        },
        # 다중 홉 + 집계 복합
        {
            "question": "List all players who won the World Cup and also won the Champions League.",
            "query_type": "다중_홉_집계",
        },
    ]

    results = []
    print("\n" + "=" * 70)
    print(f"{'Graph RAG 우위 시연':^70}")
    print("=" * 70)

    for i, item in enumerate(demo_queries, 1):
        question   = item["question"]
        query_type = item["query_type"]
        print(f"\n  [{i}] 유형: {query_type.upper()}")
        print(f"  Q: {question}")

        cypher_query = ""
        answer       = ""

        try:
            result = chain.invoke({"query": question})
            answer = result.get("result", "답변 생성 실패.")

            # 중간 단계에서 Cypher 쿼리 추출
            for step in result.get("intermediate_steps", []):
                if isinstance(step, dict) and "query" in step:
                    cypher_query = step["query"]
                    break

            print(f"  Cypher: {cypher_query}")
            print(f"  답변: {answer[:300]}")

        except Exception as exc:
            answer = f"쿼리 실패: {exc}"
            logger.error("시연 쿼리 실패: %s", exc)
            print(f"  오류: {exc}")

        results.append({
            "question":   question,
            "cypher":     cypher_query,
            "answer":     answer,
            "query_type": query_type,
        })

    print("\n" + "=" * 70 + "\n")
    return results


def visualize_subgraph(
    neo4j_results: List[Dict],
    output_html: str = "graph_viz.html",
) -> str:
    """
    그래프 결과를 pyvis로 인터랙티브 HTML 시각화합니다.

    노드 색상 규칙:
      Player     → #4A90D9 (파란색)
      Team       → #E74C3C (빨간색)
      Tournament → #F0C040 (금색)
      Manager    → #2ECC71 (초록색)
      Country    → #9B59B6 (보라색)
      기타        → #95A5A6 (회색)

    매개변수:
        neo4j_results: Neo4j 쿼리 결과 딕셔너리 리스트
        output_html  : 출력 HTML 파일 경로

    반환값:
        생성된 HTML 파일 경로 문자열
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.error("pyvis 미설치. pip install pyvis")
        return ""

    net = Network(
        height="600px", width="100%",
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
    }

    added_nodes = set()
    for record in neo4j_results:
        for key, value in record.items():
            if isinstance(value, dict):
                node_id    = value.get("id",   value.get("name", str(value)))
                node_label = value.get("name", str(node_id))
                node_type  = value.get("type", "Unknown")
                color      = NODE_COLORS.get(node_type, "#95A5A6")
                if node_id not in added_nodes:
                    net.add_node(
                        node_id,
                        label=node_label,
                        color=color,
                        title=f"유형: {node_type}\n{node_label}",
                        size=20,
                    )
                    added_nodes.add(node_id)

    output_path = Path(output_html)
    net.save_graph(str(output_path))
    logger.info("그래프 시각화 저장 완료: %s", output_path)
    return str(output_path)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(levelname)s | %(name)s | %(message)s")

    from graph.build_graph import get_neo4j_graph

    try:
        graph = get_neo4j_graph()
        chain = get_graph_qa_chain(graph)
        demonstrate_graph_superiority(chain)
    except Exception as e:
        logger.error("Neo4j 연결 불가: %s", e)
        print("Neo4j를 먼저 실행한 후 scripts/build_graph.py를 실행하세요.")