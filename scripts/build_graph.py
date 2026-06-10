"""
scripts/build_graph.py

Neo4j 지식 그래프를 구축하는 멱등성 스크립트.

실행 단계:
  1. Neo4j 연결 확인
  2. 기존 노드 수 확인 (이미 있으면 빌드 건너뜀)
  3. 위키 아티클 로드
  4. LLMGraphTransformer(gpt-4o)로 엔티티·관계 추출 후 삽입
  5. 스키마 검증 및 품질 보고서 출력

비용 절감 포인트:
  - 추출 결과를 JSON 캐시에 저장 → 재실행 시 gpt-4o 호출 없음
  - Neo4j 노드가 이미 존재하면 전체 빌드 건너뜀
  - --force-rebuild: 명시적으로 요청할 때만 재구축

사용법:
  python scripts/build_graph.py
  python scripts/build_graph.py --force-rebuild
  python scripts/build_graph.py --demo
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
        description="Sports RAG용 Neo4j 지식 그래프를 구축합니다."
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="기존 그래프를 초기화하고 재구축합니다.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="구축 후 시연 쿼리를 실행합니다.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # ── 1단계: Neo4j 연결 ──────────────────────────────────────────────────
    logger.info("1/4단계: Neo4j 연결 중...")
    from graph.build_graph import get_neo4j_graph

    try:
        graph = get_neo4j_graph()
        logger.info("✓ Neo4j 연결 성공.")
    except Exception as e:
        logger.error(
            "Neo4j 연결 실패: %s\n"
            "  → Neo4j가 실행 중인지 확인하세요 (Docker 또는 AuraDB)\n"
            "  → .env의 NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD를 확인하세요",
            e,
        )
        sys.exit(1)

    # ── 2단계: 기존 데이터 확인 ────────────────────────────────────────────
    logger.info("2/4단계: 기존 그래프 데이터 확인 중...")
    from graph.build_graph import _count_existing_nodes

    existing = _count_existing_nodes(graph)
    if existing > 0 and not args.force_rebuild:
        logger.info(
            "그래프에 이미 %d개 노드 존재 → 빌드 건너뜀.\n"
            "  → 재구축하려면 --force-rebuild 옵션을 사용하세요.",
            existing,
        )
        from graph.build_graph import verify_graph_schema
        verify_graph_schema(graph)
        if args.demo:
            _run_demo(graph)
        return

    # ── 3단계: 위키 아티클 로드 ────────────────────────────────────────────
    logger.info("3/4단계: 위키 아티클 로드 중...")
    from ingestion.loaders import load_article_documents

    article_docs = load_article_documents()
    if not article_docs:
        logger.error(
            "data/raw/wiki_articles/에 아티클 파일 없음. "
            "지식 그래프를 구축할 수 없습니다."
        )
        sys.exit(1)
    logger.info("위키 아티클 %d개 로드 완료.", len(article_docs))

    # ── 4단계: 그래프 구축 ─────────────────────────────────────────────────
    logger.info("4/4단계: LLMGraphTransformer(gpt-4o)로 그래프 추출 중...")
    from graph.build_graph import build_graph_from_documents, verify_graph_schema

    stats = build_graph_from_documents(
        article_docs, graph, force_rebuild=args.force_rebuild
    )

    # 스키마 검증
    verify_graph_schema(graph)

    # ── 최종 요약 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'그래프 구축 완료':^55}")
    print("=" * 55)
    print(f"  처리된 문서 수       : {stats['docs_processed']}개")
    print(f"  생성된 노드 수       : {stats['nodes_created']}개")
    print(f"  생성된 관계 수       : {stats['relationships_created']}개")
    print(f"  추출 오류 수         : {len(stats['extraction_errors'])}개")
    print("=" * 55)
    print("\n  다음 단계: python scripts/run_comparison.py\n")

    if args.demo:
        _run_demo(graph)


def _run_demo(graph) -> None:
    """그래프 QA 시연 쿼리를 실행합니다."""
    from graph.graph_qa import get_graph_qa_chain, demonstrate_graph_superiority

    logger.info("Graph RAG 시연 쿼리 실행 중...")
    try:
        chain = get_graph_qa_chain(graph)
        demonstrate_graph_superiority(chain)
    except Exception as e:
        logger.error("시연 실패: %s", e)


if __name__ == "__main__":
    main()
