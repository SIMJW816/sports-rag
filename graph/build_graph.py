"""
graph/build_graph.py

스포츠 RAG 시스템의 지식 그래프 구축 모듈.

비정형 위키 아티클 텍스트를 Neo4j 속성 그래프로 변환합니다.
LLMGraphTransformer(gpt-4o)로 엔티티와 관계를 자동 추출합니다.

비용 관리 전략 (3중 방어):
  Layer 1 — Neo4j 멱등성 검사
            노드가 1개 이상 존재하면 전체 빌드 건너뜀
            (force_rebuild=True로 강제 재구축 가능)

  Layer 2 — 추출 결과 JSON 캐시
            LLMGraphTransformer 호출 결과를
            data/processed/graph_extraction_cache.json에 저장.
            이후 실행에서는 캐시를 불러와 gpt-4o 호출 비용 0원.
            Neo4j가 재시작되더라도 그래프를 재삽입할 수 있습니다.

  Layer 3 — MERGE (CREATE 대신)
            Neo4j 삽입 시 항상 MERGE를 사용하여
            중복 노드·관계 생성을 방지합니다.

그래프 스키마:
  노드  : Player, Team, Tournament, Season, Country, Manager
  관계  : PLAYED_FOR, WON, COMPETED_IN, MANAGED_BY,
          TRANSFERRED_TO, NATIONALITY, SCORED_IN
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

load_dotenv()
logger = logging.getLogger(__name__)

# ── 추출 결과 캐시 경로 ────────────────────────────────────────────────────────
# LLMGraphTransformer 결과를 JSON으로 저장 → 재실행 시 gpt-4o 호출 없음
# 12개 아티클 기준 약 $0.72 절감
EXTRACTION_CACHE_PATH = (
    Path(__file__).parent.parent / "data" / "processed" / "graph_extraction_cache.json"
)

# ── 허용 스키마 ────────────────────────────────────────────────────────────────
# LLMGraphTransformer가 이 목록 밖의 타입을 생성하면 경고로 기록됩니다.
ALLOWED_NODES = ["Player", "Team", "Tournament", "Season", "Country", "Manager"]
ALLOWED_RELATIONSHIPS = [
    "PLAYED_FOR",
    "WON",
    "COMPETED_IN",
    "MANAGED_BY",
    "TRANSFERRED_TO",
    "NATIONALITY",
    "SCORED_IN",
]


# ── 캐시 I/O ──────────────────────────────────────────────────────────────────

def _save_extraction_cache(cache: Dict[str, Any]) -> None:
    """
    그래프 추출 결과를 JSON 파일로 저장합니다.

    다음 실행 시 이 파일을 불러와 gpt-4o 호출을 생략합니다.

    매개변수:
        cache: {entity_name: {nodes, relationships}} 형태의 딕셔너리
    """
    EXTRACTION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXTRACTION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.info("추출 캐시 저장 완료 → %s (%d개 항목)", EXTRACTION_CACHE_PATH, len(cache))


def _load_extraction_cache() -> Dict[str, Any]:
    """
    저장된 추출 캐시를 불러옵니다.

    반환값:
        저장된 캐시 딕셔너리 (없으면 빈 딕셔너리)
    """
    if not EXTRACTION_CACHE_PATH.exists():
        return {}
    with open(EXTRACTION_CACHE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("추출 캐시 로드: %d개 항목 (gpt-4o 호출 생략)", len(data))
    return data


# ── Neo4j 연결 ─────────────────────────────────────────────────────────────────

def get_neo4j_graph():
    """
    Neo4jGraph 연결 인스턴스를 생성하여 반환합니다.

    .env에서 NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD를 읽습니다.

    반환값:
        Neo4jGraph 인스턴스

    예외:
        ImportError : langchain-neo4j 미설치 시
        Exception   : Neo4j 연결 실패 시
    """
    try:
        from langchain_neo4j import Neo4jGraph
    except ImportError:
        from langchain_community.graphs import Neo4jGraph

    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    try:
        graph = Neo4jGraph(url=uri, username=username, password=password)
        logger.info("Neo4j 연결 완료: %s", uri)
        return graph
    except Exception as exc:
        logger.error("Neo4j 연결 실패 (%s): %s", uri, exc)
        raise


def _count_existing_nodes(graph) -> int:
    """
    Neo4j 데이터베이스의 전체 노드 수를 조회합니다.

    매개변수:
        graph: Neo4jGraph 인스턴스

    반환값:
        현재 노드 수 (조회 실패 시 0)
    """
    try:
        result = graph.query("MATCH (n) RETURN count(n) AS cnt")
        return result[0]["cnt"] if result else 0
    except Exception as exc:
        logger.error("노드 수 조회 실패: %s", exc)
        return 0


# ── 핵심: 그래프 구축 ──────────────────────────────────────────────────────────

def build_graph_from_documents(
    docs: List[Document],
    graph,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    문서에서 엔티티와 관계를 추출하여 Neo4j에 삽입합니다.

    3중 비용 방어:
      1) Neo4j에 노드가 존재하면 전체 건너뜀
      2) 추출 캐시에 있는 문서는 LLM 호출 없이 삽입
      3) Neo4j 삽입은 항상 MERGE로 중복 방지

    매개변수:
        docs          : 처리할 Document 리스트 (위키 아티클 권장)
        graph         : 초기화된 Neo4jGraph 인스턴스
        force_rebuild : True이면 그래프 초기화 후 재구축

    반환값:
        {
          "docs_processed"          : int,
          "nodes_created"           : int,
          "relationships_created"   : int,
          "node_type_distribution"  : dict,
          "edge_type_distribution"  : dict,
          "extraction_errors"       : List[str],
        }
    """
    try:
        from langchain_experimental.graph_transformers import LLMGraphTransformer
    except ImportError:
        logger.error("langchain_experimental 미설치. pip install langchain-experimental")
        raise

    # ── Layer 1: Neo4j 멱등성 검사 ─────────────────────────────────────────
    existing_count = _count_existing_nodes(graph)
    if existing_count > 0 and not force_rebuild:
        logger.info(
            "그래프에 이미 %d개 노드 존재 → 빌드 건너뜀. "
            "재구축하려면 force_rebuild=True 사용.",
            existing_count,
        )
        return {
            "docs_processed": 0, "nodes_created": 0,
            "relationships_created": 0,
            "node_type_distribution": {}, "edge_type_distribution": {},
            "extraction_errors": ["건너뜀: 그래프가 이미 존재합니다."],
        }

    if force_rebuild and existing_count > 0:
        logger.warning("force_rebuild=True — 기존 그래프 초기화 중 (%d개 노드)", existing_count)
        graph.query("MATCH (n) DETACH DELETE n")

    # ── Layer 2: 추출 캐시 확인 ────────────────────────────────────────────
    extraction_cache = _load_extraction_cache() if not force_rebuild else {}
    docs_to_extract  = [
        d for d in docs
        if d.metadata.get("entity_name", "") not in extraction_cache
    ]
    cached_count = len(docs) - len(docs_to_extract)
    if cached_count:
        logger.info(
            "캐시 적중: %d/%d개 문서 (LLM 호출 생략)",
            cached_count, len(docs),
        )

    # 비용 추정 출력 (미추출 문서에 대해서만)
    if docs_to_extract:
        print(f"\n{'='*60}")
        print(f"  그래프 구축 비용 추정")
        print(f"  전체 문서 수        : {len(docs)}개")
        print(f"  캐시 적중 (무료)    : {cached_count}개")
        print(f"  실제 LLM 호출 대상  : {len(docs_to_extract)}개")
        print(f"  예상 토큰 (gpt-4o)  : ~{len(docs_to_extract) * 1200:,}")
        print(f"  예상 비용           : ~${len(docs_to_extract) * 1200 * 0.000005:.3f}")
        print(f"  캐시 저장 경로      : {EXTRACTION_CACHE_PATH}")
        print(f"{'='*60}\n")
    else:
        print(f"\n  [캐시] 모든 문서({len(docs)}개)가 캐시에 있습니다 — LLM 비용 $0.00\n")

    # ── LLM 추출 (미캐시 문서만) ───────────────────────────────────────────
    if docs_to_extract:
        llm         = ChatOpenAI(model="gpt-4o", temperature=0)
        transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=ALLOWED_NODES,
            allowed_relationships=ALLOWED_RELATIONSHIPS,
            strict_mode=False,  # 타입이 정확히 일치하지 않아도 추출 허용
        )

        for i, doc in enumerate(docs_to_extract, 1):
            entity_name = doc.metadata.get("entity_name", f"doc_{i}")
            logger.info("[%d/%d] 추출 중 (gpt-4o): %s", i, len(docs_to_extract), entity_name)
            try:
                graph_docs = transformer.convert_to_graph_documents([doc])
                # 직렬화 가능한 형태로 캐시에 저장 (다음 실행 시 재사용)
                extraction_cache[entity_name] = {
                    "nodes": [
                        {"id": n.id, "type": n.type, "properties": n.properties}
                        for gd in graph_docs for n in gd.nodes
                    ],
                    "relationships": [
                        {"source": r.source.id, "target": r.target.id,
                         "type": r.type, "properties": r.properties}
                        for gd in graph_docs for r in gd.relationships
                    ],
                    "_graph_docs_obj": graph_docs,  # 인메모리 전용 (저장 제외)
                }
            except Exception as exc:
                logger.error("추출 실패 — %s: %s", entity_name, exc)
                extraction_cache[entity_name] = {
                    "nodes": [], "relationships": [], "error": str(exc)
                }

        # 직렬화 불가 키(_graph_docs_obj) 제외 후 캐시 저장
        serialisable = {
            k: {kk: vv for kk, vv in v.items() if kk != "_graph_docs_obj"}
            for k, v in extraction_cache.items()
        }
        _save_extraction_cache(serialisable)

    # ── Neo4j 삽입 ─────────────────────────────────────────────────────────
    stats = {
        "docs_processed": 0, "nodes_created": 0,
        "relationships_created": 0,
        "node_type_distribution": {}, "edge_type_distribution": {},
        "extraction_errors": [],
    }

    for doc in docs:
        entity_name = doc.metadata.get("entity_name", "")
        cached      = extraction_cache.get(entity_name, {})

        if cached.get("error"):
            stats["extraction_errors"].append(f"{entity_name}: {cached['error']}")
            continue

        # 이번 실행에서 새로 추출한 경우: 인메모리 GraphDocument 객체 사용
        in_memory = cached.get("_graph_docs_obj")
        if in_memory:
            try:
                graph.add_graph_documents(
                    in_memory, baseEntityLabel=True, include_source=True
                )
            except Exception as exc:
                logger.error("Neo4j 삽입 실패 (%s): %s", entity_name, exc)
                stats["extraction_errors"].append(f"{entity_name}(삽입): {exc}")
                continue
        else:
            # 캐시 적중: 저장된 노드/관계를 Cypher MERGE로 삽입
            _insert_from_cache(graph, cached)

        # 통계 집계
        for node in cached.get("nodes", []):
            nt = node.get("type", "Unknown")
            stats["node_type_distribution"][nt] = (
                stats["node_type_distribution"].get(nt, 0) + 1
            )
            stats["nodes_created"] += 1

        for rel in cached.get("relationships", []):
            rt = rel.get("type", "UNKNOWN")
            stats["edge_type_distribution"][rt] = (
                stats["edge_type_distribution"].get(rt, 0) + 1
            )
            stats["relationships_created"] += 1

        stats["docs_processed"] += 1

    _print_extraction_report(stats)
    return stats


def _insert_from_cache(graph, cached: Dict[str, Any]) -> None:
    """
    캐시에서 불러온 노드와 관계를 Neo4j에 MERGE로 삽입합니다.

    Neo4j가 재시작된 후 캐시를 이용해 그래프를 재구성할 때 사용합니다.
    MERGE는 이미 존재하는 노드/관계를 중복 생성하지 않습니다.

    매개변수:
        graph  : Neo4jGraph 인스턴스
        cached : {nodes: [...], relationships: [...]} 형태의 캐시 항목
    """
    # 노드 삽입
    for node in cached.get("nodes", []):
        node_id   = node.get("id", "").replace("'", "\\'")
        node_type = node.get("type", "Unknown")
        if not node_id or node_type not in ALLOWED_NODES:
            continue
        try:
            graph.query(
                f"MERGE (n:{node_type} {{id: $nid}}) "
                "ON CREATE SET n.name = $nid",
                {"nid": node_id},
            )
        except Exception as exc:
            logger.debug("노드 삽입 건너뜀 (%s): %s", node_id, exc)

    # 관계 삽입
    for rel in cached.get("relationships", []):
        src      = rel.get("source", "").replace("'", "\\'")
        tgt      = rel.get("target", "").replace("'", "\\'")
        rel_type = rel.get("type", "")
        if not src or not tgt or rel_type not in ALLOWED_RELATIONSHIPS:
            continue
        try:
            graph.query(
                f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
                f"MERGE (a)-[r:{rel_type}]->(b)",
                {"src": src, "tgt": tgt},
            )
        except Exception as exc:
            logger.debug("관계 삽입 건너뜀 (%s→%s): %s", src, tgt, exc)


def _print_extraction_report(stats: Dict[str, Any]) -> None:
    """
    추출 품질 보고서를 형식화하여 출력합니다.

    매개변수:
        stats: build_graph_from_documents()가 반환하는 통계 딕셔너리
    """
    print("\n" + "=" * 55)
    print(f"{'그래프 추출 품질 보고서':^55}")
    print("=" * 55)
    print(f"  처리된 문서 수       : {stats['docs_processed']}")
    print(f"  생성된 노드 수       : {stats['nodes_created']}")
    print(f"  생성된 관계 수       : {stats['relationships_created']}")

    print("\n  노드 타입 분포:")
    for k, v in sorted(stats["node_type_distribution"].items(), key=lambda x: -x[1]):
        bar = "█" * min(v, 30)
        print(f"    {k:<15} {v:>4}  {bar}")

    print("\n  관계 타입 분포:")
    for k, v in sorted(stats["edge_type_distribution"].items(), key=lambda x: -x[1]):
        bar = "█" * min(v, 30)
        print(f"    {k:<20} {v:>4}  {bar}")

    if stats["extraction_errors"]:
        print(f"\n  추출 오류 ({len(stats['extraction_errors'])}개) — 보고서에 기록 필요:")
        for err in stats["extraction_errors"][:5]:
            print(f"    ⚠ {err}")
    else:
        print("\n  추출 오류 없음. ✓")
    print("=" * 55 + "\n")


def verify_graph_schema(graph) -> Dict[str, Any]:
    """
    실제 Neo4j 스키마와 허용 스키마를 비교 검증합니다.

    LLM이 허용 목록 외의 노드 레이블이나 관계 타입을 생성했는지 확인합니다.
    예상치 못한 항목은 보고서에 기록해야 합니다.

    매개변수:
        graph: Neo4jGraph 인스턴스

    반환값:
        {actual_node_labels, actual_rel_types, unexpected_nodes, unexpected_rels}
    """
    result = {
        "actual_node_labels": [], "actual_rel_types": [],
        "unexpected_nodes":   [], "unexpected_rels": [],
    }

    try:
        # 실제 노드 레이블 조회
        node_res = graph.query(
            "CALL db.labels() YIELD label RETURN collect(label) AS labels"
        )
        actual_labels = node_res[0]["labels"] if node_res else []
        result["actual_node_labels"] = actual_labels
        result["unexpected_nodes"]   = [
            lb for lb in actual_labels
            if lb not in ALLOWED_NODES and lb != "__Entity__"
        ]

        # 실제 관계 타입 조회
        rel_res = graph.query(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN collect(relationshipType) AS types"
        )
        actual_rels = rel_res[0]["types"] if rel_res else []
        result["actual_rel_types"] = actual_rels
        result["unexpected_rels"]  = [
            r for r in actual_rels if r not in ALLOWED_RELATIONSHIPS
        ]

        print("\n" + "=" * 55)
        print(f"{'그래프 스키마 검증':^55}")
        print("=" * 55)
        print(f"  실제 노드 레이블  : {actual_labels}")
        print(f"  예상 외 노드      : {result['unexpected_nodes'] or '없음'}")
        print(f"  실제 관계 타입    : {actual_rels}")
        print(f"  예상 외 관계      : {result['unexpected_rels'] or '없음'}")
        print("=" * 55 + "\n")

    except Exception as exc:
        logger.error("스키마 검증 실패: %s", exc)
        result["error"] = str(exc)

    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(levelname)s | %(name)s | %(message)s")

    from ingestion.loaders import load_article_documents

    graph        = get_neo4j_graph()
    article_docs = load_article_documents()
    print(f"위키 아티클 {len(article_docs)}개 로드 완료.")

    stats = build_graph_from_documents(article_docs, graph, force_rebuild=False)
    verify_graph_schema(graph)
