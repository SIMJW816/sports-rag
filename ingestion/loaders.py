"""
ingestion/loaders.py

스포츠 RAG 시스템의 문서 로딩 모듈.

두 가지 Document Loader를 사용합니다:
  1. CSVLoader        — 정형 표 데이터 (선수, 팀, 대회)
  2. DirectoryLoader  — 비정형 위키 아티클 (.txt)

각 Document에는 아래 메타데이터 필드를 부여합니다:
  source      (str) : 원본 파일 경로
  doc_type    (str) : "csv" 또는 "article"
  entity_type (str) : "player" / "team" / "tournament"
  entity_name (str) : 엔티티 이름 (예: "Lionel Messi")
  date_added  (str) : ISO 날짜 (예: "2024-01-01")
"""

import logging
from datetime import date
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, DirectoryLoader, TextLoader

logger = logging.getLogger(__name__)

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent.parent / "data" / "raw"
PLAYERS_CSV  = DATA_DIR / "players.csv"
TEAMS_CSV    = DATA_DIR / "teams.csv"
TOURN_CSV    = DATA_DIR / "tournaments.csv"
WIKI_DIR     = DATA_DIR / "wiki_articles"

TODAY = date.today().isoformat()


def _enrich_metadata(
    docs: List[Document],
    doc_type: str,
    entity_type: str,
    name_col: str = None,
) -> List[Document]:
    """
    문서 리스트에 공통 메타데이터를 추가합니다.

    매개변수:
        docs        : 원본 Document 리스트
        doc_type    : "csv" 또는 "article"
        entity_type : "player" / "team" / "tournament"
        name_col    : CSV에서 entity_name으로 쓸 열 이름 (없으면 파일명 사용)

    반환값:
        메타데이터가 보강된 Document 리스트
    """
    enriched = []
    for doc in docs:
        meta = dict(doc.metadata)
        meta["doc_type"]    = doc_type
        meta["entity_type"] = entity_type
        meta["date_added"]  = TODAY

        if "source" not in meta:
            meta["source"] = "unknown"

        # entity_name: CSV의 name 열 파싱 → 없으면 파일명에서 추출
        if name_col:
            for line in doc.page_content.split("\n"):
                if line.lower().startswith(f"{name_col}:"):
                    meta["entity_name"] = line.split(":", 1)[1].strip()
                    break
            else:
                meta.setdefault("entity_name", entity_type)
        else:
            src = meta.get("source", "")
            meta["entity_name"] = Path(src).stem.replace("_", " ").title()

        enriched.append(Document(page_content=doc.page_content, metadata=meta))
    return enriched


def load_csv_documents() -> List[Document]:
    """
    CSVLoader를 사용해 세 개의 CSV 파일을 로드합니다.

    각 행이 독립된 Document가 되며, 행의 필드-값 쌍이 page_content에 담깁니다.
    players.csv → entity_type="player"
    teams.csv   → entity_type="team"
    tournaments.csv → entity_type="tournament"

    반환값:
        세 CSV의 Document를 합친 리스트
    """
    all_docs: List[Document] = []

    # (파일 경로, entity_type, name 열 이름) 세 쌍
    csv_configs = [
        (PLAYERS_CSV,  "player",     "name"),
        (TEAMS_CSV,    "team",       "name"),
        (TOURN_CSV,    "tournament", "name"),
    ]

    for csv_path, entity_type, name_col in csv_configs:
        if not csv_path.exists():
            logger.warning("CSV 파일 없음: %s", csv_path)
            continue
        try:
            loader = CSVLoader(
                file_path=str(csv_path),
                encoding="utf-8",
                csv_args={"delimiter": ","},
            )
            docs = loader.load()
            docs = _enrich_metadata(docs, doc_type="csv",
                                    entity_type=entity_type, name_col=name_col)
            all_docs.extend(docs)
            logger.info("%s에서 %d개 문서 로드", csv_path.name, len(docs))
        except Exception as exc:
            logger.error("%s 로드 실패: %s", csv_path, exc)

    return all_docs


def load_article_documents() -> List[Document]:
    """
    DirectoryLoader + TextLoader로 위키 아티클 .txt 파일을 로드합니다.

    wiki_articles/ 디렉토리의 각 .txt 파일이 하나의 Document가 됩니다.
    파일명에서 entity_type을 추론합니다:
      messi, ronaldo, zidane, guardiola → player
      madrid, barcelona, united, liverpool, bayern → team
      나머지 → tournament

    반환값:
        아티클 Document 리스트 (파일당 1개)
    """
    if not WIKI_DIR.exists():
        logger.warning("위키 아티클 디렉토리 없음: %s", WIKI_DIR)
        return []

    try:
        loader = DirectoryLoader(
            path=str(WIKI_DIR),
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        docs = loader.load()
    except Exception as exc:
        logger.error("위키 아티클 로드 실패: %s", exc)
        return []

    enriched = []
    for doc in docs:
        stem = Path(doc.metadata.get("source", "")).stem  # 예: "lionel_messi"

        # 파일명 키워드로 entity_type 추론
        player_kw    = {"messi", "ronaldo", "zidane", "guardiola",
                        "ronaldinho", "pele", "lewandowski"}
        team_kw      = {"madrid", "barcelona", "united",
                        "liverpool", "bayern", "city"}

        if any(kw in stem for kw in player_kw):
            entity_type = "player"
        elif any(kw in stem for kw in team_kw):
            entity_type = "team"
        else:
            entity_type = "tournament"

        meta = dict(doc.metadata)
        meta["doc_type"]    = "article"
        meta["entity_type"] = entity_type
        meta["entity_name"] = stem.replace("_", " ").title()
        meta["date_added"]  = TODAY
        enriched.append(Document(page_content=doc.page_content, metadata=meta))

    logger.info("위키 아티클 %d개 로드 완료", len(enriched))
    return enriched


def load_all_documents() -> List[Document]:
    """
    모든 소스에서 문서를 로드하여 합칩니다.

    반환값:
        CSV + 아티클 Document를 합친 리스트
    """
    csv_docs     = load_csv_documents()
    article_docs = load_article_documents()
    all_docs     = csv_docs + article_docs
    logger.info("전체 문서 로드 완료: %d개 (CSV: %d, 아티클: %d)",
                len(all_docs), len(csv_docs), len(article_docs))
    return all_docs


def print_loader_stats(docs: List[Document]) -> None:
    """
    로드된 문서의 통계를 형식화하여 출력합니다.

    doc_type, entity_type, source 파일별 문서 수를 보여줍니다.

    매개변수:
        docs: 로드된 Document 리스트
    """
    by_type:   Dict[str, int] = {}
    by_entity: Dict[str, int] = {}
    by_source: Dict[str, int] = {}

    for doc in docs:
        dt  = doc.metadata.get("doc_type",    "unknown")
        et  = doc.metadata.get("entity_type", "unknown")
        src = Path(doc.metadata.get("source", "unknown")).name

        by_type[dt]     = by_type.get(dt, 0)     + 1
        by_entity[et]   = by_entity.get(et, 0)   + 1
        by_source[src]  = by_source.get(src, 0)  + 1

    print("\n" + "=" * 55)
    print(f"{'문서 로더 통계':^55}")
    print("=" * 55)
    print(f"  전체 문서 수: {len(docs)}")

    print("\n  doc_type별:")
    for k, v in sorted(by_type.items()):
        print(f"    {k:<15} {v:>5}")

    print("\n  entity_type별:")
    for k, v in sorted(by_entity.items()):
        print(f"    {k:<15} {v:>5}")

    print("\n  소스 파일별 (상위 10개):")
    for k, v in sorted(by_source.items(), key=lambda x: -x[1])[:10]:
        print(f"    {k:<30} {v:>5}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(levelname)s | %(name)s | %(message)s")

    docs = load_all_documents()
    print_loader_stats(docs)

    # 샘플 문서 출력
    articles = [d for d in docs if d.metadata.get("doc_type") == "article"]
    if articles:
        d = articles[0]
        print("샘플 아티클 문서:")
        print(f"  entity_name : {d.metadata.get('entity_name')}")
        print(f"  entity_type : {d.metadata.get('entity_type')}")
        print(f"  date_added  : {d.metadata.get('date_added')}")
        print(f"  내용 첫 200자: {d.page_content[:200]}")
