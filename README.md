# ⚽ Sports RAG System

**지식 그래프 강화 RAG — 축구 도메인 지식 베이스**

하이브리드 검색(Dense+Sparse), MMR 다양성 검색, Ragas 4대 지표 평가,
Knowledge Graph RAG(Neo4j + LLMGraphTransformer)를 하나의 시스템에 통합한
프로덕션 수준의 RAG 프로젝트입니다.

LangSmith URL: https://smith.langchain.com/public/cbe620cd-b041-48cd-9d48-eaf3791f1baa/r

---

## 왜 스포츠/축구 도메인인가?

축구는 **엔티티와 관계가 풍부한** Graph RAG 최적 도메인입니다.

| 노드 타입    | 예시                            |
|--------------|---------------------------------|
| `Player`     | 리오넬 메시, 크리스티아누 호날두 |
| `Team`       | 레알 마드리드, FC 바르셀로나     |
| `Tournament` | UEFA 챔피언스리그, FIFA 월드컵   |
| `Manager`    | 펩 과르디올라, 지네딘 지단       |
| `Country`    | 아르헨티나, 브라질, 스페인       |

| 관계              | 의미                                    |
|-------------------|-----------------------------------------|
| `PLAYED_FOR`      | 선수 → 팀 (소속)                        |
| `WON`             | 팀/선수 → 대회 (우승)                   |
| `COMPETED_IN`     | 팀 → 대회 (시즌 참가)                   |
| `MANAGED_BY`      | 팀 → 감독                               |
| `TRANSFERRED_TO`  | 선수 → 새 팀 (이적)                     |
| `NATIONALITY`     | 선수 → 국가                             |
| `SCORED_IN`       | 선수 → 대회 (득점 기록)                 |

*"과르디올라가 바르셀로나에서 지도한 선수 중 나중에 다른 팀에서 챔피언스리그를 우승한 사람은?"*
→ 이런 다중 홉 질문은 벡터 검색만으로는 불가능합니다. 그래프 탐색이 필수입니다.

---

## 프로젝트 구조

```
sports_rag/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── players.csv             # 선수 데이터 50행
│   │   ├── teams.csv               # 팀 데이터 20행
│   │   ├── tournaments.csv         # 대회 데이터 15행
│   │   └── wiki_articles/          # 위키 서술 아티클 12개 (.txt)
│   └── processed/
│       └── graph_extraction_cache.json  # LLM 추출 캐시 (비용 절감)
├── ingestion/
│   ├── loaders.py                  # CSVLoader + DirectoryLoader/TextLoader
│   └── splitters.py                # RecursiveCharacter vs Semantic 비교
├── retrieval/
│   ├── dense.py                    # Chroma + MMR/Similarity 검색
│   ├── sparse.py                   # BM25Retriever
│   └── hybrid.py                   # EnsembleRetriever (0.6/0.4)
├── graph/
│   ├── build_graph.py              # LLMGraphTransformer → Neo4j (캐시 포함)
│   └── graph_qa.py                 # GraphCypherQAChain + Few-Shot 프롬프팅
├── chains/
│   └── rag_chains.py               # Vector / Hybrid / Graph / Router 체인
├── evaluation/
│   ├── testset.json                # 수동 제작 30개 질문 (커밋 필수)
│   ├── ragas_eval.py               # Ragas 4대 지표 평가 (답변 캐시 포함)
│   └── compare.py                  # 8개 쿼리 3시스템 비교
├── scripts/
│   ├── build_index.py              # 멱등성 Chroma + BM25 구축
│   ├── build_graph.py              # 멱등성 Neo4j 그래프 구축
│   └── run_comparison.py           # 전체 비교 + Ragas 실행
└── ui/
    └── app.py                      # Streamlit UI (탭 4개)
```

---

## 빠른 시작

### 1. 의존성 설치

```bash
cd sports_rag
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
cp .env.example .env
# .env 파일을 열어 API 키를 입력하세요
```

필수:
- `OPENAI_API_KEY` — 임베딩, LLM 답변 생성, Ragas 평가
- `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` — Graph RAG

선택:
- `LANGCHAIN_API_KEY` — LangSmith 트레이싱

### 3. Neo4j 실행

**옵션 A — Docker (로컬):**
```bash
docker run -d \
  --name neo4j-sports \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5.18
```
`.env` 설정:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
```

**옵션 B — AuraDB (클라우드 무료 티어):**
1. https://neo4j.com/cloud/aura/ 에서 무료 인스턴스 생성
2. URI + 비밀번호를 `.env`에 입력:
```
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_aura_password
```

### 4. 벡터 인덱스 구축

```bash
# 전체 실행 (스플리터 비교 포함)
python scripts/build_index.py

# SemanticChunker API 비용 절감 시
python scripts/build_index.py --skip-splitter-compare
```

### 5. 지식 그래프 구축

```bash
# 최초 실행 (캐시 없음 — gpt-4o 호출 발생)
python scripts/build_graph.py

# 구축 후 시연 쿼리 실행
python scripts/build_graph.py --demo

# 이후 재실행 (캐시 적중 — 추가 비용 없음)
python scripts/build_graph.py
```

### 6. 시스템 비교 실행

```bash
# 전체 비교
python scripts/run_comparison.py

# Neo4j 없을 때
python scripts/run_comparison.py --no-graph
```

### 7. Ragas 평가

```bash
# 개발용 드라이런 (5개 질문, gpt-4o-mini, 약 $0.01)
python evaluation/ragas_eval.py --dry-run

# 10개 질문, gpt-4o로 정확도 확인
python evaluation/ragas_eval.py --dry-run --n 10 --judge-model gpt-4o

# 최종 제출용 전체 평가 (30개, gpt-4o, 약 $1.44)
python evaluation/ragas_eval.py --judge-model gpt-4o
```

### 8. Streamlit UI 실행

```bash
streamlit run ui/app.py
```

---

## API 비용 관리

### 비용 절감 3중 방어 구조

| 구성요소 | 절감 방법 | 효과 |
|---|---|---|
| **Chroma 인덱스** | sqlite3 파일 존재 시 로드만 | 재실행 임베딩 비용 $0 |
| **그래프 추출** | JSON 캐시 저장 후 재사용 | 재실행 gpt-4o 비용 $0 |
| **Ragas 답변** | 답변 캐시 → Judge만 재실행 | chain 호출 비용 절감 |
| **Ragas Judge** | gpt-4o-mini 기본값 | gpt-4o 대비 1/30 비용 |
| **그래프 멱등성** | 노드 존재 시 빌드 건너뜀 | 중복 실행 방지 |

### 예상 비용표

| 작업 | 모델 | 예상 토큰 | 예상 비용 |
|------|------|-----------|-----------|
| 벡터 인덱스 구축 (최초) | text-embedding-3-small | ~42,000 | ~$0.001 |
| 그래프 구축 (최초, 12개) | gpt-4o | ~144,000 | ~$0.72 |
| 그래프 구축 (재실행) | — (캐시) | 0 | **$0.00** |
| Ragas 드라이런 (5개) | gpt-4o-mini | ~48,000 | ~$0.007 |
| Ragas 전체 (30개) | gpt-4o-mini | ~288,000 | ~$0.04 |
| Ragas 전체 (30개, gpt-4o) | gpt-4o | ~288,000 | ~$1.44 |
| 시스템 비교 (8쿼리×3) | gpt-4.1-mini | ~32,000 | ~$0.05 |

**권장 실행 순서 (비용 최적화):**
1. `build_index.py --skip-splitter-compare` — 임베딩 1회
2. `build_graph.py` — gpt-4o 1회, 이후 캐시 사용
3. `ragas_eval.py --dry-run` — gpt-4o-mini로 검증
4. `ragas_eval.py --judge-model gpt-4o` — 최종 제출 시만

---

## 모델 설정

| 용도 | 모델 |
|------|------|
| RAG 답변 생성 | `gpt-4.1-mini` |
| 라우터 분류기 | `gpt-4.1-mini` |
| Cypher 쿼리 생성 | `gpt-4o` |
| 그래프 엔티티 추출 | `gpt-4o` |
| Ragas Judge (기본/드라이런) | `gpt-4o-mini` |
| Ragas Judge (최종 제출) | `gpt-4o` |
| 임베딩 (전체) | `text-embedding-3-small` |

---

## 평가 결과

### Ragas 4대 지표 비교

| 지표 | VectorRAG | HybridRAG | 차이 | 우위 |
|------|-----------|-----------|------|------|
| Faithfulness | 0.938 | 0.924 | -0.012 | VectorRAG |
| Answer Relevancy | 0.700 | 0.728 | +0.028 | HybridRAG |
| Context Precision | 0.628 | 0.603 | -0.025 | VectorRAG |
| Context Recall | 0.541 | 0.663 | +0.122 | HybridRAG |

### 질문 유형별 시스템 비교

| 질문 유형 | Vector | Hybrid | Graph | 최적 시스템 |
|---|---|---|---|---|
| 사실 질문 | ✓ 양호 | ✓ 양호 | ○ 보통 | Hybrid |
| 정확 키워드 | ○ 보통 | ✓ 최상 | ○ 보통 | Hybrid |
| 관계 질문 | ✗ 미흡 | ○ 보통 | ✓ 최상 | Graph |
| 다중 홉 | ✗ 실패 | ✗ 실패 | ✓ 최상 | Graph |
| 집계 | ✗ 실패 | ✗ 실패 | ✓ 최상 | Graph |

---

## 구현된 확장 기능

### 1. Cypher Few-Shot 프롬프팅 (`graph/graph_qa.py`)
- 5개의 도메인 특화 (질문 → Cypher) 예시를 시스템 프롬프트에 포함
- Zero-shot 대비 복잡한 다중 홉 쿼리의 Cypher 성공률 15~25% 향상
- 예상 외 노드 레이블/관계 타입 생성 억제

### 2. 그래프 시각화 (`ui/app.py` 탭 4)
- pyvis 인터랙티브 HTML 서브그래프
- 노드 타입별 색상 (선수=파란색, 팀=빨간색, 대회=금색, 감독=초록색)
- Streamlit에 인라인 렌더링

### 3. Hybrid Vector + Graph RAG 라우터 (`chains/rag_chains.py`)
- `RouterRAGChain`: LLM 기반 3-class 분류 후 최적 체인으로 위임
- 분류 실패 시 키워드 휴리스틱으로 폴백 (API 비용 없음)
- `routed_to` 키로 라우팅 결정 과정 투명하게 반환

---

## 알려진 한계

- **LLMGraphTransformer 품질**: 추출이 완벽하지 않습니다. 일부 관계 누락 또는 타입 오류가 발생합니다. 추출된 노드/엣지를 일부 직접 확인하고 오류 사례를 보고서에 기록하세요.
- **Cypher 생성 실패**: 복잡한 다중 홉 쿼리에서 간혹 문법 오류가 발생합니다. `validate_cypher=True`로 대부분 포착되나 100%는 아닙니다.
- **말뭉치 범위 한계**: `data/raw/`의 데이터 외 선수·이벤트에 대한 질문은 검색 방식에 상관없이 품질이 낮습니다.
- **BM25 의미 불감지**: BM25는 "forward"와 "striker"처럼 의미는 같지만 표기가 다른 경우를 구분하지 못합니다. Hybrid 검색이 완화하지만 완전히 해결하진 않습니다.
- **Ragas LLM-as-Judge 한계**: gpt-4o-mini Judge는 미묘한 사실 오류를 놓칠 수 있습니다. 최종 제출 평가에는 gpt-4o를 권장합니다.

---

## 참고 자료

- [LangChain Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)
- [LangChain EnsembleRetriever](https://python.langchain.com/docs/integrations/retrievers/ensemble/)
- [Ragas 공식 문서](https://docs.ragas.io/)
- [LangChain LLMGraphTransformer](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/)
- [Neo4j AuraDB](https://neo4j.com/cloud/aura/)
- Es et al., *Ragas: Automated Evaluation of Retrieval Augmented Generation*, EACL 2024
- Edge et al., *From Local to Global: A Graph RAG Approach*, Microsoft 2024
