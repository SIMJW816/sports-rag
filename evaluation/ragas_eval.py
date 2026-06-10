"""
evaluation/ragas_eval.py

Ragas 평가 모듈 — RAG 시스템 품질을 4가지 지표로 측정합니다.

  1. Faithfulness      — 답변이 검색된 컨텍스트에 충실한가? (환각 감지)
  2. Answer Relevance  — 답변이 질문에 실제로 관련이 있는가?
  3. Context Precision — 상위 검색 문서가 진짜로 유용한가?
  4. Context Recall    — 정답 도출에 필요한 정보가 검색되었는가?

비용 관리 전략:
  - --dry-run 플래그: 5개 질문으로만 테스트 (개발 단계용, 비용 1/6)
  - --judge-model 옵션: gpt-4o(정확) vs gpt-4o-mini(저렴, 약 1/30 비용)
  - 답변 캐시: 동일 시스템의 chain.invoke() 결과를 JSON으로 저장,
    재평가 시 LLM 호출 없이 캐시에서 로드
  - 실행 전 비용 추정 출력 후 사용자 확인 요청

사용법:
  python evaluation/ragas_eval.py --dry-run              # 5개 질문, 저렴한 모델
  python evaluation/ragas_eval.py --dry-run --judge-model gpt-4o
  python evaluation/ragas_eval.py                        # 전체 30개, gpt-4o
  python evaluation/ragas_eval.py --system vector        # VectorRAG만 평가
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Python 3.12+ asyncio 호환성 패치 ─────────────────────────────────────────
# Ragas의 executor가 Python 3.12+에서 "Timeout should be used inside a task"
# RuntimeError를 발생시키는 문제를 nest_asyncio로 우회한다.
# nest_asyncio가 없으면 경고만 출력하고 계속 진행한다.
try:
    import nest_asyncio
    nest_asyncio.apply()
    logger.debug("nest_asyncio 적용 완료 (Python 3.12+ asyncio 호환)")
except ImportError:
    logger.debug("nest_asyncio 미설치 — pip install nest_asyncio 권장 (Python 3.12+)")
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"
TESTSET_PATH = Path(__file__).parent / "testset.json"

# 답변 캐시 경로: chain.invoke() 결과를 저장해 재평가 비용 절감
ANSWER_CACHE_DIR = RESULTS_DIR / "answer_cache"

# 기본 Judge LLM — --judge-model 옵션으로 변경 가능
# gpt-4o-mini: 비용 1/30, 정확도 약간 낮음 (개발용 권장)
# gpt-4o    : 최고 정확도 (최종 제출용 권장)
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
FINAL_JUDGE_MODEL = "gpt-4o"


# ──────────────────────────────────────────────────────────────────────────────
# 비용 추정 출력
# ──────────────────────────────────────────────────────────────────────────────
def _estimate_cost(n_questions: int, judge_model: str, n_metrics: int = 4) -> float:
    """
    평가 실행 전 예상 API 비용을 출력합니다.

    Ragas는 지표당 질문 1개에 LLM을 약 3회 호출합니다.
    gpt-4o   : 입력 $5/M, 출력 $15/M 토큰
    gpt-4o-mini: 입력 $0.15/M, 출력 $0.60/M 토큰

    매개변수:
        n_questions: 평가할 질문 수
        judge_model: 사용할 Judge LLM 모델명
        n_metrics: 측정할 Ragas 지표 수 (기본 4)

    반환값:
        예상 비용 (달러)
    """
    llm_calls = n_questions * n_metrics * 3
    est_tokens = llm_calls * 800  # 질문당 평균 800 토큰

    # 모델별 단가 (입력 기준)
    price_per_token = {
        "gpt-4o": 0.000005,
        "gpt-4o-mini": 0.00000015,
        "gpt-4.1-mini": 0.0000004,
    }
    price = price_per_token.get(judge_model, 0.000005)
    est_cost = est_tokens * price

    print("\n" + "=" * 58)
    print(f"  Ragas 평가 비용 추정")
    print(f"  {'질문 수':<20}: {n_questions}개")
    print(f"  {'지표 수':<20}: {n_metrics}개")
    print(f"  {'Judge LLM':<20}: {judge_model}")
    print(f"  {'예상 LLM 호출 수':<20}: {llm_calls}회")
    print(f"  {'예상 토큰':<20}: {est_tokens:,}")
    print(f"  {'예상 비용':<20}: ${est_cost:.4f}")
    if judge_model != FINAL_JUDGE_MODEL:
        final_cost = est_tokens * price_per_token.get(FINAL_JUDGE_MODEL, 0.000005)
        print(f"  (gpt-4o 기준 비교) : ${final_cost:.4f}")
    print("=" * 58 + "\n")
    return est_cost


# ──────────────────────────────────────────────────────────────────────────────
# 답변 캐시 (체인 호출 결과 저장 → 재평가 시 LLM 호출 없음)
# ──────────────────────────────────────────────────────────────────────────────
def _cache_path(system_name: str) -> Path:
    """시스템별 답변 캐시 파일 경로를 반환합니다."""
    ANSWER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = system_name.lower().replace(" ", "_")
    return ANSWER_CACHE_DIR / f"answers_{safe_name}.json"


def _load_answer_cache(system_name: str) -> Dict[str, Dict]:
    """
    이전에 저장된 답변 캐시를 불러옵니다.

    반환값:
        {question_id: {"answer": str, "contexts": List[str]}}
    """
    path = _cache_path(system_name)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("[답변 캐시] %s: %d개 로드 (chain 호출 생략)", path.name, len(data))
        return data
    return {}


def _save_answer_cache(system_name: str, cache: Dict[str, Dict]) -> None:
    """답변 캐시를 JSON 파일로 저장합니다."""
    path = _cache_path(system_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.info("[답변 캐시] %d개 저장 → %s", len(cache), path)


# ──────────────────────────────────────────────────────────────────────────────
# 테스트셋 로더
# ──────────────────────────────────────────────────────────────────────────────
def load_testset(path: Path = TESTSET_PATH) -> List[Dict[str, Any]]:
    """
    testset.json에서 수동 제작 테스트셋을 불러옵니다.

    매개변수:
        path: testset.json 파일 경로

    반환값:
        질문 딕셔너리 리스트
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("테스트셋 로드: %d개 질문 (%s)", len(data), path)
    return data


# ──────────────────────────────────────────────────────────────────────────────
# 핵심 평가 함수
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_rag_system(
    chain,
    testset: List[Dict[str, Any]],
    system_name: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    use_cache: bool = True,
    force_reinvoke: bool = False,
) -> pd.DataFrame:
    """
    RAG 체인에 대해 Ragas 4대 지표를 모두 측정합니다.

    비용 절감 장치:
      1. 답변 캐시: 이미 호출한 chain.invoke() 결과는 JSON으로 저장.
         재실행 시 캐시에서 로드하여 chain LLM 비용 0원.
      2. judge_model: gpt-4o-mini(기본)와 gpt-4o 중 선택 가능.
         최종 제출 직전에만 gpt-4o로 교체 권장.

    각 질문에 대해:
      1. 캐시 확인 → 없으면 chain.invoke(question)
      2. Ragas SingleTurnSample 생성
      3. 4가지 지표 계산 (LLM-as-Judge)

    매개변수:
        chain: BaseRAGChain 서브클래스 인스턴스
        testset: testset.json에서 로드한 질문 딕셔너리 리스트
        system_name: 결과 파일명에 사용할 레이블 (예: "VectorRAG")
        judge_model: Ragas 평가에 사용할 LLM 모델명
        use_cache: True이면 이전 답변 캐시 활용
        force_reinvoke: True이면 캐시 무시하고 chain 재호출

    반환값:
        question_id, question_type, difficulty, 4대 지표 점수가 담긴 DataFrame.
        마지막 행에 "MEAN" 집계 행 포함.
    """
    # ── ragas 버전별 호환 임포트 ───────────────────────────────────────────────
    # ragas 0.1.x (안정): datasets.Dataset 기반 API 사용
    # ragas 0.2.x (최신): EvaluationDataset, SingleTurnSample 사용
    # 설치된 버전을 자동 감지하여 적절한 API 선택
    try:
        import ragas as _ragas_pkg
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        _ver = tuple(int(x) for x in _ragas_pkg.__version__.split(".")[:2])
        logger.info("ragas 버전 감지: %s", _ragas_pkg.__version__)

        if _ver >= (0, 2):
            # ragas 0.2.x API
            from ragas import evaluate, EvaluationDataset
            from ragas.metrics import (
                faithfulness, answer_relevancy,
                context_precision, context_recall,
            )
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.dataset_schema import SingleTurnSample

            judge_llm = LangchainLLMWrapper(
                ChatOpenAI(model=judge_model, temperature=0)
            )
            judge_embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model="text-embedding-3-small")
            )
            metrics = [faithfulness, answer_relevancy,
                       context_precision, context_recall]
            for m in metrics:
                m.llm = judge_llm
                if hasattr(m, "embeddings"):
                    m.embeddings = judge_embeddings
            USE_NEW_API = True

        else:
            # ragas 0.1.x API — datasets.Dataset 방식 사용
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness, answer_relevancy,
                context_precision, context_recall,
            )
            try:
                from datasets import Dataset as HFDataset
            except ImportError:
                raise ImportError(
                    "ragas 0.1.x 사용 시 datasets 패키지 필요: pip install datasets"
                )
            metrics = [faithfulness, answer_relevancy,
                       context_precision, context_recall]
            USE_NEW_API = False
            EvaluationDataset = None
            SingleTurnSample  = None

    except ImportError as e:
        logger.error("ragas 임포트 실패: %s", e)
        raise

    print(f"\n{'='*58}")
    print(f"  평가 시스템 : {system_name}  ({len(testset)}개 질문)")
    print(f"  Judge LLM  : {judge_model}")
    print(f"{'='*58}")

    # 답변 캐시 로드
    answer_cache = _load_answer_cache(system_name) if (use_cache and not force_reinvoke) else {}
    new_cache_entries: Dict[str, Dict] = {}

    samples = []
    rows = []
    cache_hits = 0

    for i, item in enumerate(testset, 1):
        q_id = item["id"]
        q = item["question"]
        gt = item["ground_truth"]

        # ── 캐시 우선 확인 ──────────────────────────────────────────────────
        if q_id in answer_cache and not force_reinvoke:
            cached = answer_cache[q_id]
            answer = cached["answer"]
            contexts = cached["contexts"]
            cache_hits += 1
            print(f"  [{i:02d}/{len(testset)}] [캐시] {q[:55]}...")
        else:
            # ── chain 호출 (LLM 비용 발생) ──────────────────────────────────
            print(f"  [{i:02d}/{len(testset)}] [호출] {q[:55]}...")
            try:
                result = chain.invoke(q)
                answer = result.get("answer", "")
                contexts = result.get("contexts", [])
                if not contexts:
                    contexts = ["검색된 컨텍스트 없음."]
            except Exception as exc:
                logger.error("chain.invoke 실패 [%s]: %s", q_id, exc)
                answer = f"오류: {exc}"
                contexts = ["검색 중 오류 발생."]

            # 새 답변을 캐시에 추가
            new_cache_entries[q_id] = {"answer": answer, "contexts": contexts}

        samples.append({
            "user_input": q,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": gt,
        })
        rows.append({
            "question_id": q_id,
            "question": q,
            "question_type": item.get("question_type", ""),
            "difficulty": item.get("difficulty", ""),
            "answer": answer,
        })

    # 새로 호출한 답변을 캐시에 병합 저장 (다음 실행 시 재사용)
    if new_cache_entries and use_cache:
        merged = {**answer_cache, **new_cache_entries}
        _save_answer_cache(system_name, merged)

    print(f"\n  캐시 적중: {cache_hits}/{len(testset)}개 (chain 호출 절감)")
    print(f"  Ragas 지표 계산 중 (Judge: {judge_model}) ...")

    # ── ragas 버전에 따라 다른 데이터셋 구성 및 evaluate 호출 ──────────────────
    import numpy as np

    try:
        if USE_NEW_API:
            # ragas 0.2.x: EvaluationDataset + SingleTurnSample
            ragas_samples = [
                SingleTurnSample(
                    user_input=s["user_input"],
                    response=s["response"],
                    retrieved_contexts=s["retrieved_contexts"],
                    reference=s["reference"],
                )
                for s in samples
            ]
            dataset = EvaluationDataset(samples=ragas_samples)
            eval_result = evaluate(dataset=dataset, metrics=metrics)
            scores_df = eval_result.to_pandas()

        else:
            # ragas 0.1.x: HFDataset 딕셔너리 방식
            # 컬럼명이 0.1.x API와 정확히 일치해야 함
            hf_dict = {
                "question":   [s["user_input"]          for s in samples],
                "answer":     [s["response"]            for s in samples],
                "contexts":   [s["retrieved_contexts"]  for s in samples],
                "ground_truth":[s["reference"]          for s in samples],
            }
            dataset = HFDataset.from_dict(hf_dict)
            eval_result = evaluate(dataset, metrics=metrics)
            scores_df = eval_result.to_pandas()
            # 0.1.x 컬럼명 → 공통 컬럼명으로 정규화
            col_map = {
                "answer_relevancy": "answer_relevancy",
                "context_precision": "context_precision",
                "context_recall": "context_recall",
                "faithfulness": "faithfulness",
            }
            scores_df = scores_df.rename(columns=col_map)

        logger.info("Ragas 평가 완료.")

    except Exception as exc:
        logger.error("Ragas 평가 실패: %s", exc)
        logger.warning("모든 지표를 NaN으로 처리합니다.")
        scores_df = pd.DataFrame(
            {col: [float("nan")] * len(samples)
             for col in ["faithfulness", "answer_relevancy",
                         "context_precision", "context_recall"]}
        )

    metric_cols = ["faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"]

    # 메타데이터와 점수 병합
    meta_df = pd.DataFrame(rows)
    final_df = meta_df.copy()

    for col in metric_cols:
        if col in scores_df.columns:
            final_df[col] = pd.to_numeric(scores_df[col].values, errors="coerce")
        else:
            final_df[col] = float("nan")

    # ── NaN 안전 집계 ──────────────────────────────────────────────────────────
    # NaN이 섞여 있어도 nanmean으로 계산하고, 전부 NaN이면 0.0으로 처리한다.
    import numpy as np

    mean_row = {
        "question_id": "MEAN",
        "question": "— 전체 평균 —",
        "question_type": "all",
        "difficulty": "all",
        "answer": "",
    }
    for col in metric_cols:
        raw_vals = final_df[col].values.astype(float)
        mean_val = float(np.nanmean(raw_vals)) if not np.all(np.isnan(raw_vals)) else 0.0
        mean_row[col] = round(mean_val, 4)

    final_df = pd.concat([final_df, pd.DataFrame([mean_row])], ignore_index=True)

    # CSV 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ragas_{system_name.lower().replace(' ', '_')}.csv"
    final_df.to_csv(out_path, index=False)

    # 결과 요약 출력 (NaN 안전)
    print(f"\n  {'지표':<25} {'점수':>8}")
    print(f"  {'-'*35}")
    all_nan = True
    for col in metric_cols:
        val = mean_row[col]
        if not np.isnan(val):
            all_nan = False
            bar = "█" * int(val * 20)
            print(f"  {col:<25} {val:>6.4f}  {bar}")
        else:
            print(f"  {col:<25}    NaN  (평가 실패 — 아래 오류 확인)")

    if all_nan:
        print("\n  ⚠️  모든 지표가 NaN입니다.")
        print("  Python 3.12+ 와 Ragas 버전 호환 문제일 수 있습니다.")
        print("  해결 방법:")
        print("    1) pip install 'ragas>=0.2.0' --upgrade")
        print("    2) pip install nest_asyncio  # Jupyter/Streamlit 환경")
        print("    3) Python 3.11 가상환경 사용 권장")

    print(f"\n  결과 저장 → {out_path}")
    return final_df


# ──────────────────────────────────────────────────────────────────────────────
# 두 가지 구성 비교
# ──────────────────────────────────────────────────────────────────────────────
def compare_two_configs(
    testset: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> pd.DataFrame:
    """
    Config A (VectorRAG) vs Config B (HybridRAG)를 Ragas로 비교합니다.

    두 체인을 각각 evaluate_rag_system으로 평가한 후,
    결과를 나란히 비교하는 표와 matplotlib 시각화를 생성합니다.

    매개변수:
        testset: 질문 딕셔너리 리스트
        judge_model: 평가에 사용할 LLM 모델명

    반환값:
        두 구성의 점수를 나란히 담은 비교 DataFrame
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from ingestion.loaders import load_all_documents
    from ingestion.splitters import get_production_chunks
    from retrieval.dense import build_or_load_vectorstore
    from retrieval.sparse import build_bm25_retriever, load_bm25_retriever, BM25_PICKLE_PATH
    from retrieval.hybrid import build_hybrid_retriever
    from chains.rag_chains import VectorRAGChain, HybridRAGChain

    docs = load_all_documents()
    chunks = get_production_chunks(docs)
    vs = build_or_load_vectorstore(chunks)

    try:
        bm25 = load_bm25_retriever()
    except FileNotFoundError:
        from retrieval.sparse import save_bm25_retriever
        bm25 = build_bm25_retriever(chunks)
        save_bm25_retriever(bm25)

    ensemble = build_hybrid_retriever(vs, bm25)
    vector_chain = VectorRAGChain(vs)
    hybrid_chain = HybridRAGChain(ensemble)

    df_vector = evaluate_rag_system(vector_chain, testset, "VectorRAG",
                                    judge_model=judge_model)
    df_hybrid = evaluate_rag_system(hybrid_chain, testset, "HybridRAG",
                                    judge_model=judge_model)

    metric_cols = ["faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"]

    v_mean = df_vector[df_vector["question_id"] == "MEAN"][metric_cols].iloc[0]
    h_mean = df_hybrid[df_hybrid["question_id"] == "MEAN"][metric_cols].iloc[0]

    comparison = pd.DataFrame({
        "metric": metric_cols,
        "VectorRAG": v_mean.values,
        "HybridRAG": h_mean.values,
    })
    comparison["delta"] = comparison["HybridRAG"] - comparison["VectorRAG"]
    comparison["winner"] = comparison.apply(
        lambda r: "HybridRAG" if r["delta"] > 0 else "VectorRAG", axis=1
    )

    # 비교 표 출력
    print("\n" + "=" * 65)
    print(f"{'Ragas 비교: VectorRAG vs HybridRAG':^65}")
    print("=" * 65)
    print(f"  {'지표':<25} {'Vector':>8} {'Hybrid':>8} {'차이':>8} {'승자':<12}")
    print("-" * 65)
    for _, row in comparison.iterrows():
        delta_str = f"{row['delta']:+.4f}"
        print(f"  {row['metric']:<25} {row['VectorRAG']:>8.4f} "
              f"{row['HybridRAG']:>8.4f} {delta_str:>8} {row['winner']:<12}")
    print("=" * 65)

    # ASCII 막대 그래프 (각 █ = 0.05)
    print("\n  ASCII 막대 그래프 (각 █ = 0.05)")
    print(f"  {'지표':<25} {'Vector':<22} {'Hybrid':<22}")
    print("-" * 72)
    for _, row in comparison.iterrows():
        v_bar = "█" * int(row["VectorRAG"] * 20)
        h_bar = "█" * int(row["HybridRAG"] * 20)
        print(f"  {row['metric']:<25} {v_bar:<22} {h_bar:<22}")

    # CSV 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comp_path = RESULTS_DIR / "ragas_comparison.csv"
    comparison.to_csv(comp_path, index=False)
    print(f"\n  비교 결과 저장 → {comp_path}")

    # 시각화 생성
    plot_ragas_results(comparison, str(RESULTS_DIR / "ragas_comparison.png"))

    return comparison


# ──────────────────────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────────────────────
def plot_ragas_results(df: pd.DataFrame, output_path: str) -> None:
    """
    그룹 막대 그래프와 레이더 차트를 담은 matplotlib 그림을 생성합니다.

    매개변수:
        df: [metric, VectorRAG, HybridRAG] 열을 가진 DataFrame
        output_path: PNG 저장 경로
    """
    import numpy as np

    metrics = df["metric"].tolist()
    vector_scores = df["VectorRAG"].tolist()
    hybrid_scores = df["HybridRAG"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Ragas 평가: VectorRAG vs HybridRAG",
                 fontsize=14, fontweight="bold")

    # ── 그룹 막대 그래프 ────────────────────────────────────────────────────
    ax1 = axes[0]
    x = range(len(metrics))
    width = 0.35
    bars1 = ax1.bar([i - width/2 for i in x], vector_scores,
                    width, label="VectorRAG", color="#4A90D9", alpha=0.85)
    bars2 = ax1.bar([i + width/2 for i in x], hybrid_scores,
                    width, label="HybridRAG", color="#E74C3C", alpha=0.85)

    ax1.set_xlabel("지표")
    ax1.set_ylabel("점수 (0–1)")
    ax1.set_title("4대 지표 비교")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # ── 레이더 차트 ─────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.remove()
    ax_radar = fig.add_subplot(1, 2, 2, polar=True)

    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 원 닫기

    v_vals = vector_scores + vector_scores[:1]
    h_vals = hybrid_scores + hybrid_scores[:1]

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)

    ax_radar.plot(angles, v_vals, "o-", linewidth=2,
                  label="VectorRAG", color="#4A90D9")
    ax_radar.fill(angles, v_vals, alpha=0.15, color="#4A90D9")
    ax_radar.plot(angles, h_vals, "o-", linewidth=2,
                  label="HybridRAG", color="#E74C3C")
    ax_radar.fill(angles, h_vals, alpha=0.15, color="#E74C3C")
    ax_radar.set_title("레이더 프로파일", pad=20)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장 → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI 진입점
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Sports RAG 시스템에 대한 Ragas 평가를 실행합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 개발 단계: 5개 질문, 저렴한 모델로 빠른 검증
  python evaluation/ragas_eval.py --dry-run

  # 개발 단계: 10개 질문, gpt-4o로 정확도 확인
  python evaluation/ragas_eval.py --dry-run --n 10 --judge-model gpt-4o

  # 최종 제출용: 30개 전체, gpt-4o
  python evaluation/ragas_eval.py --judge-model gpt-4o

  # VectorRAG만, 캐시 강제 갱신
  python evaluation/ragas_eval.py --system vector --force-reinvoke
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="처음 5개 질문만 평가 (개발·비용 절감용).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="--dry-run 시 평가할 질문 수 (기본: 5).",
    )
    parser.add_argument(
        "--system",
        choices=["vector", "hybrid", "both"],
        default="both",
        help="평가할 시스템 (기본: both).",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        choices=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        help=f"Ragas Judge LLM 모델 (기본: {DEFAULT_JUDGE_MODEL}). "
             "최종 제출 시 gpt-4o 권장.",
    )
    parser.add_argument(
        "--force-reinvoke",
        action="store_true",
        help="답변 캐시를 무시하고 chain을 다시 호출합니다.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    # 테스트셋 로드
    testset = load_testset()
    if args.dry_run:
        testset = testset[:args.n]
        print(f"\n  [DRY-RUN] {len(testset)}개 질문만 평가합니다.")

    # 비용 추정 출력
    est = _estimate_cost(len(testset), args.judge_model)

    # 30개 이상 + gpt-4o 조합이면 사용자 확인 요청
    if len(testset) >= 20 and args.judge_model == FINAL_JUDGE_MODEL:
        answer = input(
            f"  예상 비용 ${est:.3f}. 계속하려면 'yes' 입력: "
        ).strip().lower()
        if answer != "yes":
            print("  평가를 취소했습니다.")
            return

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from ingestion.loaders import load_all_documents
    from ingestion.splitters import get_production_chunks
    from retrieval.dense import build_or_load_vectorstore
    from retrieval.sparse import build_bm25_retriever, load_bm25_retriever, BM25_PICKLE_PATH
    from retrieval.hybrid import build_hybrid_retriever
    from chains.rag_chains import VectorRAGChain, HybridRAGChain

    docs = load_all_documents()
    chunks = get_production_chunks(docs)
    vs = build_or_load_vectorstore(chunks)

    try:
        bm25 = load_bm25_retriever()
    except FileNotFoundError:
        from retrieval.sparse import save_bm25_retriever
        bm25 = build_bm25_retriever(chunks)
        save_bm25_retriever(bm25)

    ensemble = build_hybrid_retriever(vs, bm25)

    if args.system in ("vector", "both"):
        v_chain = VectorRAGChain(vs)
        evaluate_rag_system(v_chain, testset, "VectorRAG",
                            judge_model=args.judge_model,
                            force_reinvoke=args.force_reinvoke)

    if args.system in ("hybrid", "both"):
        h_chain = HybridRAGChain(ensemble)
        evaluate_rag_system(h_chain, testset, "HybridRAG",
                            judge_model=args.judge_model,
                            force_reinvoke=args.force_reinvoke)

    # 두 구성 비교 차트 생성 (both 모드, 전체 평가 완료 후)
    if args.system == "both":
        v_path = RESULTS_DIR / "ragas_vectorrag.csv"
        h_path = RESULTS_DIR / "ragas_hybridrag.csv"
        if v_path.exists() and h_path.exists():
            v_df = pd.read_csv(v_path)
            h_df = pd.read_csv(h_path)
            metric_cols = ["faithfulness", "answer_relevancy",
                           "context_precision", "context_recall"]
            v_mean = v_df[v_df["question_id"] == "MEAN"][metric_cols].iloc[0]
            h_mean = h_df[h_df["question_id"] == "MEAN"][metric_cols].iloc[0]
            comp = pd.DataFrame({
                "metric": metric_cols,
                "VectorRAG": v_mean.values,
                "HybridRAG": h_mean.values,
            })
            plot_ragas_results(comp, str(RESULTS_DIR / "ragas_comparison.png"))
            print("\n  최종 비교 차트 생성 완료.")


if __name__ == "__main__":
    main()
