"""
The evaluation gate orchestrates the entire prompt regression pipeline.

Workflow:
1. Compare the old and new prompts to understand what changed.
2. Retrieve only the most relevant golden test cases.
3. Reuse cached results whenever possible; otherwise evaluate both prompts.
4. Compare quality, cost, latency, and response length.
5. Perform statistical significance tests on the collected metrics.
6. Produce a final PASS/WARN/FAIL report with explanations.
"""

from dataclasses import dataclass, field

from src import config
from src.cache import ResultCache
from src.judge import judge_output
from src.runner import run_prompt
from src.stats import paired_significance_test
from src.test_bank_retriever import GoldenCaseRetriever, diff_prompts


@dataclass
class CaseResult:
    """Stores evaluation results for a single golden test case."""

    case_id: str
    question: str
    old_length: int
    new_length: int
    old_cost: float
    new_cost: float
    old_latency: float
    new_latency: float
    judge_score: float
    judge_reasoning: str
    from_cache: bool


@dataclass
class GateReport:
    """Final summary returned by the evaluation gate."""

    verdict: str  # PASS | WARN | FAIL

    # Human-readable explanations for the verdict.
    reasons: list[str] = field(default_factory=list)

    # Aggregated statistics across all evaluated cases.
    metric_summary: dict = field(default_factory=dict)

    # Detailed per-case evaluation results.
    case_results: list[dict] = field(default_factory=list)

    cases_run: int = 0
    cache_hits: int = 0


def run_gate(old_prompt_path: str, new_prompt_path: str) -> GateReport:
    """Run the complete prompt regression evaluation."""

    # Read both prompt versions from disk.
    with open(old_prompt_path, "r", encoding="utf-8") as f:
        old_prompt = f.read()

    with open(new_prompt_path, "r", encoding="utf-8") as f:
        new_prompt = f.read()

    # Retrieve only the golden cases that are relevant to the prompt changes.
    retriever = GoldenCaseRetriever()
    diff_query = diff_prompts(old_prompt, new_prompt)
    selected_cases = retriever.select_relevant_cases(diff_query)

    cache = ResultCache()
    case_results: list[CaseResult] = []
    cache_hits = 0

    # Evaluate each selected golden case.
    for case in selected_cases:

        # First try the semantic cache to avoid unnecessary API calls.
        cached = cache.get(new_prompt, case["question"])

        if cached:
            cache_hits += 1
            case_results.append(CaseResult(from_cache=True, **cached))
            continue

        # Run both prompt versions against the same question.
        old_run = run_prompt(old_prompt, case["question"])
        new_run = run_prompt(new_prompt, case["question"])

        # Ask the judge model to compare the new response against
        # the expected reference answer.
        judgment = judge_output(
            case["question"],
            case["expected_reference"],
            new_run.output_text,
        )

        # Collect all metrics needed for reporting.
        result_dict = {
            "case_id": case["id"],
            "question": case["question"],
            "old_length": len(old_run.output_text),
            "new_length": len(new_run.output_text),
            "old_cost": old_run.cost_usd,
            "new_cost": new_run.cost_usd,
            "old_latency": old_run.latency_sec,
            "new_latency": new_run.latency_sec,
            "judge_score": judgment["score"],
            "judge_reasoning": judgment["reasoning"],
        }

        # Save the result so future runs can reuse it.
        cache.set(new_prompt, case["question"], result_dict)

        case_results.append(
            CaseResult(
                from_cache=False,
                **result_dict,
            )
        )

    # Build the final PASS/WARN/FAIL report.
    return _build_report(case_results, cache_hits)


def _build_report(
    case_results: list[CaseResult],
    cache_hits: int,
) -> GateReport:
    """Aggregate metrics and determine the final gate verdict."""

    # No selected cases means the evaluation cannot produce a reliable result.
    if not case_results:
        return GateReport(
            verdict="FAIL",
            reasons=["No golden cases were selected or ran."],
        )

    # Compare old vs new metrics using paired statistical tests.
    length_stats = paired_significance_test(
        [c.old_length for c in case_results],
        [c.new_length for c in case_results],
    )

    cost_stats = paired_significance_test(
        [c.old_cost for c in case_results],
        [c.new_cost for c in case_results],
    )

    latency_stats = paired_significance_test(
        [c.old_latency for c in case_results],
        [c.new_latency for c in case_results],
    )

    # Track overall response quality.
    min_judge_score = min(c.judge_score for c in case_results)
    avg_judge_score = (
        sum(c.judge_score for c in case_results)
        / len(case_results)
    )

    reasons = []
    verdict = "PASS"

    # Fail if response length grows significantly beyond the allowed threshold.
    if (
        length_stats["is_significant"]
        and length_stats["mean_delta_pct"]
        > config.MAX_LENGTH_INCREASE_PCT
    ):
        verdict = "FAIL"
        reasons.append(
            f"Response length increased "
            f"{length_stats['mean_delta_pct']}% on average "
            f"(p={length_stats['p_value']}), exceeding the "
            f"{config.MAX_LENGTH_INCREASE_PCT}% threshold."
        )

    # Fail if token cost increases too much.
    if (
        cost_stats["is_significant"]
        and cost_stats["mean_delta_pct"]
        > config.MAX_COST_INCREASE_PCT
    ):
        verdict = "FAIL"
        reasons.append(
            f"Token cost increased "
            f"{cost_stats['mean_delta_pct']}% on average "
            f"(p={cost_stats['p_value']}), exceeding the "
            f"{config.MAX_COST_INCREASE_PCT}% threshold."
        )

    # Latency regressions are treated as warnings unless another failure exists.
    if (
        latency_stats["is_significant"]
        and latency_stats["mean_delta_pct"]
        > config.MAX_LATENCY_INCREASE_PCT
    ):
        verdict = "WARN" if verdict == "PASS" else verdict

        reasons.append(
            f"Latency increased "
            f"{latency_stats['mean_delta_pct']}% on average "
            f"(p={latency_stats['p_value']}), exceeding the "
            f"{config.MAX_LATENCY_INCREASE_PCT}% threshold."
        )

    # A low judge score indicates the new prompt produced
    # lower-quality or less faithful responses.
    if min_judge_score < config.MIN_JUDGE_SCORE:

        verdict = "FAIL"

        # Highlight the weakest-performing test case.
        worst = min(
            case_results,
            key=lambda c: c.judge_score,
        )

        reasons.append(
            f"Faithfulness regression: case '{worst.case_id}' "
            f"scored {worst.judge_score} "
            f"(< {config.MIN_JUDGE_SCORE} threshold). "
            f"Reason: {worst.judge_reasoning}"
        )

    # If no regressions were detected, report success.
    if not reasons:
        reasons.append(
            "All metrics are within thresholds and no "
            "statistically significant regressions were detected."
        )

    return GateReport(
        verdict=verdict,
        reasons=reasons,
        metric_summary={
            "length": length_stats,
            "cost": cost_stats,
            "latency": latency_stats,
            "avg_judge_score": round(avg_judge_score, 3),
            "min_judge_score": round(min_judge_score, 3),
        },
        case_results=[c.__dict__ for c in case_results],
        cases_run=len(case_results),
        cache_hits=cache_hits,
    )