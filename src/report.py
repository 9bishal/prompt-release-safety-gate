"""Turns a GateReport into a markdown string suitable for posting as a PR comment."""
from src.gate import GateReport

VERDICT_EMOJI = {
    "PASS": "✅",
    "WARN": "⚠️",
    "FAIL": "❌"
}


def render_markdown(report: GateReport) -> str:
    lines = [
        f"## {VERDICT_EMOJI.get(report.verdict, '')} Prompt Release Safety Gate: {report.verdict}",
        "",
        f"Ran **{report.cases_run}** golden cases ({report.cache_hits} served from semantic cache).",
        "",
        "### Reasons",
    ]
    for r in report.reasons:
        lines.append(f"- {r}")

    lines += ["", "### Metric deltas (new vs old, averaged across cases)", "",
              "| Metric | Mean delta | Mean delta % | p-value | Significant? |",
              "|---|---|---|---|---|"]
    for name, key in [("Response length (chars)", "length"), ("Cost (USD)", "cost"), ("Latency (sec)", "latency")]:
        m = report.metric_summary.get(key, {})
        lines.append(
            f"| {name} | {m.get('mean_delta', '-')} | {m.get('mean_delta_pct', '-')}% "
            f"| {m.get('p_value', '-')} | {'Yes' if m.get('is_significant') else 'No'} |"
        )

    lines += [
        "",
        f"**Avg faithfulness (judge) score:** {report.metric_summary.get('avg_judge_score', '-')}  ",
        f"**Min faithfulness (judge) score:** {report.metric_summary.get('min_judge_score', '-')}",
        "",
        "### Per-case results",
        "",
        "| Case | Old len | New len | Old cost | New cost | Judge score | Cached? |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in report.case_results:
        lines.append(
            f"| {c['case_id']} | {c['old_length']} | {c['new_length']} | "
            f"${c['old_cost']:.6f} | ${c['new_cost']:.6f} | {c['judge_score']} | "
            f"{'yes' if c['from_cache'] else 'no'} |"
        )

    return "\n".join(lines)
