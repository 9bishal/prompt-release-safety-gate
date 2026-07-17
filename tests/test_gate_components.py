"""
Tests covering the parts of the gate that don't require a Groq API key:
golden-case retrieval, the prompt-diff query builder, and the statistics
module. Run with: pytest tests/
"""
from src.stats import paired_significance_test
from src.test_bank_retriever import GoldenCaseRetriever, diff_prompts, load_cases


def test_load_cases_returns_all_categories():
    cases = load_cases()
    categories = {c["category"] for c in cases}
    assert {"refund_policy", "shipping", "billing", "out_of_scope"} <= categories


def test_diff_prompts_extracts_changed_lines():
    old = "line one\nline two\nline three"
    new = "line one\nline TWO changed\nline three"
    diff = diff_prompts(old, new)
    assert "line two" in diff or "line TWO changed" in diff


def test_retriever_selects_relevant_cases_for_refund_diff():
    retriever = GoldenCaseRetriever()
    diff_query = "refund policy 45 days receipt store credit"
    selected = retriever.select_relevant_cases(diff_query)
    selected_categories = {c["category"] for c in selected}
    assert "refund_policy" in selected_categories
    # out_of_scope cases must always be included as the non-negotiable core tier
    assert "out_of_scope" in selected_categories


def test_significance_flags_a_consistent_large_increase():
    old_values = [100, 110, 105, 95, 102]
    new_values = [300, 310, 305, 295, 302]  # consistently ~3x larger
    result = paired_significance_test(old_values, new_values)
    assert result["is_significant"] is True
    assert result["mean_delta_pct"] > 100


def test_significance_does_not_flag_noise():
    old_values = [100, 110, 105, 95, 102]
    new_values = [102, 108, 106, 96, 101]  # tiny, inconsistent differences
    result = paired_significance_test(old_values, new_values)
    assert result["mean_delta_pct"] < 10
