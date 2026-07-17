"""
Statistical significance: with only a handful of golden cases running per
PR, a single unusually long or slow response could look like a "regression"
that's actually just noise. A paired Wilcoxon signed-rank test checks
whether the old-vs-new differences are consistently in one direction across
cases, rather than reacting to any single outlier.

Falls back to a simple paired t-test if scipy's Wilcoxon can't run (e.g. too
few non-zero differences, which happens with very small golden sets).
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy import stats

from src import config


def paired_significance_test(old_values: list[float], new_values: list[float]) -> dict:
    """Returns {is_significant, p_value, mean_delta, mean_delta_pct, test_used}."""
    old = np.array(old_values, dtype=float)
    new = np.array(new_values, dtype=float)
    deltas = new - old

    mean_old = float(np.mean(old)) if len(old) else 0.0
    mean_delta = float(np.mean(deltas))
    mean_delta_pct = (mean_delta / mean_old * 100) if mean_old else 0.0

    test_used = "wilcoxon"
    try:
        if np.count_nonzero(deltas) < 2:
            raise ValueError("Not enough non-zero deltas for Wilcoxon")
        _, p_value = stats.wilcoxon(new, old)
    except Exception:
        test_used = "paired_t_test"
        if len(deltas) < 2 or np.std(deltas) == 0:
            p_value = 0.0 if abs(mean_delta) > 1e-9 else 1.0
        else:
            _, p_value = stats.ttest_rel(new, old)

    return {
        "is_significant": bool(p_value < config.SIGNIFICANCE_ALPHA),
        "p_value": round(float(p_value), 5),
        "mean_delta": round(mean_delta, 6),
        "mean_delta_pct": round(mean_delta_pct, 2),
        "test_used": test_used,
    }
