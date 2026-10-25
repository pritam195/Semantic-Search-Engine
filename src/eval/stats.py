"""Statistical significance testing and confidence interval estimation.

Provides:
- Paired Bootstrap Hypothesis Testing (p-values)
- Paired Student's t-test
- 95% Empirical Bootstrap Confidence Intervals
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy.stats as stats


def compute_confidence_interval(
    scores: Sequence[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """Compute empirical bootstrap confidence interval for a metric mean."""
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))

    rng = np.random.default_rng(random_seed)
    n = len(arr)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_means = np.mean(arr[indices], axis=1)

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - (alpha / 2.0))

    ci_lower = float(np.percentile(bootstrap_means, lower_pct))
    ci_upper = float(np.percentile(bootstrap_means, upper_pct))
    return ci_lower, ci_upper


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Perform a paired bootstrap hypothesis test comparing model A vs model B.

    Null Hypothesis (H0): mean(scores_a) == mean(scores_b)
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("scores_a and scores_b must have the exact same length")

    n = len(a)
    if n == 0:
        return {"p_value": 1.0, "mean_diff": 0.0, "significant": False}

    diffs = a - b
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(random_seed)
    # Center diffs under H0: mean(diffs) = 0
    centered_diffs = diffs - mean_diff

    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(centered_diffs[boot_indices], axis=1)

    # Two-sided empirical p-value
    p_value = float(np.mean(np.abs(boot_means) >= np.abs(mean_diff)))

    # Paired Student's t-test for cross-validation
    t_stat, t_p_value = stats.ttest_rel(a, b)

    return {
        "mean_diff": round(mean_diff, 5),
        "bootstrap_p_value": round(p_value, 5),
        "t_test_p_value": round(float(t_p_value), 5),
        "t_statistic": round(float(t_stat), 4) if not np.isnan(t_stat) else 0.0,
        "statistically_significant_05": p_value < 0.05,
        "statistically_significant_01": p_value < 0.01,
    }
