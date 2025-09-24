import numpy as np
import pandas as pd
from bayesflow.amortizers import AmortizedPosterior

PERCENTILES = [50, 90, 95]
COVERAGE_QUANTILES = {
    50: (0.25, 0.75),
    90: (0.05, 0.95),
    95: (0.025, 0.975)
}
ALPHAS = {
    50: 0.8,
    90: 0.5,
    95: 0.2
}

def compute_trajectory_bounds(trajectories, percentiles=PERCENTILES):
    """
    Given simulation trajectories of shape (samples, time), return a dict of (lower, upper) bounds per percentile.
    """
    bounds = {}
    for p in percentiles:
        q_low, q_high = COVERAGE_QUANTILES[p]
        bounds[p] = np.quantile(trajectories, q=[q_low, q_high], axis=0)
    return bounds

def compute_coverage(observed, bounds_dict):
    """
    Returns a dict {percentile: percent of observed points covered} based on bounds_dict.
    """
    coverage = {}
    for p, (lower, upper) in bounds_dict.items():
        in_bounds = (observed >= lower) & (observed <= upper)
        coverage[p] = np.mean(in_bounds) * 100  # Percent
    return coverage

def coverage_dataframe(obs1, obs2, bounds1, bounds2, prefix1="Infc", prefix2="Sprev"):
    """
    Combines coverage of two outputs into a single labeled DataFrame.
    """
    cov1 = compute_coverage(obs1, bounds1)
    cov2 = compute_coverage(obs2, bounds2)

    names = [f"${prefix1}_{{{p}}}$" for p in PERCENTILES] + [f"${prefix2}_{{{p}}}$" for p in PERCENTILES]
    data = [cov1[p] for p in PERCENTILES] + [cov2[p] for p in PERCENTILES]
    return pd.DataFrame([data], columns=names)

