"""
TopoTrader V2 -- Graph Attention Signal (GAT-Inspired)
======================================================
Replaces the hand-crafted Graph Laplacian (V1) with an attention-weighted
graph aggregation signal.

Key improvements over V1 Laplacian:
  1. Adaptive threshold -- no more hardcoded 0.5.
     Threshold = percentile of current correlation distribution,
     so graph density stays stable even when all correlations spike in a crash.
  2. Attention-weighted aggregation -- instead of treating all edges equally
     (Laplacian), each edge weight is proportional to its correlation strength
     (normalised via softmax). Strong correlations get high attention.
  3. Signal interpretation is preserved: positive = stock above its weighted
     sector expectation (potential short), negative = below (potential long).

This is parameter-free (no training required), making it a drop-in
replacement for the Laplacian in the feature pipeline.
"""

import numpy as np


# -- Adaptive Threshold ---------------------------------------------------------

def get_adaptive_threshold(corr_matrix: np.ndarray, percentile: int = 70) -> float:
    """
    Compute an adaptive correlation threshold as the Nth percentile of the
    absolute correlation values in the upper triangle (excluding diagonal).

    This ensures the graph always keeps the top (100-percentile)% of edges,
    regardless of the current market correlation level.

    Args:
        corr_matrix: (N, N) symmetric correlation matrix.
        percentile: Edge inclusion threshold percentile (default 70 = keep top 30%).

    Returns:
        threshold: Scalar threshold value (minimum 0.10).
    """
    n = corr_matrix.shape[0]
    idx = np.triu_indices(n, k=1)
    upper_vals = np.abs(corr_matrix[idx])

    if len(upper_vals) == 0:
        return 0.5  # Fallback

    threshold = float(np.percentile(upper_vals, percentile))
    return max(threshold, 0.10)  # Floor at 0.10 to prevent fully connected graph


# -- Attention-Weighted Graph Signal -------------------------------------------

def get_gat_signal(returns_window: np.ndarray, percentile: int = 70) -> np.ndarray:
    """
    Attention-Weighted Graph Signal -- V2 replacement for the Laplacian residual.

    For each stock i:
      1. Build neighbourhood N(i) = {j : |corr(i,j)| > adaptive_threshold}
      2. Compute attention weights: a_ij = softmax(|corr(i,j)|) over j in N(i)
      3. Compute attention-weighted expected return: r̂_i = Σ_j a_ij * r_j
      4. Signal: s_i = r_i - r̂_i   (deviation from attention-weighted expectation)

    Positive signal -> stock above its attention-weighted sector -> potential short
    Negative signal -> stock below its attention-weighted sector -> potential long

    Args:
        returns_window: (N_assets, Window_Size) array of log returns.
        percentile: Adaptive threshold percentile (default 70).

    Returns:
        signals: (N_assets,) attention-weighted deviation vector.
    """
    n_assets = returns_window.shape[0]

    # -- Correlation matrix --------------------------------------------------
    corr = np.corrcoef(returns_window)
    np.nan_to_num(corr, copy=False)
    corr = np.clip(corr, -1.0, 1.0)

    # -- Adaptive threshold --------------------------------------------------
    threshold = get_adaptive_threshold(corr, percentile=percentile)

    # -- Weighted adjacency (absolute correlation above threshold) -----------
    adj = np.where(np.abs(corr) > threshold, np.abs(corr), 0.0)
    np.fill_diagonal(adj, 0.0)  # No self-loops

    if adj.sum() == 0:
        return np.zeros(n_assets)

    # -- Current return vector -----------------------------------------------
    r_t = returns_window[:, -1]

    # -- Attention-weighted aggregation for each stock -----------------------
    signals = np.zeros(n_assets)

    for i in range(n_assets):
        neighbour_weights = adj[i, :]  # (N_assets,) raw weights

        if neighbour_weights.sum() == 0:
            signals[i] = 0.0
            continue

        # Softmax attention (numerically stable)
        shifted    = neighbour_weights - neighbour_weights.max()
        exp_w      = np.exp(shifted)
        exp_w      = np.where(neighbour_weights > 0, exp_w, 0.0)  # Zero non-edges
        attention  = exp_w / (exp_w.sum() + 1e-12)

        # Attention-weighted sector expectation
        expected_return = float(np.dot(attention, r_t))

        # Deviation signal
        signals[i] = r_t[i] - expected_return

    return signals


# -- Convenience: both Laplacian + GAT for comparison --------------------------

def get_both_signals(returns_window: np.ndarray, percentile: int = 70):
    """
    Returns both the V1 Laplacian residual and V2 GAT signal for ablation studies.

    Returns:
        laplacian_signal: (N,) V1 signal
        gat_signal: (N,) V2 signal
    """
    from .graph_engine import get_laplacian_residuals
    lap = get_laplacian_residuals(returns_window, threshold=0.5)
    gat = get_gat_signal(returns_window, percentile=percentile)
    return lap, gat
