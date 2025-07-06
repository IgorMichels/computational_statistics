import argparse
import os
import time
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class State:
    """Represents the state of the Gibbs sampler.

    Attributes:
        z: Cluster assignments for each data point.
        pi: Mixture weights for each component.
        mu: Mean parameters for each component.
    """

    z: np.ndarray
    pi: np.ndarray
    mu: np.ndarray


def relabel(state: State) -> State:
    """Relabel the state by sorting components by their means.

    This function ensures identifiability by ordering components
    according to their mean values.

    Args:
        state: Current state of the sampler.

    Returns:
        Relabeled state with components sorted by mean.
    """
    order = np.argsort(state.mu)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return State(inv[state.z], state.pi[order], state.mu[order])


def sample_z(y: np.ndarray, state: State) -> np.ndarray:
    """Sample cluster assignments for each data point.

    Uses the current values of pi and mu to compute posterior
    probabilities and sample new cluster assignments.

    Args:
        y: Observed data points.
        state: Current state containing pi and mu values.

    Returns:
        New cluster assignments for each data point.
    """
    K = len(state.pi)
    logw = np.empty((len(y), K))
    for k in range(K):
        log_prior = np.log(state.pi[k])
        log_like = -0.5 * (y - state.mu[k]) ** 2 - 0.5 * np.log(2 * np.pi)
        logw[:, k] = log_prior + log_like
    logw -= logw.max(axis=1, keepdims=True)  # pylint: disable=unexpected-keyword-arg
    probs = np.exp(logw)
    probs /= probs.sum(axis=1, keepdims=True)
    u = np.random.rand(len(y), 1)
    return (np.cumsum(probs, axis=1) > u).argmax(axis=1)


def sample_pi(z: np.ndarray, K: int, alpha: float = 1.0) -> np.ndarray:
    """Sample mixture weights from Dirichlet distribution.

    Args:
        z: Current cluster assignments.
        K: Number of mixture components.
        alpha: Dirichlet concentration parameter.

    Returns:
        New mixture weights sampled from posterior Dirichlet.
    """
    return np.random.dirichlet(alpha + np.bincount(z, minlength=K))


def sample_mu(
    y: np.ndarray, z: np.ndarray, K: int, m0: float = 0.0, s0_2: float = 25.0
) -> np.ndarray:
    """Sample mean parameters from normal distributions.

    For each component, samples from the posterior normal distribution
    given the assigned data points and prior parameters.

    Args:
        y: Observed data points.
        z: Current cluster assignments.
        K: Number of mixture components.
        m0: Prior mean for component means.
        s0_2: Prior variance for component means.

    Returns:
        New mean parameters for each component.
    """
    mu = np.empty(K)
    for k in range(K):
        idx = z == k
        n_k = idx.sum()
        if n_k == 0:
            mu[k] = np.random.normal(m0, np.sqrt(s0_2))
        else:
            ybar = y[idx].mean()
            post_var = 1.0 / (n_k + 1 / s0_2)
            post_mean = post_var * (n_k * ybar + m0 / s0_2)
            mu[k] = np.random.normal(post_mean, np.sqrt(post_var))
    return mu


def gibbs_step(y: np.ndarray, state: State, K: int) -> State:
    """Perform one step of the Gibbs sampler.

    Sequentially samples z, pi, and mu given the current state.

    Args:
        y: Observed data points.
        state: Current state of the sampler.
        K: Number of mixture components.

    Returns:
        New state after one Gibbs sampling step.
    """
    z = sample_z(y, state)
    pi = sample_pi(z, K)
    mu = sample_mu(y, z, K)
    return State(z, pi, mu)


def acf_1d(x: np.ndarray, max_lag: int = 40):
    """Compute autocorrelation function for a 1D time series.

    Args:
        x: Input time series.
        max_lag: Maximum lag to compute autocorrelation for.

    Returns:
        Autocorrelation values from lag 0 to max_lag.
    """
    n = len(x)
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[n - 1 : n + max_lag] / n
    return ac / ac[0]


def ess_1d(x: np.ndarray, max_lag: int = 40):
    """Compute effective sample size for a 1D time series.

    Uses the autocorrelation function to estimate the effective
    sample size accounting for serial correlation.

    Args:
        x: Input time series.
        max_lag: Maximum lag to use in autocorrelation computation.

    Returns:
        Effective sample size.
    """
    rho = acf_1d(x, max_lag)[1:]
    pos = rho[rho > 0]
    return len(x) / (1 + 2 * pos.sum())


def rhat_scalar(chains: List[np.ndarray]):
    """Compute R-hat convergence diagnostic for multiple chains.

    Computes the potential scale reduction factor (R-hat) to assess
    convergence of MCMC chains.

    Args:
        chains: List of chains for the same parameter.

    Returns:
        R-hat value (should be close to 1 for convergence).
    """
    n = min(len(c) for c in chains)
    chains = [c[:n] for c in chains]
    means = np.array([c.mean() for c in chains])
    W = np.array([c.var(ddof=1) for c in chains]).mean()
    B = n * means.var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W)


def credible_interval(samples: np.ndarray, alpha: float = 0.05):
    """Compute credible interval from posterior samples.

    Args:
        samples: Posterior samples.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (lower, upper) bounds of the credible interval.
    """
    lower = np.percentile(samples, 100 * alpha / 2)
    upper = np.percentile(samples, 100 * (1 - alpha / 2))
    return lower, upper


def run_chain(y: np.ndarray, K: int, n_iter: int, burn: int, seed: int):
    """Run a single Gibbs sampling chain.

    Args:
        y: Observed data points.
        K: Number of mixture components.
        n_iter: Total number of iterations.
        burn: Number of burn-in iterations to discard.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (samples, runtime) where samples contains the
        post-burn-in samples and runtime is the elapsed time.
    """
    rng = np.random.default_rng(seed)
    state = State(rng.integers(0, K, len(y)), np.ones(K) / K, rng.normal(0, 1, K))
    kept = []
    t0 = time.perf_counter()
    for it in range(n_iter):
        state = gibbs_step(y, state, K)
        if it >= burn:
            kept.append(relabel(state).mu.copy())
    return np.vstack(kept), time.perf_counter() - t0


def setup_plot_colors_and_positions(K: int, n_chains: int):
    """Set up colors and subplot positions for diagnostic plots.

    Args:
        K: Number of mixture components.
        n_chains: Number of chains.

    Returns:
        Tuple of (param_colors, positions) where param_colors maps
        component indices to colors and positions contains subplot
        coordinates for each chain.
    """
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    param_colors = {k: colors[k % len(colors)] for k in range(K)}

    cols = min(2, n_chains)
    positions = []
    for i in range(n_chains):
        row = (i // cols) + 1
        col = (i % cols) + 1
        positions.append((row, col))

    return param_colors, positions


def create_trace_plots(chains: List[np.ndarray], K: int, n_chains: int):
    """Create trace plots for all chains and parameters.

    Args:
        chains: List of sample arrays, one per chain.
        K: Number of mixture components.
        n_chains: Number of chains.
    """
    param_colors, positions = setup_plot_colors_and_positions(K, n_chains)
    n_rows = max(positions, key=lambda x: x[0])[0]
    n_cols = max(positions, key=lambda x: x[1])[1]

    fig_trace_all = sp.make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=[f"Chain {c}" for c in range(n_chains)]
    )

    for c, (row, col) in enumerate(positions):
        for k in range(K):
            fig_trace_all.add_trace(
                go.Scatter(
                    y=chains[c][:, k],
                    mode="lines",
                    line={"width": 0.8, "color": param_colors[k]},
                    name=f"$\\mu_{{{k}}}$",
                    showlegend=(c == 0),
                ),
                row=row,
                col=col,
            )

    fig_trace_all.update_layout(
        height=600,
        width=1000,
        title_text="Trace Plots by Chain - All Means",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_trace_all.update_xaxes(gridcolor="lightgray")
    fig_trace_all.update_yaxes(gridcolor="lightgray")
    fig_trace_all.write_image("figures/gibbs_trace.png")


def create_acf_plots(chains: List[np.ndarray], K: int, n_chains: int):
    """Create autocorrelation function plots for all chains and parameters.

    Args:
        chains: List of sample arrays, one per chain.
        K: Number of mixture components.
        n_chains: Number of chains.
    """
    param_colors, positions = setup_plot_colors_and_positions(K, n_chains)
    n_rows = max(positions, key=lambda x: x[0])[0]
    n_cols = max(positions, key=lambda x: x[1])[1]

    fig_acf_all = sp.make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=[f"Chain {c}" for c in range(n_chains)]
    )

    for c, (row, col) in enumerate(positions):
        for k in range(K):
            acf_vals = acf_1d(chains[c][:, k])
            fig_acf_all.add_trace(
                go.Scatter(
                    x=list(range(len(acf_vals))),
                    y=acf_vals,
                    mode="markers+lines",
                    line={"color": param_colors[k]},
                    marker={"color": param_colors[k]},
                    name=f"$\\mu_{{{k}}}$",
                    showlegend=(c == 0),
                ),
                row=row,
                col=col,
            )

    fig_acf_all.update_layout(
        height=600,
        width=1000,
        title_text="ACF Plots by Chain - All Means",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_acf_all.update_xaxes(gridcolor="lightgray")
    fig_acf_all.update_yaxes(gridcolor="lightgray")
    fig_acf_all.write_image("figures/gibbs_acf.png")


def create_histogram_plots(chains: List[np.ndarray], K: int, n_chains: int):
    """Create histogram plots for all chains and parameters.

    Args:
        chains: List of sample arrays, one per chain.
        K: Number of mixture components.
        n_chains: Number of chains.
    """
    param_colors, positions = setup_plot_colors_and_positions(K, n_chains)
    n_rows = max(positions, key=lambda x: x[0])[0]
    n_cols = max(positions, key=lambda x: x[1])[1]

    fig_hist_all = sp.make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=[f"Chain {c}" for c in range(n_chains)]
    )

    for c, (row, col) in enumerate(positions):
        for k in range(K):
            fig_hist_all.add_trace(
                go.Histogram(
                    x=chains[c][:, k],
                    nbinsx=30,
                    opacity=0.6,
                    marker={"color": param_colors[k]},
                    name=f"$\\mu_{{{k}}}$",
                    showlegend=(c == 0),
                ),
                row=row,
                col=col,
            )

    fig_hist_all.update_layout(
        height=600,
        width=1000,
        title_text="Histograms by Chain - All Means",
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_hist_all.update_xaxes(gridcolor="lightgray")
    fig_hist_all.update_yaxes(gridcolor="lightgray")
    fig_hist_all.write_image("figures/gibbs_hist.png")


def create_diagnostic_plots(chains: List[np.ndarray], K: int, n_chains: int):
    """Create all diagnostic plots (trace, ACF, and histograms).

    Args:
        chains: List of sample arrays, one per chain.
        K: Number of mixture components.
        n_chains: Number of chains.
    """
    create_trace_plots(chains, K, n_chains)
    create_acf_plots(chains, K, n_chains)
    create_histogram_plots(chains, K, n_chains)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gibbs Sampler for Gaussian mixture")
    ap.add_argument(
        "--data", type=str, default="data.npy", help="Path to the data file"
    )
    ap.add_argument("--K", type=int, required=True, help="Number of mixture components")
    ap.add_argument("--n_iter", type=int, default=10000, help="Number of iterations")
    ap.add_argument("--burn", type=int, default=2000, help="Burn-in period")
    ap.add_argument("--chains", type=int, default=4, help="Number of chains")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    rng_master = np.random.default_rng(args.seed)
    y = np.load(args.data)

    chains, times = [], []
    print(f"Running {args.chains} Gibbs chains…")
    for c in range(args.chains):
        mu, rt = run_chain(
            y, args.K, args.n_iter, args.burn, int(rng_master.integers(2**32))
        )
        chains.append(mu)
        times.append(rt)

    create_diagnostic_plots(chains, args.K, args.chains)

    pooled = np.vstack(chains)
    mu_mean = pooled.mean(axis=0)
    rhat = [rhat_scalar([c[:, k] for c in chains]) for k in range(args.K)]
    ess = [ess_1d(pooled[:, k]) for k in range(args.K)]

    ci_lower = np.array([credible_interval(pooled[:, k])[0] for k in range(args.K)])
    ci_upper = np.array([credible_interval(pooled[:, k])[1] for k in range(args.K)])

    print("\n=== Gibbs SUMMARY ===")
    print("Posterior mean μ  :", np.round(mu_mean, 4))
    print("95% CI lower      :", np.round(ci_lower, 4))
    print("95% CI upper      :", np.round(ci_upper, 4))
    print("R‑hat (μ)         :", np.round(rhat, 3))
    print("ESS  (μ)          :", np.round(ess, 1))
    print(f"Mean runtime / chain: {np.mean(times):.2f}s")

    print("\nPNG diagnostics saved in", os.getcwd())
