# pylint: disable=too-many-locals

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from metrics import acf_1d


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


def get_param_label(param_name: str, k: Optional[int] = None) -> str:
    """Get parameter label for plots."""
    if param_name == "mu":
        return f"\\mu_{{{k + 1}}}" if k is not None else "\\mu"
    if param_name == "sigma2":
        return f"\\sigma^2_{{{k + 1}}}" if k is not None else "\\sigma^2"

    return f"{param_name}_{{{k + 1}}}" if k is not None else param_name


def create_trace_plots(
    sampler_name: str,
    chains: List[np.ndarray],
    n_chains: int,
    data_name: str,
    param_name: str = "mu",
):
    """Create trace plots for all chains and parameters.

    Generates trace plots showing the evolution of parameter values across
    iterations for each MCMC chain. Each parameter is displayed with a
    different color, and each chain gets its own subplot.

    Args:
        sampler_name: Name of the sampling algorithm used.
        chains: List of sample arrays, one per chain.
        n_chains: Number of chains.
        data_name: Name of the dataset for file naming.
        param_name: Name of the parameter type ("mu" or "sigma2").
    """
    K = chains[0].shape[1]
    param_colors, positions = setup_plot_colors_and_positions(K, n_chains)
    n_rows = max(positions, key=lambda x: x[0])[0]
    n_cols = max(positions, key=lambda x: x[1])[1]

    fig_trace_all = sp.make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Chain {c + 1}" for c in range(n_chains)],
    )

    for c, (row, col) in enumerate(positions):
        for k in range(K):
            fig_trace_all.add_trace(
                go.Scatter(
                    y=chains[c][:, k],
                    mode="lines",
                    line={"width": 0.8, "color": param_colors[k]},
                    name=f"${get_param_label(param_name, k)}$",
                    showlegend=(c == 0),
                ),
                row=row,
                col=col,
            )

    fig_trace_all.update_layout(
        height=600,
        width=1000,
        title_text="$\\text{Trace Plots by Chain - Parameter }"
        f"{get_param_label(param_name)}$",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_trace_all.update_xaxes(gridcolor="lightgray")
    fig_trace_all.update_yaxes(gridcolor="lightgray")
    fig_trace_all.write_image(
        f"../figures/{data_name}/{sampler_name}_trace_{param_name}.png"
    )


def create_acf_plots(
    sampler_name: str,
    chains: List[np.ndarray],
    n_chains: int,
    data_name: str,
    param_name: str = "mu",
):
    """Create autocorrelation function plots for all chains and parameters.

    Generates autocorrelation function (ACF) plots to assess the serial
    correlation in MCMC samples. Lower autocorrelation indicates better
    mixing and more efficient sampling.

    Args:
        sampler_name: Name of the sampling algorithm used.
        chains: List of sample arrays, one per chain.
        n_chains: Number of chains.
        data_name: Name of the dataset for file naming.
        param_name: Name of the parameter type ("mu" or "sigma2").
    """
    K = chains[0].shape[1]
    param_colors, positions = setup_plot_colors_and_positions(K, n_chains)
    n_rows = max(positions, key=lambda x: x[0])[0]
    n_cols = max(positions, key=lambda x: x[1])[1]

    fig_acf_all = sp.make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Chain {c + 1}" for c in range(n_chains)],
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
                    name=f"${get_param_label(param_name, k)}$",
                    showlegend=(c == 0),
                ),
                row=row,
                col=col,
            )

    fig_acf_all.update_layout(
        height=600,
        width=1000,
        title_text="$\\text{ACF Plots by Chain - Parameter }"
        f"{get_param_label(param_name)}$",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_acf_all.update_xaxes(gridcolor="lightgray")
    fig_acf_all.update_yaxes(gridcolor="lightgray")
    fig_acf_all.write_image(
        f"../figures/{data_name}/{sampler_name}_acf_{param_name}.png"
    )


def create_histogram_plots(
    sampler_name: str,
    chains: List[np.ndarray],
    n_chains: int,
    data_name: str,
    param_name: str = "mu",
):
    """Create histogram plots for all chains and parameters.

    Generates histogram plots showing the marginal posterior distributions
    for each parameter from each MCMC chain. Overlapping histograms help
    visualize the posterior density and assess convergence across chains.

    Args:
        sampler_name: Name of the sampling algorithm used.
        chains: List of sample arrays, one per chain.
        n_chains: Number of chains.
        data_name: Name of the dataset for file naming.
        param_name: Name of the parameter type ("mu" or "sigma2").
    """
    K = chains[0].shape[1]
    param_colors, positions = setup_plot_colors_and_positions(K, n_chains)
    n_rows = max(positions, key=lambda x: x[0])[0]
    n_cols = max(positions, key=lambda x: x[1])[1]

    fig_hist_all = sp.make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Chain {c + 1}" for c in range(n_chains)],
    )

    for c, (row, col) in enumerate(positions):
        for k in range(K):
            fig_hist_all.add_trace(
                go.Histogram(
                    x=chains[c][:, k],
                    nbinsx=30,
                    opacity=0.6,
                    marker={"color": param_colors[k]},
                    name=f"${get_param_label(param_name, k)}$",
                    showlegend=(c == 0),
                ),
                row=row,
                col=col,
            )

    fig_hist_all.update_layout(
        height=600,
        width=1000,
        title_text="$\\text{Histograms by Chain - Parameter }"
        f"{get_param_label(param_name)}$",
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_hist_all.update_xaxes(gridcolor="lightgray")
    fig_hist_all.update_yaxes(gridcolor="lightgray")
    fig_hist_all.write_image(
        f"../figures/{data_name}/{sampler_name}_hist_{param_name}.png"
    )


def create_diagnostic_plots(
    sampler_name: str,
    chains: List[np.ndarray],
    n_chains: int,
    data_name: str,
    param_name: str = "mu",
):
    """Create all diagnostic plots (trace, ACF, and histograms).

    Generates a comprehensive set of diagnostic plots for MCMC analysis,
    including trace plots for convergence assessment, autocorrelation
    function plots for mixing evaluation, and histograms for posterior
    visualization.

    Args:
        sampler_name: Name of the sampling algorithm used.
        chains: List of sample arrays, one per chain.
        n_chains: Number of chains.
        data_name: Name of the dataset for file naming.
        param_name: Name of the parameter type ("mu" or "sigma2").
    """
    create_trace_plots(sampler_name, chains, n_chains, data_name, param_name)
    create_acf_plots(sampler_name, chains, n_chains, data_name, param_name)
    create_histogram_plots(sampler_name, chains, n_chains, data_name, param_name)
