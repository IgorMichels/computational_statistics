from typing import List

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


def create_trace_plots(
    sampler_name: str, chains: List[np.ndarray], K: int, n_chains: int
):
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
    fig_trace_all.write_image(f"../figures/{sampler_name}_trace.png")


def create_acf_plots(
    sampler_name: str, chains: List[np.ndarray], K: int, n_chains: int
):
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
    fig_acf_all.write_image(f"../figures/{sampler_name}_acf.png")


def create_histogram_plots(
    sampler_name: str, chains: List[np.ndarray], K: int, n_chains: int
):
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
    fig_hist_all.write_image(f"../figures/{sampler_name}_hist.png")


def create_diagnostic_plots(
    sampler_name: str, chains: List[np.ndarray], K: int, n_chains: int
):
    """Create all diagnostic plots (trace, ACF, and histograms).

    Args:
        chains: List of sample arrays, one per chain.
        K: Number of mixture components.
        n_chains: Number of chains.
    """
    create_trace_plots(sampler_name, chains, K, n_chains)
    create_acf_plots(sampler_name, chains, K, n_chains)
    create_histogram_plots(sampler_name, chains, K, n_chains)
