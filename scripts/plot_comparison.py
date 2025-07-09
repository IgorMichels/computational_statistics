# pylint: disable=too-many-locals

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from metrics import acf_1d


def load_comparison_results():
    """
    Load comparison results saved by run_comparison.py
    """
    results_dir = Path("../data/comparison_results")

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Directory {results_dir} not found. " "Run run_comparison.py script first"
        )

    # Load main data
    data = np.load(results_dir / "comparison_results.npz", allow_pickle=True)

    # Reconstruct data structure
    results = {
        "gibbs": {
            "samples": data["gibbs"].item()["samples"],
            "time": data["gibbs"].item()["time"],
            "means": data["gibbs"].item()["means"],
            "stds": data["gibbs"].item()["stds"],
            "credible_intervals": data["gibbs"].item()["credible_intervals"],
            "ess": data["gibbs"].item()["ess"],
        },
        "tempered": {
            "samples": data["tempered"].item()["samples"],
            "time": data["tempered"].item()["time"],
            "means": data["tempered"].item()["means"],
            "stds": data["tempered"].item()["stds"],
            "credible_intervals": data["tempered"].item()["credible_intervals"],
            "ess": data["tempered"].item()["ess"],
        },
        "parameters": {
            "K": int(data["K"]),
        },
    }

    return results


def create_trace_plots(gibbs_samples, tempered_samples, K):
    """Create comparative trace plots"""
    fig = sp.make_subplots(
        rows=1,
        cols=K,
        subplot_titles=[f"$\\mu_{{{k+1}}}$" for k in range(K)],
        horizontal_spacing=0.05,
    )

    colors = ["blue", "red"]
    names = ["Gibbs Sampler", "Tempered Transitions"]
    samples = [gibbs_samples, tempered_samples]

    for k in range(K):
        col = k + 1
        for _, (sample, color, name) in enumerate(zip(samples, colors, names)):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(sample))),
                    y=sample[:, k],
                    mode="lines",
                    name=name,
                    line={"color": color, "width": 1},
                    opacity=0.7,
                    showlegend=(k == 0),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Iteration", row=1, col=col)
        fig.update_yaxes(title_text="Value", row=1, col=col)

    fig.update_layout(
        title="Trace Plots - Method Comparison",
        title_x=0.5,
        height=400,
        width=1400,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def create_acf_plots(gibbs_samples, tempered_samples, K, max_lag=40):
    """Create autocorrelation plots"""
    fig = sp.make_subplots(
        rows=1,
        cols=K,
        subplot_titles=[f"$\\mu_{{{k+1}}}$" for k in range(K)],
        horizontal_spacing=0.05,
    )

    colors = ["blue", "red"]
    names = ["Gibbs Sampler", "Tempered Transitions"]
    samples = [gibbs_samples, tempered_samples]

    for k in range(K):
        col = k + 1
        for _, (sample, color, name) in enumerate(zip(samples, colors, names)):
            acf = acf_1d(sample[:, k], max_lag)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(acf))),
                    y=acf,
                    mode="lines+markers",
                    name=name,
                    line={"color": color, "width": 2},
                    marker={"size": 3},
                    showlegend=(k == 0),
                ),
                row=1,
                col=col,
            )
        fig.add_hline(
            y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=col
        )
        fig.update_xaxes(title_text="Lag", row=1, col=col)
        fig.update_yaxes(title_text="ACF", row=1, col=col)

    fig.update_layout(
        title="Autocorrelation Functions - Method Comparison",
        title_x=0.5,
        height=400,
        width=1400,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def create_histogram_plots(gibbs_samples, tempered_samples, gibbs_ci, tempered_ci, K):
    """Create histograms with credible intervals"""
    fig = sp.make_subplots(
        rows=1,
        cols=K,
        subplot_titles=[f"$\\mu_{{{k+1}}}$" for k in range(K)],
        horizontal_spacing=0.05,
    )

    colors = ["blue", "red"]
    names = ["Gibbs Sampler", "Tempered Transitions"]
    samples = [gibbs_samples, tempered_samples]
    intervals = [gibbs_ci, tempered_ci]

    for k in range(K):
        col = k + 1
        for _, (sample, color, name, ci) in enumerate(
            zip(samples, colors, names, intervals)
        ):
            fig.add_trace(
                go.Histogram(
                    x=sample[:, k],
                    name=name,
                    opacity=0.6,
                    marker_color=color,
                    nbinsx=30,
                    showlegend=(k == 0),
                ),
                row=1,
                col=col,
            )

            # Add vertical lines for credible intervals
            fig.add_vline(
                x=ci[k, 0],
                line_dash="dash",
                line_color=color,
                line_width=2,
                opacity=0.8,
                row=1,
                col=col,
            )
            fig.add_vline(
                x=ci[k, 1],
                line_dash="dash",
                line_color=color,
                line_width=2,
                opacity=0.8,
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Value", row=1, col=col)
        fig.update_yaxes(title_text="Frequency", row=1, col=col)

    fig.update_layout(
        title="Histograms with Credible Intervals - Method Comparison",
        title_x=0.5,
        height=400,
        width=1400,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def create_complete_plot(gibbs_samples, tempered_samples, gibbs_ci, tempered_ci, K):
    """Create complete figure with all plots"""
    fig = sp.make_subplots(
        rows=3,
        cols=K,
        subplot_titles=[f"$\\mu_{{{k+1}}}$" for k in range(K)] + [""] * (2 * K),
        specs=[
            [{"secondary_y": False} for _ in range(K)],
            [{"secondary_y": False} for _ in range(K)],
            [{"secondary_y": False} for _ in range(K)],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    colors = ["blue", "red"]
    names = ["Gibbs Sampler", "Tempered Transitions"]
    samples = [gibbs_samples, tempered_samples]
    intervals = [gibbs_ci, tempered_ci]

    # 1. Trace plots
    for k in range(K):
        col = k + 1
        for _, (sample, color, name) in enumerate(zip(samples, colors, names)):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(sample))),
                    y=sample[:, k],
                    mode="lines",
                    name=name,
                    line={"color": color, "width": 1},
                    opacity=0.7,
                    showlegend=(k == 0),
                ),
                row=1,
                col=col,
            )

    # 2. Autocorrelation plots
    max_lag = 40
    for k in range(K):
        col = k + 1
        for _, (sample, color, name) in enumerate(zip(samples, colors, names)):
            acf = acf_1d(sample[:, k], max_lag)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(acf))),
                    y=acf,
                    mode="lines+markers",
                    name=name,
                    line={"color": color, "width": 2},
                    marker={"size": 3},
                    showlegend=False,
                ),
                row=2,
                col=col,
            )
        fig.add_hline(
            y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=col
        )

    # 3. Histograms
    for k in range(K):
        col = k + 1
        for _, (sample, color, name, ci) in enumerate(
            zip(samples, colors, names, intervals)
        ):
            fig.add_trace(
                go.Histogram(
                    x=sample[:, k],
                    name=name,
                    opacity=0.6,
                    marker_color=color,
                    nbinsx=30,
                    showlegend=False,
                ),
                row=3,
                col=col,
            )

            # Credible intervals
            fig.add_vline(
                x=ci[k, 0],
                line_dash="dash",
                line_color=color,
                line_width=2,
                opacity=0.8,
                row=3,
                col=col,
            )
            fig.add_vline(
                x=ci[k, 1],
                line_dash="dash",
                line_color=color,
                line_width=2,
                opacity=0.8,
                row=3,
                col=col,
            )

    # Configure axes
    for k in range(K):
        fig.update_xaxes(title_text="Iteration", row=1, col=k + 1)
        fig.update_yaxes(title_text="Value", row=1, col=k + 1)

        fig.update_xaxes(title_text="Lag", row=2, col=k + 1)
        fig.update_yaxes(title_text="ACF", row=2, col=k + 1)

        fig.update_xaxes(title_text="Value", row=3, col=k + 1)
        fig.update_yaxes(title_text="Frequency", row=3, col=k + 1)

    fig.update_layout(
        title="Detailed Comparison: Gibbs Sampler vs Tempered Transitions",
        title_x=0.5,
        height=900,
        width=1400,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def main():
    """
    Load comparison results and generate all plots.
    """
    print("=== COMPARATIVE PLOTS GENERATION ===\n")

    # Load data
    print("1. Loading comparison results...")
    try:
        results = load_comparison_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Extract necessary data
    gibbs_samples = results["gibbs"]["samples"]
    tempered_samples = results["tempered"]["samples"]
    gibbs_ci = results["gibbs"]["credible_intervals"]
    tempered_ci = results["tempered"]["credible_intervals"]
    K = results["parameters"]["K"]

    print(f"   Data loaded: {gibbs_samples.shape[0]} samples, {K} parameters")

    # Create figures directory if it doesn't exist
    figures_dir = Path("../figures")
    figures_dir.mkdir(exist_ok=True)

    # Generate individual plots
    print("\n2. Generating trace plots...")
    fig_trace = create_trace_plots(gibbs_samples, tempered_samples, K)
    fig_trace.write_image(
        figures_dir / "comparison_traces.png", width=1400, height=400, scale=2
    )

    print("3. Generating autocorrelation plots...")
    fig_acf = create_acf_plots(gibbs_samples, tempered_samples, K)
    fig_acf.write_image(
        figures_dir / "comparison_acf.png", width=1400, height=400, scale=2
    )

    print("4. Generating histograms...")
    fig_hist = create_histogram_plots(
        gibbs_samples, tempered_samples, gibbs_ci, tempered_ci, K
    )
    fig_hist.write_image(
        figures_dir / "comparison_histograms.png", width=1400, height=400, scale=2
    )

    print("5. Generating complete figure...")
    fig_complete = create_complete_plot(
        gibbs_samples, tempered_samples, gibbs_ci, tempered_ci, K
    )
    fig_complete.write_image(
        figures_dir / "comparison_complete.png", width=1400, height=900, scale=2
    )

    print(f"\n✅ Plots saved in: {figures_dir}")
    print("  - comparison_traces.png")
    print("  - comparison_acf.png")
    print("  - comparison_histograms.png")
    print("  - comparison_complete.png")

    # Print results summary
    print("\n=== RESULTS SUMMARY ===")
    print("Gibbs Sampler:")
    print(f"  - Time: {results['gibbs']['time']:.2f}s")
    print(f"  - ESS: {np.round(results['gibbs']['ess'], 1)}")

    print("\nTempered Transitions:")
    print(f"  - Time: {results['tempered']['time']:.2f}s")
    print(f"  - ESS: {np.round(results['tempered']['ess'], 1)}")


if __name__ == "__main__":
    main()
