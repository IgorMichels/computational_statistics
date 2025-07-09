# pylint: disable=too-many-locals

import argparse
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from metrics import acf_1d, credible_interval, ess_1d
from samplers import run_chain


def run_comparison(args, show_summary=True):
    """
    Run comparison between Gibbs Sampler and Tempered Transitions.
    Save chain results and metrics to data directory.
    """
    # Load data
    y = np.load(f"../data/{args.data}/data.npy")

    print(f"Running comparison on dataset: {args.data}")

    # Run Gibbs Sampler
    print("Running Gibbs Sampler...")
    start_time = time.time()
    gibbs_samples, _, gibbs_acc_rate = run_chain(
        y,
        args.K,
        args.n_iter,
        args.burn,
        args.seed,
        n_temps=1,
        max_temp=1,
        n_gibbs_per_temp=1,
    )
    gibbs_time = time.time() - start_time

    # Run Tempered Transitions
    print("Running Tempered Transitions...")
    start_time = time.time()
    tempered_samples, _, tempered_acc_rate = run_chain(
        y,
        args.K,
        args.n_iter,
        args.burn,
        args.seed,
        n_temps=args.n_temps,
        max_temp=args.max_temp,
        n_gibbs_per_temp=args.n_gibbs_per_temp,
        keep_last=True,
    )
    tempered_time = time.time() - start_time

    # Calculate metrics
    print("Computing metrics...")
    gibbs_ci = np.array([credible_interval(gibbs_samples[:, k]) for k in range(args.K)])
    tempered_ci = np.array(
        [credible_interval(tempered_samples[:, k]) for k in range(args.K)]
    )

    gibbs_ess = np.array([ess_1d(gibbs_samples[:, k]) for k in range(args.K)])
    tempered_ess = np.array([ess_1d(tempered_samples[:, k]) for k in range(args.K)])

    # Create results directory
    results_dir = Path(f"../data/{args.data}/comparison_results")
    results_dir.mkdir(exist_ok=True, parents=True)

    # Save samples
    np.save(results_dir / "gibbs_samples.npy", gibbs_samples)
    np.save(results_dir / "tempered_samples.npy", tempered_samples)

    # Save structured results
    results = {
        "gibbs": {
            "samples": gibbs_samples,
            "time": gibbs_time,
            "acceptance_rate": gibbs_acc_rate,
            "means": gibbs_samples.mean(axis=0),
            "stds": gibbs_samples.std(axis=0),
            "credible_intervals": gibbs_ci,
            "ess": gibbs_ess,
        },
        "tempered": {
            "samples": tempered_samples,
            "time": tempered_time,
            "acceptance_rate": tempered_acc_rate,
            "means": tempered_samples.mean(axis=0),
            "stds": tempered_samples.std(axis=0),
            "credible_intervals": tempered_ci,
            "ess": tempered_ess,
        },
        "parameters": {
            "dataset": args.data,
            "K": args.K,
            "n_iter": args.n_iter,
            "burn": args.burn,
            "seed": args.seed,
            "gibbs_params": {"n_temps": 1, "max_temp": 1, "n_gibbs_per_temp": 1},
            "tempered_params": {
                "n_temps": args.n_temps,
                "max_temp": args.max_temp,
                "n_gibbs_per_temp": args.n_gibbs_per_temp,
                "keep_last": True,
            },
        },
    }

    np.savez_compressed(
        results_dir / "comparison_results.npz",
        **{key: value for key, value in results.items() if key != "parameters"},
        **results["parameters"],
    )

    # Print summary
    if show_summary:
        print("\n=== COMPARISON SUMMARY ===")
        print(
            f"Gibbs:    {gibbs_time:.2f}s,",
            f"acc={gibbs_acc_rate:.2%},",
            f"ESS={np.round(gibbs_ess, 1)}",
        )
        print(
            f"Tempered: {tempered_time:.2f}s,",
            f"acc={tempered_acc_rate:.2%},",
            f"ESS={np.round(tempered_ess, 1)}",
        )

    print(f"Results saved to: {results_dir}")

    return results


def load_comparison_results(dataset_name):
    """
    Load comparison results saved by run_comparison.
    """
    results_dir = Path(f"../data/{dataset_name}/comparison_results")

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Directory {results_dir} not found. "
            f"Run comparison first with: python comparison_tool.py compare --data {dataset_name}"
        )

    # Load main data
    data = np.load(results_dir / "comparison_results.npz", allow_pickle=True)

    # Reconstruct data structure
    results = {
        "gibbs": {
            "samples": data["gibbs"].item()["samples"],
            "time": data["gibbs"].item()["time"],
            "acceptance_rate": data["gibbs"].item().get("acceptance_rate", 0.0),
            "means": data["gibbs"].item()["means"],
            "stds": data["gibbs"].item()["stds"],
            "credible_intervals": data["gibbs"].item()["credible_intervals"],
            "ess": data["gibbs"].item()["ess"],
        },
        "tempered": {
            "samples": data["tempered"].item()["samples"],
            "time": data["tempered"].item()["time"],
            "acceptance_rate": data["tempered"].item().get("acceptance_rate", 0.0),
            "means": data["tempered"].item()["means"],
            "stds": data["tempered"].item()["stds"],
            "credible_intervals": data["tempered"].item()["credible_intervals"],
            "ess": data["tempered"].item()["ess"],
        },
        "parameters": {
            "dataset": str(data.get("dataset", dataset_name)),
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


def generate_plots(args):
    """
    Load comparison results and generate all plots.
    """
    print(f"Generating comparison plots for dataset: {args.data}")

    # Load data
    results = load_comparison_results(args.data)

    # Extract necessary data
    gibbs_samples = results["gibbs"]["samples"]
    tempered_samples = results["tempered"]["samples"]
    gibbs_ci = results["gibbs"]["credible_intervals"]
    tempered_ci = results["tempered"]["credible_intervals"]
    K = results["parameters"]["K"]

    # Create figures directory
    figures_dir = Path(f"../figures/{args.data}")
    figures_dir.mkdir(exist_ok=True, parents=True)

    # Generate plots
    print("Generating trace plots...")
    fig_trace = create_trace_plots(gibbs_samples, tempered_samples, K)
    fig_trace.write_image(
        figures_dir / "comparison_traces.png", width=1400, height=400, scale=2
    )

    print("Generating autocorrelation plots...")
    fig_acf = create_acf_plots(gibbs_samples, tempered_samples, K)
    fig_acf.write_image(
        figures_dir / "comparison_acf.png", width=1400, height=400, scale=2
    )

    print("Generating histograms...")
    fig_hist = create_histogram_plots(
        gibbs_samples, tempered_samples, gibbs_ci, tempered_ci, K
    )
    fig_hist.write_image(
        figures_dir / "comparison_histograms.png", width=1400, height=400, scale=2
    )

    print("Generating complete figure...")
    fig_complete = create_complete_plot(
        gibbs_samples, tempered_samples, gibbs_ci, tempered_ci, K
    )
    fig_complete.write_image(
        figures_dir / "comparison_complete.png", width=1400, height=900, scale=2
    )

    print(f"Plots saved to: {figures_dir}")

    # Print summary
    print("\n=== RESULTS SUMMARY ===")
    print(
        f"Gibbs:    {results['gibbs']['time']:.2f}s,",
        f"acc={results['gibbs']['acceptance_rate']:.2%},",
        f"ESS={np.round(results['gibbs']['ess'], 1)}",
    )
    print(
        f"Tempered: {results['tempered']['time']:.2f}s,",
        f"acc={results['tempered']['acceptance_rate']:.2%},",
        f"ESS={np.round(results['tempered']['ess'], 1)}",
    )


def run_all(args):
    """
    Run complete comparison and generate plots.
    """
    # Check if dataset exists
    data_path = Path(f"../data/{args.data}/data.npy")
    if not data_path.exists():
        print(f"Error: Dataset {data_path} not found!")
        return

    print(f"Running complete comparison on dataset: {args.data}")

    # Step 1: Run comparison
    print("Step 1: Running comparison...")
    run_comparison(args, show_summary=False)

    # Step 2: Generate plots
    print("\nStep 2: Generating plots...")
    generate_plots(args)

    # Final report
    print("\nComparison completed successfully!")
    print(f"Results: ../data/{args.data}/comparison_results/")
    print(f"Plots: ../figures/{args.data}/")


def main():
    """
    Main entry point with subcommands.
    """
    parser = argparse.ArgumentParser(
        description="Unified comparison tool: Gibbs Sampler vs Tempered Transitions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments for all subcommands
    def add_common_args(subparser):
        subparser.add_argument(
            "--data", type=str, default="example_1", help="Dataset name"
        )

    def add_comparison_args(subparser):
        add_common_args(subparser)
        subparser.add_argument(
            "--K", type=int, default=4, help="Number of mixture components"
        )
        subparser.add_argument(
            "--n_iter", type=int, default=5000, help="Number of iterations"
        )
        subparser.add_argument("--burn", type=int, default=1000, help="Burn-in period")
        subparser.add_argument("--seed", type=int, default=0, help="Random seed")
        subparser.add_argument(
            "--n_temps", type=int, default=10, help="Number of temperatures"
        )
        subparser.add_argument(
            "--max_temp", type=float, default=10.0, help="Maximum temperature"
        )
        subparser.add_argument(
            "--n_gibbs_per_temp",
            type=int,
            default=1,
            help="Gibbs steps per temperature",
        )

    # Subcommand: all (run comparison + plots)
    parser_all = subparsers.add_parser(
        "all", help="Run complete comparison and generate plots"
    )
    add_comparison_args(parser_all)

    # Subcommand: compare (run comparison only)
    parser_compare = subparsers.add_parser("compare", help="Run comparison only")
    add_comparison_args(parser_compare)

    # Subcommand: plot (generate plots only)
    parser_plot = subparsers.add_parser("plot", help="Generate plots only")
    add_common_args(parser_plot)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "compare":
        run_comparison(args, show_summary=True)
        return

    if args.command == "plot":
        generate_plots(args)
        return

    run_all(args)


if __name__ == "__main__":
    main()
