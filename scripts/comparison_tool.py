# pylint: disable=duplicate-code
# pylint: disable=too-many-locals

import argparse
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from metrics import acf_1d, compute_credible_intervals, ess_1d
from samplers import run_chain
from utils import add_common_args, add_comparison_args, parse_all_prior_args


def run_comparison(args):
    """
    Run comparison between Gibbs Sampler and Tempered Transitions.
    Save chain results and metrics to data directory.
    """
    # Load data
    y = np.load(f"../data/{args.data}/data.npy")

    # Parse prior parameters
    m0, s0_2, alpha0, beta0 = parse_all_prior_args(args)

    if args.verbose:
        print(f"Running comparison on dataset: {args.data}")

    # Run Gibbs Sampler
    if args.verbose:
        print("Running Gibbs Sampler...")
    start_time = time.time()
    gibbs_samples_mu, gibbs_samples_sigma2, _, gibbs_acc_rate = run_chain(
        y,
        args.K,
        args.n_iter,
        args.burn,
        args.seed,
        m0=m0,
        s0_2=s0_2,
        alpha0=alpha0,
        beta0=beta0,
        n_temps=1,
        max_temp=1,
        n_gibbs_per_temp=1,
    )
    gibbs_time = time.time() - start_time

    # Run Tempered Transitions
    if args.verbose:
        print("Running Tempered Transitions...")
    start_time = time.time()
    tempered_samples_mu, tempered_samples_sigma2, _, tempered_acc_rate = run_chain(
        y,
        args.K,
        args.n_iter,
        args.burn,
        args.seed,
        m0=m0,
        s0_2=s0_2,
        alpha0=alpha0,
        beta0=beta0,
        n_temps=getattr(args, "n_temps", 10),
        max_temp=getattr(args, "max_temp", 5.0),
        n_gibbs_per_temp=getattr(args, "n_gibbs_per_temp", 5),
    )
    tempered_time = time.time() - start_time

    # Calculate metrics for means
    if args.verbose:
        print("Computing metrics...")
    gibbs_ci_mu = np.array(
        [compute_credible_intervals(gibbs_samples_mu[:, k]) for k in range(args.K)]
    )
    tempered_ci_mu = np.array(
        [compute_credible_intervals(tempered_samples_mu[:, k]) for k in range(args.K)]
    )

    gibbs_ess_mu = np.array([ess_1d(gibbs_samples_mu[:, k]) for k in range(args.K)])
    tempered_ess_mu = np.array(
        [ess_1d(tempered_samples_mu[:, k]) for k in range(args.K)]
    )

    # Calculate metrics for variances
    gibbs_ci_sigma2 = np.array(
        [compute_credible_intervals(gibbs_samples_sigma2[:, k]) for k in range(args.K)]
    )
    tempered_ci_sigma2 = np.array(
        [
            compute_credible_intervals(tempered_samples_sigma2[:, k])
            for k in range(args.K)
        ]
    )

    gibbs_ess_sigma2 = np.array(
        [ess_1d(gibbs_samples_sigma2[:, k]) for k in range(args.K)]
    )
    tempered_ess_sigma2 = np.array(
        [ess_1d(tempered_samples_sigma2[:, k]) for k in range(args.K)]
    )

    # Create results directory
    results_dir = Path(f"../data/{args.data}/comparison_results")
    results_dir.mkdir(exist_ok=True, parents=True)

    # Save samples
    np.save(results_dir / "gibbs_samples_mu.npy", gibbs_samples_mu)
    np.save(results_dir / "gibbs_samples_sigma2.npy", gibbs_samples_sigma2)
    np.save(results_dir / "tempered_samples_mu.npy", tempered_samples_mu)
    np.save(results_dir / "tempered_samples_sigma2.npy", tempered_samples_sigma2)

    # Save structured results
    results = {
        "gibbs": {
            "samples_mu": gibbs_samples_mu,
            "samples_sigma2": gibbs_samples_sigma2,
            "time": gibbs_time,
            "acceptance_rate": gibbs_acc_rate,
            "means_mu": gibbs_samples_mu.mean(axis=0),
            "stds_mu": gibbs_samples_mu.std(axis=0),
            "means_sigma2": gibbs_samples_sigma2.mean(axis=0),
            "stds_sigma2": gibbs_samples_sigma2.std(axis=0),
            "credible_intervals_mu": gibbs_ci_mu,
            "credible_intervals_sigma2": gibbs_ci_sigma2,
            "ess_mu": gibbs_ess_mu,
            "ess_sigma2": gibbs_ess_sigma2,
        },
        "tempered": {
            "samples_mu": tempered_samples_mu,
            "samples_sigma2": tempered_samples_sigma2,
            "time": tempered_time,
            "acceptance_rate": tempered_acc_rate,
            "means_mu": tempered_samples_mu.mean(axis=0),
            "stds_mu": tempered_samples_mu.std(axis=0),
            "means_sigma2": tempered_samples_sigma2.mean(axis=0),
            "stds_sigma2": tempered_samples_sigma2.std(axis=0),
            "credible_intervals_mu": tempered_ci_mu,
            "credible_intervals_sigma2": tempered_ci_sigma2,
            "ess_mu": tempered_ess_mu,
            "ess_sigma2": tempered_ess_sigma2,
        },
        "parameters": {
            "dataset": args.data,
            "K": args.K,
            "n_iter": args.n_iter,
            "burn": args.burn,
            "seed": args.seed,
            "m0": m0.tolist() if hasattr(m0, "tolist") else m0,
            "s0_2": s0_2.tolist() if hasattr(s0_2, "tolist") else s0_2,
            "alpha0": alpha0.tolist() if hasattr(alpha0, "tolist") else alpha0,
            "beta0": beta0.tolist() if hasattr(beta0, "tolist") else beta0,
            "gibbs_params": {"n_temps": 1, "max_temp": 1, "n_gibbs_per_temp": 1},
            "tempered_params": {
                "n_temps": getattr(args, "n_temps", 10),
                "max_temp": getattr(args, "max_temp", 5.0),
                "n_gibbs_per_temp": getattr(args, "n_gibbs_per_temp", 5),
            },
        },
    }

    np.savez_compressed(
        results_dir / "comparison_results.npz",
        **{key: value for key, value in results.items() if key != "parameters"},
        **results["parameters"],
    )

    # Print summary
    if args.verbose:
        print("\n=== COMPARISON SUMMARY ===")
        print("MEANS:")
        print(
            f"  Gibbs:    {gibbs_time:.2f}s,",
            f"acc={gibbs_acc_rate:.2%},",
            f"ESS={np.round(gibbs_ess_mu, 1)}",
        )
        print(
            f"  Tempered: {tempered_time:.2f}s,",
            f"acc={tempered_acc_rate:.2%},",
            f"ESS={np.round(tempered_ess_mu, 1)}",
        )
        print("VARIANCES:")
        print(f"  Gibbs:    ESS={np.round(gibbs_ess_sigma2, 1)}")
        print(f"  Tempered: ESS={np.round(tempered_ess_sigma2, 1)}")

    if args.verbose:
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
            "samples_mu": data["gibbs"].item()["samples_mu"],
            "samples_sigma2": data["gibbs"].item()["samples_sigma2"],
            "time": data["gibbs"].item()["time"],
            "acceptance_rate": data["gibbs"].item().get("acceptance_rate", 0.0),
            "means_mu": data["gibbs"].item()["means_mu"],
            "stds_mu": data["gibbs"].item()["stds_mu"],
            "means_sigma2": data["gibbs"].item()["means_sigma2"],
            "stds_sigma2": data["gibbs"].item()["stds_sigma2"],
            "credible_intervals_mu": data["gibbs"].item()["credible_intervals_mu"],
            "credible_intervals_sigma2": data["gibbs"].item()[
                "credible_intervals_sigma2"
            ],
            "ess_mu": data["gibbs"].item()["ess_mu"],
            "ess_sigma2": data["gibbs"].item()["ess_sigma2"],
        },
        "tempered": {
            "samples_mu": data["tempered"].item()["samples_mu"],
            "samples_sigma2": data["tempered"].item()["samples_sigma2"],
            "time": data["tempered"].item()["time"],
            "acceptance_rate": data["tempered"].item().get("acceptance_rate", 0.0),
            "means_mu": data["tempered"].item()["means_mu"],
            "stds_mu": data["tempered"].item()["stds_mu"],
            "means_sigma2": data["tempered"].item()["means_sigma2"],
            "stds_sigma2": data["tempered"].item()["stds_sigma2"],
            "credible_intervals_mu": data["tempered"].item()["credible_intervals_mu"],
            "credible_intervals_sigma2": data["tempered"].item()[
                "credible_intervals_sigma2"
            ],
            "ess_mu": data["tempered"].item()["ess_mu"],
            "ess_sigma2": data["tempered"].item()["ess_sigma2"],
        },
        "parameters": {
            "dataset": str(data.get("dataset", dataset_name)),
            "K": int(data["K"]),
        },
    }

    return results


def create_trace_plots(gibbs_samples, tempered_samples, K, param_name="mu"):
    """Create comparative trace plots"""
    param_label = "\\mu" if param_name == "mu" else "\\sigma^2"
    fig = sp.make_subplots(
        rows=1,
        cols=K,
        subplot_titles=[f"${param_label}_{{{k + 1}}}$" for k in range(K)],
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
                    showlegend=(k == 0),
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        title="$\\text{Trace Plots Comparison - Parameter }" f"{param_label}$",
        height=400,
        width=1200,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="lightgray")
    fig.update_yaxes(gridcolor="lightgray")

    return fig


def create_acf_plots(gibbs_samples, tempered_samples, K, param_name="mu", max_lag=40):
    """Create comparative ACF plots"""
    param_label = "\\mu" if param_name == "mu" else "\\sigma^2"
    fig = sp.make_subplots(
        rows=1,
        cols=K,
        subplot_titles=[f"${param_label}_{{{k + 1}}}$" for k in range(K)],
        horizontal_spacing=0.05,
    )

    colors = ["blue", "red"]
    names = ["Gibbs Sampler", "Tempered Transitions"]
    samples = [gibbs_samples, tempered_samples]

    for k in range(K):
        col = k + 1
        for _, (sample, color, name) in enumerate(zip(samples, colors, names)):
            acf_vals = acf_1d(sample[:, k], max_lag)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(acf_vals))),
                    y=acf_vals,
                    mode="lines+markers",
                    name=name,
                    line={"color": color},
                    marker={"color": color, "size": 3},
                    showlegend=(k == 0),
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        title="$\\text{ACF Plots Comparison - Parameter }" f"{param_label}$",
        height=400,
        width=1200,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="lightgray")
    fig.update_yaxes(gridcolor="lightgray")

    return fig


def create_histogram_plots(
    gibbs_samples, tempered_samples, gibbs_ci, tempered_ci, param_name="mu"
):
    """Create comparative histogram plots"""
    K = gibbs_samples.shape[1]
    param_label = "\\mu" if param_name == "mu" else "\\sigma^2"
    fig = sp.make_subplots(
        rows=1,
        cols=K,
        subplot_titles=[f"${param_label}_{{{k + 1}}}$" for k in range(K)],
        horizontal_spacing=0.05,
    )

    colors = ["blue", "red"]
    names = ["Gibbs Sampler", "Tempered Transitions"]
    samples = [gibbs_samples, tempered_samples]
    cis = [gibbs_ci, tempered_ci]

    for k in range(K):
        col = k + 1
        for _, (sample, color, name, ci) in enumerate(zip(samples, colors, names, cis)):
            fig.add_trace(
                go.Histogram(
                    x=sample[:, k],
                    nbinsx=30,
                    name=name,
                    opacity=0.6,
                    marker={"color": color},
                    showlegend=(k == 0),
                ),
                row=1,
                col=col,
            )

            # Add credible interval lines
            fig.add_vline(
                x=ci[k][0],
                line={"dash": "dash", "color": color, "width": 1},
                row=1,
                col=col,
            )
            fig.add_vline(
                x=ci[k][1],
                line={"dash": "dash", "color": color, "width": 1},
                row=1,
                col=col,
            )

    fig.update_layout(
        title="$\\text{Posterior Distributions Comparison - Parameter }"
        f"{param_label}$",
        height=400,
        width=1200,
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="lightgray")
    fig.update_yaxes(gridcolor="lightgray")

    return fig


def create_complete_plot(results, K, param_name):
    """Create a complete comparison plot with all diagnostics for a specific parameter"""
    param_label = "\\mu" if param_name == "mu" else "\\sigma^2"
    # Create subplots: 3 rows (trace, ACF, histogram), K columns
    fig = sp.make_subplots(
        rows=3,
        cols=K,
        subplot_titles=[f"${param_label}_{{{k + 1}}}$" for k in range(K)],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        specs=[[{"secondary_y": False} for _ in range(K)] for _ in range(3)],
    )

    colors = ["blue", "red"]
    names = ["Gibbs Sampler", "Tempered Transitions"]

    gibbs_samples = results["gibbs"][f"samples_{param_name}"]
    tempered_samples = results["tempered"][f"samples_{param_name}"]
    gibbs_ci = results["gibbs"][f"credible_intervals_{param_name}"]
    tempered_ci = results["tempered"][f"credible_intervals_{param_name}"]

    samples = [gibbs_samples, tempered_samples]
    cis = [gibbs_ci, tempered_ci]

    for k in range(K):
        col = k + 1

        # Trace plots
        for _, (sample, color, name) in enumerate(zip(samples, colors, names)):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(sample))),
                    y=sample[:, k],
                    mode="lines",
                    name=name,
                    line={"color": color, "width": 1},
                    showlegend=(k == 0),
                ),
                row=1,
                col=col,
            )

        # ACF plots
        for _, (sample, color, name) in enumerate(zip(samples, colors, names)):
            acf_vals = acf_1d(sample[:, k], 40)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(acf_vals))),
                    y=acf_vals,
                    mode="lines+markers",
                    name=name,
                    line={"color": color},
                    marker={"color": color, "size": 3},
                    showlegend=False,
                ),
                row=2,
                col=col,
            )

        # Histograms
        for _, (sample, color, name, ci) in enumerate(zip(samples, colors, names, cis)):
            fig.add_trace(
                go.Histogram(
                    x=sample[:, k],
                    nbinsx=30,
                    name=name,
                    opacity=0.6,
                    marker={"color": color},
                    showlegend=False,
                ),
                row=3,
                col=col,
            )

            # Add credible interval lines
            fig.add_vline(
                x=ci[k][0],
                line={"dash": "dash", "color": color, "width": 1},
                row=3,
                col=col,
            )
            fig.add_vline(
                x=ci[k][1],
                line={"dash": "dash", "color": color, "width": 1},
                row=3,
                col=col,
            )

    # Update layout
    param_display = "μ" if param_name == "mu" else "σ²"
    fig.update_layout(
        title="$\\text{Complete Comparison: Gibbs vs Tempered Transitions - Parameter }"
        f"{param_display}$",
        height=800,
        width=1400,
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Add row labels
    fig.add_annotation(
        x=-0.05,
        y=0.90,
        xref="paper",
        yref="paper",
        text="$\\text{Trace }" f"{param_display}$",
        textangle=90,
        showarrow=False,
        font={"size": 12, "color": "black"},
        xanchor="center",
    )
    fig.add_annotation(
        x=-0.05,
        y=0.50,
        xref="paper",
        yref="paper",
        text="$\\text{ACF }" f"{param_display}$",
        textangle=90,
        showarrow=False,
        font={"size": 12, "color": "black"},
        xanchor="center",
    )
    fig.add_annotation(
        x=-0.05,
        y=0.10,
        xref="paper",
        yref="paper",
        text="$\\text{Hist }" f"{param_display}$",
        textangle=90,
        showarrow=False,
        font={"size": 12, "color": "black"},
        xanchor="center",
    )

    fig.update_xaxes(gridcolor="lightgray")
    fig.update_yaxes(gridcolor="lightgray")

    return fig


def generate_plots(args):
    """
    Generate comparison plots from saved results.
    """
    results = load_comparison_results(args.data)
    K = results["parameters"]["K"]

    # Create output directory
    output_dir = Path(f"../figures/{args.data}/comparison")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate individual plots for means
    fig_trace_mu = create_trace_plots(
        results["gibbs"]["samples_mu"], results["tempered"]["samples_mu"], K, "mu"
    )
    fig_trace_mu.write_image(output_dir / "trace_comparison_mu.png")

    fig_acf_mu = create_acf_plots(
        results["gibbs"]["samples_mu"], results["tempered"]["samples_mu"], K, "mu"
    )
    fig_acf_mu.write_image(output_dir / "acf_comparison_mu.png")

    fig_hist_mu = create_histogram_plots(
        results["gibbs"]["samples_mu"],
        results["tempered"]["samples_mu"],
        results["gibbs"]["credible_intervals_mu"],
        results["tempered"]["credible_intervals_mu"],
        "mu",
    )
    fig_hist_mu.write_image(output_dir / "histogram_comparison_mu.png")

    # Generate individual plots for variances
    fig_trace_sigma2 = create_trace_plots(
        results["gibbs"]["samples_sigma2"],
        results["tempered"]["samples_sigma2"],
        K,
        "sigma2",
    )
    fig_trace_sigma2.write_image(output_dir / "trace_comparison_sigma2.png")

    fig_acf_sigma2 = create_acf_plots(
        results["gibbs"]["samples_sigma2"],
        results["tempered"]["samples_sigma2"],
        K,
        "sigma2",
    )
    fig_acf_sigma2.write_image(output_dir / "acf_comparison_sigma2.png")

    fig_hist_sigma2 = create_histogram_plots(
        results["gibbs"]["samples_sigma2"],
        results["tempered"]["samples_sigma2"],
        results["gibbs"]["credible_intervals_sigma2"],
        results["tempered"]["credible_intervals_sigma2"],
        "sigma2",
    )
    fig_hist_sigma2.write_image(output_dir / "histogram_comparison_sigma2.png")

    # Generate complete comparison plot
    fig_complete = create_complete_plot(results, K, "mu")
    fig_complete.write_image(output_dir / "complete_comparison_mu.png")

    fig_complete = create_complete_plot(results, K, "sigma2")
    fig_complete.write_image(output_dir / "complete_comparison_sigma2.png")

    if args.verbose:
        print(f"Plots saved to: {output_dir}")


def run_all(args):
    """
    Run complete comparison: compute results and generate plots.
    """
    if args.verbose:
        print("Step 1: Running comparison...")
    run_comparison(args)

    if args.verbose:
        print("\nStep 2: Generating plots...")
    generate_plots(args)

    if args.verbose:
        print("\nComparison complete!")


def main():
    """
    Main function to handle command line arguments and run appropriate functions.
    """
    parser = argparse.ArgumentParser(
        description="Comparison tool for Gibbs vs Tempered Transitions"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Run comparison between methods"
    )
    add_comparison_args(compare_parser)

    # Plot command
    plot_parser = subparsers.add_parser(
        "plot", help="Generate plots from saved results"
    )
    add_common_args(plot_parser)

    # All command
    all_parser = subparsers.add_parser("all", help="Run comparison and generate plots")
    add_comparison_args(all_parser)

    args = parser.parse_args()

    if args.command == "compare":
        run_comparison(args)
    elif args.command == "plot":
        generate_plots(args)
    elif args.command == "all":
        run_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
