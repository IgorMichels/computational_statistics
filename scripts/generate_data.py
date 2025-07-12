# pylint: disable=too-many-locals

from pathlib import Path
from typing import List

import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm


def generate_data(
    n: int, means: List[float], weights: List[float], sigmas: List[float], name: str
) -> None:
    """
    Generate synthetic data from a Gaussian mixture model and create visualizations.

    This function generates n samples from a mixture of Gaussian distributions with
    specified means, weights, and standard deviations. It saves the generated data
    and creates two plots: one showing the theoretical density components and mixture,
    and another comparing the simulated data histogram with the theoretical density.

    Args:
        n: Number of data points to generate.
        means: List of mean values for each Gaussian component.
        weights: List of mixture weights for each component (must sum to 1).
        sigmas: List of standard deviations for each Gaussian component.
        name: Name identifier for the dataset (used for directory and file naming).

    Returns:
        None. The function saves data to '../data/{name}/data.npy' and plots to
        '../figures/{name}/' directory.

    Raises:
        ValueError: If the lengths of means, weights, and sigmas don't match.
    """
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)
    data_dir = Path(f"../data/{name}")
    data_dir.mkdir(exist_ok=True)

    figure_dir = Path("../figures/")
    figure_dir.mkdir(exist_ok=True)
    figure_dir = Path(f"../figures/{name}")
    figure_dir.mkdir(exist_ok=True)

    K = len(means)
    classes = np.random.choice(range(K), size=n, p=weights)
    sample_means = [means[c] for c in classes]
    sample_sigmas = [sigmas[c] for c in classes]
    data = np.random.normal(loc=sample_means, scale=sample_sigmas)
    np.save(data_dir / "data.npy", data)

    # save true parameters for reference
    true_params = {
        "means": means,
        "weights": weights,
        "sigmas": sigmas,
        "variances": [s**2 for s in sigmas],
    }
    np.save(data_dir / "true_params.npy", true_params)

    # plot generator density
    lim_inf = min(means) - 4 * max(sigmas)
    lim_sup = max(means) + 4 * max(sigmas)
    amplitude = int(lim_sup - lim_inf)
    x = np.linspace(lim_inf, lim_sup, amplitude * 100)
    mixture_density = sum(
        weights[i] * norm.pdf(x, means[i], sigmas[i]) for i in range(K)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=mixture_density, mode="lines", showlegend=False, line={"width": 3}
        )
    )

    for i in range(K):
        individual_density = weights[i] * norm.pdf(x, means[i], sigmas[i])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=individual_density,
                mode="lines",
                name=f"N({means[i]}, {sigmas[i]}) (w = {weights[i]})",
                line={"dash": "dash", "width": 2},
            )
        )

    fig.update_layout(
        title=f"Mixture of {K} Gaussians",
        xaxis_title="x",
        yaxis_title="f(x)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=600,
        width=1000,
    )

    fig.write_image(figure_dir / "generator_density.png")

    # plot histogram of simulated data
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=data,
            nbinsx=50,
            histnorm="probability density",
            name="Simulated Data",
            opacity=0.7,
        )
    )

    fig_hist.add_trace(
        go.Scatter(
            x=x,
            y=mixture_density,
            mode="lines",
            name="Theoretical Density",
            line={"color": "red", "width": 3},
        )
    )

    fig_hist.update_layout(
        title="Simulated Data vs Theoretical Density",
        xaxis_title="x",
        yaxis_title="Density",
        plot_bgcolor="rgba(0,0,0,0)",
        height=600,
        width=1000,
    )

    fig_hist.write_image(figure_dir / "data_histogram.png")


if __name__ == "__main__":
    np.random.seed(0)

    n = 600
    sigmas = [0.5, 1.0, 1.5, 2.0]
    means = [-2.0, 0.0, 3.0, 5.0]
    weights = [0.2, 0.3, 0.1, 0.4]
    generate_data(n, means, weights, sigmas, "example_1")

    sigmas = [0.8, 1.2, 0.6, 1.8, 0.4]
    means = [-2.0, 0.0, 3.0, 5.0, 15.0]
    weights = [0.2, 0.3, 0.1, 0.35, 0.05]
    generate_data(n, means, weights, sigmas, "example_2")

    sigmas = [1.0, 1.0, 1.0, 1.0]
    means = [-2.0, 0.0, 3.0, 5.0]
    weights = [0.2, 0.3, 0.1, 0.4]
    generate_data(n, means, weights, sigmas, "example_3")
