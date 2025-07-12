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
    if not len(means) == len(weights) == len(sigmas):
        raise ValueError("The lengths of means, weights and sigmas must be equal")

    if not np.isclose(sum(weights), 1.0):
        raise ValueError("The weights must sum to 1")

    data_dir = Path(f"../data/{name}")
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(f"../figures/{name}")
    figure_dir.mkdir(parents=True, exist_ok=True)

    K = len(means)

    classes = np.random.choice(K, size=n, p=weights)
    means_array = np.array(means)
    sigmas_array = np.array(sigmas)
    data = np.random.normal(means_array[classes], sigmas_array[classes])
    np.save(data_dir / "data.npy", data)

    # save true parameters for reference
    true_params = {
        "means": means,
        "weights": weights,
        "sigmas": sigmas,
        "variances": (np.array(sigmas) ** 2).tolist(),
    }
    np.save(data_dir / "true_params.npy", true_params)

    means_array = np.array(means)
    sigmas_array = np.array(sigmas)

    lim_inf = np.min(means_array - 4 * sigmas_array)
    lim_sup = np.max(means_array + 4 * sigmas_array)

    n_points = min(1000, int((lim_sup - lim_inf) * 100))
    x = np.linspace(lim_inf, lim_sup, n_points)

    weights_array = np.array(weights)
    individual_densities = np.array(
        [norm.pdf(x, means[i], sigmas[i]) for i in range(K)]
    )
    mixture_density = np.sum(
        weights_array[:, np.newaxis] * individual_densities, axis=0
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mixture_density,
            mode="lines",
            showlegend=False,
            line={"width": 3, "color": "black"},
            name="Mixture Density",
        )
    )

    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
    for i in range(K):
        individual_density = weights[i] * individual_densities[i]
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=individual_density,
                mode="lines",
                name=f"N({means[i]}, {sigmas[i]}) (w = {weights[i]})",
                line={"dash": "dash", "width": 2, "color": color},
            )
        )

    fig.update_layout(
        title=f"Mixture of {K} Gaussians",
        xaxis_title="x",
        yaxis_title="f(x)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=600,
        width=1000,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    fig.update_xaxes(gridcolor="lightgray", gridwidth=1)
    fig.update_yaxes(gridcolor="lightgray", gridwidth=1)

    fig.write_image(figure_dir / "generator_density.png")

    # plot histogram of simulated data
    fig_hist = go.Figure()

    # n_bins = min(50, max(10, int(np.ceil(np.log2(n) + 1))))
    n_bins = 100

    fig_hist.add_trace(
        go.Histogram(
            x=data,
            nbinsx=n_bins,
            histnorm="probability density",
            name="Simulated Data",
            opacity=0.7,
            marker_color="lightblue",
            marker_line_color="darkblue",
            marker_line_width=1,
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
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=600,
        width=1000,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    fig_hist.update_xaxes(gridcolor="lightgray", gridwidth=1)
    fig_hist.update_yaxes(gridcolor="lightgray", gridwidth=1)

    fig_hist.write_image(figure_dir / "data_histogram.png")


if __name__ == "__main__":
    np.random.seed(0)

    n = 600
    sigmas = [1.0, 1.0]
    means = [-5.0, 5.0]
    weights = [0.5, 0.5]
    generate_data(n, means, weights, sigmas, "example_1")

    sigmas = [1.0, 1.0, 1.0, 1.0]
    means = [-2.0, 0.0, 3.0, 5.0]
    weights = [0.2, 0.3, 0.1, 0.4]
    generate_data(n, means, weights, sigmas, "example_2")

    sigmas = [0.5, 1.0, 1.5, 2.0]
    means = [-2.0, 0.0, 3.0, 5.0]
    weights = [0.2, 0.3, 0.1, 0.4]
    generate_data(n, means, weights, sigmas, "example_3")

    sigmas = [0.8, 1.2, 0.6, 1.8, 0.4]
    means = [-2.0, 0.0, 3.0, 5.0, 15.0]
    weights = [0.2, 0.3, 0.1, 0.35, 0.05]
    generate_data(n, means, weights, sigmas, "example_4")
