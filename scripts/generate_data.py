import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

np.random.seed(0)

# constants
K = 4
n = 600

# parameters
sigma = 1
means = [-2, 0, 3, 5]
weights = [0.2, 0.3, 0.1, 0.4]

# generate data
classes = np.random.choice(range(K), size=n, p=weights)
sample_means = [means[c] for c in classes]
data = np.random.normal(loc=sample_means, scale=sigma)
np.save("../data/data.npy", data)

# plot generator density
x = np.linspace(-5.5, 8.5, 1000)
mixture_density = sum(weights[i] * norm.pdf(x, means[i], sigma) for i in range(K))

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x, y=mixture_density, mode="lines", showlegend=False, line={"width": 3}
    )
)

for i in range(K):
    individual_density = weights[i] * norm.pdf(x, means[i], sigma)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=individual_density,
            mode="lines",
            name=f"N({means[i]}, {sigma}) (w = {weights[i]})",
            line={"dash": "dash", "width": 2},
        )
    )

fig.update_layout(
    title="Mixture of 4 Gaussians",
    xaxis_title="x",
    yaxis_title="f(x)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=600,
    width=1000,
)

fig.write_image("../figures/generator_density.png")

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

fig_hist.write_image("../figures/data_histogram.png")
