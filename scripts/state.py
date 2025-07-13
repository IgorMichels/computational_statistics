from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    """
    Represents the state of the Gibbs sampler.

    Attributes:
        z: Cluster assignments for each data point.
        pi: Mixture weights for each component.
        mu: Mean parameters for each component.
        sigma2: Variance parameters for each component.
    """

    z: np.ndarray
    pi: np.ndarray
    mu: np.ndarray
    sigma2: np.ndarray

    def relabel(self, placebo: bool = False) -> "State":
        """
        Relabel the state by sorting components by their means.

        Args:
            placebo: If True, return state unchanged (placebo relabeling).

        Returns:
            Relabeled state with components sorted by mean.
        """
        if placebo:
            return self

        order = np.argsort(self.mu)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return State(inv[self.z], self.pi[order], self.mu[order], self.sigma2[order])
