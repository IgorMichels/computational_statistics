from dataclasses import dataclass

import numpy as np


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

    def relabel(self, placebo: bool = False) -> "State":
        """Relabel the state by sorting components by their means.

        This function ensures identifiability by ordering components
        according to their mean values.

        Returns:
            Relabeled state with components sorted by mean.
        """
        if placebo:
            return self

        order = np.argsort(self.mu)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return State(inv[self.z], self.pi[order], self.mu[order])
