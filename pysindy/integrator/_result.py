"""
Result type returned by integrators.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class IntegratorResult:
    """
    Container for the result of an integration.

    Attributes
    ----------
    t: numpy array, shape (n_samples,)
        Time points at which the solution was evaluated.

    x: numpy array, shape (n_samples, n_features)
        Solution evaluated at ``t``.

    success: bool
        Whether the integration reached the end of the interval.

    message: str
        Human-readable description of the termination reason.

    n_steps: int
        Number of integration steps performed.
    """

    t: np.ndarray
    x: np.ndarray
    success: bool
    message: str
    n_steps: int