"""
Base class for integrators.
"""
import abc

import numpy as np
from sklearn.base import BaseEstimator

from ._result import IntegratorResult

class BaseIntegrator(abc.ABC):
    """
    Base class for integrators.

    Simply forces integrators to implement a
    ``solve_ivp`` function.

    Attributes:
        n_steps_: Number of integration steps performed in the last call.
    """

    n_steps_: int

    # Force subclasses to implement this
    @abc.abstractmethod
    def solve_ivp(
        self, rhs, t, x0, apply_constraints=None, callback=None, **kwargs
    ) -> IntegratorResult:
        """
        Integrate the right-hand side of a dynamical system forward in time.

        Parameters
        ----------
        rhs: callable
            Right-hand side of the system. The calling signature is
            ``rhs(t, x)``, where ``t`` is a scalar and ``x`` is an ndarray
            with ``len(x) = len(x0)``. ``rhs`` must return an array of the
            same shape as ``x``.

        t: numpy array of size [n_samples]
            Array of time points at which to evaluate the solution.

        x0: numpy array, size [n_features] or [spatial, n_features]
            Initial condition from which to integrate.

        apply_constraints: callable, optional (default None)
            Function applied to the state at the end of each numerical
            step, with signature ``apply_constraints(t, x) -> x``. Used for
            enforcing boundary conditions, applying data-assimilation
            corrections, or other per-step projections of the state.
        
        callback: callable, optional (default None)
            Function applied to the state at the end of each numerical
            step, with signature ``callback(t, x) -> x``. Used for
            enforcing boundary conditions, applying data-assimilation
            corrections, or other per-step projections of the state.

        **kwargs: dict, optional
            Optional keyword arguments to pass to the integrator.

        Returns
        -------
        result: IntegratorResult
            Container with the solution and diagnostics. See
            :class:`pysindy.integrator.IntegratorResult`.
        """
        ...