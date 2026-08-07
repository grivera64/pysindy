"""
Adapter to scipy.integrate.solve_ivp.
"""
import warnings

import numpy as np
from scipy.integrate import solve_ivp as _scipy_solve_ivp

from ._result import IntegratorResult
from .base import BaseIntegrator


class ScipyIntegrator(BaseIntegrator):
    """
    Adapter wrapping :func:`scipy.integrate.solve_ivp`.

    Delegates integration to scipy's adaptive IVP solvers (RK45, RK23,
    DOP853, Radau, BDF, LSODA). The method is selected via the ``method``
    keyword argument.

    Note
    ----
    ``apply_constraints`` is not supported because scipy's ``solve_ivp``
    does not expose a per-step hook. A ``NotImplementedError`` is raised
    if a constraint is supplied.
    """

    _default_kwargs = {"method": "LSODA", "rtol": 1e-12, "atol": 1e-12}

    def solve_ivp(
        self, rhs, t, x0, apply_constraints=None, callback=None, **kwargs
    ) -> IntegratorResult:
        if callback is not None:
            # raise NotImplementedError(
            #     "apply_constraints is not supported for ScipyIntegrator. "
            #     "Use a per-step integrator (e.g. RK4) instead."
            # )
            warnings.warn(
                "callback is not supported for ScipyIntegrator. "
                "Use a per-step integrator (e.g. RK4) instead."
            )
        _rhs = rhs
        def rhs(t, x):
            if apply_constraints is not None:
                x = apply_constraints(t, x)
            return _rhs(t, x)
        t = np.asarray(t, dtype=float)
        kwargs = {**self._default_kwargs, **kwargs}

        # scipy.solve_ivp requires a 1-D state.  For PDE-shaped x0
        # (e.g. ``(n_x, n_y, n_features)``) we flatten before integration
        # and restore the shape inside a wrapper so the rhs still
        # receives the original spatial structure.
        original_shape = np.asarray(x0).shape
        if x0.ndim != 1:
            x0_flat = np.ravel(x0)

            def rhs_flat(t, x_flat):
                x = np.reshape(x_flat, original_shape)
                return np.ravel(rhs(t, x))

            sol = _scipy_solve_ivp(
                rhs_flat,
                (t[0], t[-1]),
                x0_flat,
                t_eval=t,
                **kwargs,
            )
            # Restore spatial shape: (n_t, *spatial, n_features)
            x_out = sol.y.T.reshape((sol.y.shape[1],) + original_shape)
        else:
            sol = _scipy_solve_ivp(
                rhs,
                (t[0], t[-1]),
                x0,
                t_eval=t,
                **kwargs,
            )
            x_out = sol.y.T

        self.n_steps_ = sol.nfev
        return IntegratorResult(
            t=sol.t,
            x=x_out,
            success=sol.success,
            message=sol.message,
            n_steps=sol.nfev,
        )