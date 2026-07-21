"""
Adapter to scipy.integrate.solve_ivp.
"""
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
        self, rhs, t, x0, apply_constraints=None, **kwargs
    ) -> IntegratorResult:
        if apply_constraints is not None:
            raise NotImplementedError(
                "apply_constraints is not supported for ScipyIntegrator. "
                "Use a per-step integrator (e.g. RK4) instead."
            )
        if x0.ndim != 1:
            raise ValueError(
                "ScipyIntegrator requires a 1-D initial condition x0, "
                f"got shape {x0.shape}. For PDEs with spatial structure, "
                "use RK4Integrator instead."
            )
        t = np.asarray(t, dtype=float)
        kwargs = {**self._default_kwargs, **kwargs}
        sol = _scipy_solve_ivp(
            rhs,
            (t[0], t[-1]),
            x0,
            t_eval=t,
            **kwargs,
        )
        self.n_steps_ = sol.nfev
        return IntegratorResult(
            t=sol.t,
            x=sol.y.T,
            success=sol.success,
            message=sol.message,
            n_steps=sol.nfev,
        )