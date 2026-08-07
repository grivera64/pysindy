"""
Adapter to scipy.integrate.odeint.
"""
import warnings

import numpy as np
from scipy.integrate import odeint as _scipy_odeint

from ._result import IntegratorResult
from .base import BaseIntegrator


class OdeintIntegrator(BaseIntegrator):
    """
    Adapter wrapping :func:`scipy.integrate.odeint`.

    Delegates integration to scipy's ``odeint`` (a wrapper of the
    Fortran LSODA solver from ODEPACK).

    Note
    ----
    ``apply_constraints`` is not supported because ``odeint`` does not
    expose a per-step hook. A ``NotImplementedError`` is raised if a
    constraint is supplied.
    """

    _default_kwargs = {"rtol": 1e-12, "atol": 1e-12}

    def solve_ivp(
        self, rhs, t, x0, apply_constraints=None, callback=None, **kwargs
    ) -> IntegratorResult:
        if apply_constraints is not None:
            raise NotImplementedError(
                "apply_constraints is not supported for OdeintIntegrator. "
                "Use a per-step integrator (e.g. RK4) instead."
            )
        if x0.ndim != 1:
            raise ValueError(
                "OdeintIntegrator requires a 1-D initial condition x0, "
                f"got shape {x0.shape}. For PDEs with spatial structure, "
                "use RK4Integrator instead."
            )
        t = np.asarray(t, dtype=float)
        kwargs = {**self._default_kwargs, **kwargs}
        if "method" in kwargs:
            warnings.warn(
                "odeint always uses LSODA internally, "
                "ignoring 'method' kwarg."
            )
            kwargs.pop("method")
        x = _scipy_odeint(rhs, x0, t, tfirst=True, **kwargs)
        self.n_steps_ = len(t)
        return IntegratorResult(
            t=t,
            x=x,
            success=True,
            message="The solver successfully reached the end of the "
            "integration interval.",
            n_steps=len(t),
        )