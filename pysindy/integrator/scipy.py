import typing

import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

from .integrator import Integrator


class SolveIVPIntegrator(Integrator):
    """Integrates :math:`dx/dt = f(t, x)` using :func:`scipy.integrate.solve_ivp`.

    Defaults to ``method="LSODA"`` and ``rtol=atol=1e-12``. Additional
    keyword arguments are passed to solve_ivp, e.g. ``max_step``,
    ``first_step``, or ``jac``/``jac_sparsity`` (Radau/BDF/LSODA only);
    see :func:`scipy.integrate.solve_ivp` for the full list of options.
    ``t_span`` and ``t_eval`` are set by this wrapper, and ``dense_output``
    and ``events`` are not supported.
    """

    def integrate(
        self,
        f: typing.Callable[[float, np.ndarray], np.ndarray],
        x0: np.ndarray,
        t: np.ndarray,
        method: str = "LSODA",
        rtol: float = 1e-12,
        atol: float = 1e-12,
        **kwargs: typing.Any,
    ) -> np.ndarray:
        result = solve_ivp(
            f,
            (t[0], t[-1]),
            x0,
            t_eval=t,
            method=method,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )
        if not result.success:
            raise RuntimeError(result.message)
        return result.y.T


class ODEIntegrator(Integrator):
    """Integrates :math:`dx/dt = f(t, x)` using :func:`scipy.integrate.odeint`.

    Uses ``tfirst=True`` and leaves odeint's own tolerances in place unless
    ``rtol``/``atol`` are passed. Additional keyword arguments are passed
    to odeint, e.g. ``mxstep`` or ``tcrit``; see
    :func:`scipy.integrate.odeint` for the full list of options.
    ``full_output`` is not supported.
    """

    def integrate(
        self,
        f: typing.Callable[[float, np.ndarray], np.ndarray],
        x0: np.ndarray,
        t: np.ndarray,
        rtol: typing.Optional[float] = None,
        atol: typing.Optional[float] = None,
        **kwargs: typing.Any,
    ) -> np.ndarray:
        if kwargs.get("method") == "LSODA":
            kwargs.pop("method")
        return odeint(f, x0, t, tfirst=True, rtol=rtol, atol=atol, **kwargs)
