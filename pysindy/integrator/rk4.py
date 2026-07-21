"""
Fixed-step classical RK4 integrator.
"""
import numpy as np

from ._result import IntegratorResult
from .base import BaseIntegrator


class RK4Integrator(BaseIntegrator):
    """
    Fixed-step classical fourth-order Runge-Kutta integrator.

    Advances the state with the standard 4-stage RK4 scheme using the
    time grid ``t``. At each step, ``apply_constraints`` (if supplied)
    is applied to the new state before the next step begins.

    Note
    ----
    RK4 is explicit and conditionally stable. For stiff systems (e.g.
    diffusion on fine spatial grids), use an implicit integrator such
    as :class:`pysindy.integrator.ScipyIntegrator` with
    ``method="BDF"`` or ``method="Radau"``.
    """

    _default_kwargs = {}

    def solve_ivp(
        self, rhs, t, x0, apply_constraints=None, **kwargs
    ) -> IntegratorResult:
        t = np.asarray(t, dtype=float)
        # Preserve the spatial shape of x0: output is
        # (n_samples, *spatial, n_features) so that apply_constraints
        # can act on the spatial grid (e.g. boundary conditions).
        x = np.zeros((len(t), *x0.shape), dtype=x0.dtype)
        x[0] = x0
        n_steps = 0

        for i in range(1, len(t)):
            h = t[i] - t[i - 1]
            k1 = rhs(t[i - 1], x[i - 1])
            k2 = rhs(t[i - 1] + h / 2, x[i - 1] + h * k1 / 2)
            k3 = rhs(t[i - 1] + h / 2, x[i - 1] + h * k2 / 2)
            k4 = rhs(t[i - 1] + h, x[i - 1] + h * k3)
            x[i] = x[i - 1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            n_steps += 1

            if apply_constraints is not None:
                x[i] = apply_constraints(t[i], x[i])

        self.n_steps_ = n_steps
        return IntegratorResult(
            t=t,
            x=x,
            success=True,
            message="The solver successfully reached the end of the "
            "integration interval.",
            n_steps=n_steps,
        )