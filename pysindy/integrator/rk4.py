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

    Parameters
    ----------
    substeps : int, optional (default 1)
        Number of internal RK4 substeps between each pair of output
        time points.  Increasing this reduces the effective timestep
        without changing the output grid, which helps stability for
        stiff systems (e.g. diffusion on fine spatial grids).

    Note
    ----
    RK4 is explicit and conditionally stable. For stiff systems,
    either increase ``substeps`` or use an implicit integrator such
    as :class:`pysindy.integrator.ScipyIntegrator` with
    ``method="BDF"`` or ``method="Radau"``.
    """

    _default_kwargs = {"substeps": 1}

    def solve_ivp(
        self, rhs, t, x0, apply_constraints=None, **kwargs
    ) -> IntegratorResult:
        kwargs = {**self._default_kwargs, **kwargs}
        substeps = kwargs.pop("substeps", 1)
        if substeps < 1:
            raise ValueError("substeps must be a positive integer")
        t = np.asarray(t, dtype=float)
        # Preserve the spatial shape of x0: output is
        # (n_samples, *spatial, n_features) so that apply_constraints
        # can act on the spatial grid (e.g. boundary conditions).
        x = np.zeros((len(t), *x0.shape), dtype=x0.dtype)
        x[0] = x0
        n_steps = 0

        def _diverged(i_step, n_step):
            x[i_step:] = np.nan
            self.n_steps_ = n_step
            return IntegratorResult(
                t=t,
                x=x,
                success=False,
                message=(
                    "RK4 integration diverged (non-finite state) "
                    f"at t={t[i_step - 1]:.6g} after {n_step} steps. "
                    "Consider increasing `substeps` or using an "
                    "implicit integrator (e.g. method='Radau')."
                ),
                n_steps=n_step,
            )

        for i in range(1, len(t)):
            h = (t[i] - t[i - 1]) / substeps
            x_curr = x[i - 1]
            for _ in range(substeps):
                # The rhs may raise (e.g. sklearn rejecting Inf produced
                # by a library transform such as 1/u at a zero crossing)
                # or return non-finite values when the state diverges.
                # Either way, treat it as divergence and stop early with
                # a clear message instead of letting the error propagate.
                try:
                    k1 = rhs(t[i - 1], x_curr)
                    k2 = rhs(t[i - 1] + h / 2, x_curr + h * k1 / 2)
                    k3 = rhs(t[i - 1] + h / 2, x_curr + h * k2 / 2)
                    k4 = rhs(t[i - 1] + h, x_curr + h * k3)
                    x_curr = x_curr + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                except (ValueError, FloatingPointError, OverflowError):
                    n_steps += 1
                    return _diverged(i, n_steps)
                n_steps += 1

                if apply_constraints is not None:
                    x_curr = apply_constraints(t[i], x_curr)

                if not np.all(np.isfinite(x_curr)):
                    return _diverged(i, n_steps)

            x[i] = x_curr

        self.n_steps_ = n_steps
        return IntegratorResult(
            t=t,
            x=x,
            success=True,
            message="The solver successfully reached the end of the "
            "integration interval.",
            n_steps=n_steps,
        )