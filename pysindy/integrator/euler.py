"""
Fixed-step classical Euler integrator.
"""
import numpy as np

from ._result import IntegratorResult
from .base import BaseIntegrator


class EulerIntegrator(BaseIntegrator):
    """
    Fixed-step classical euler integrator.

    Advances the state with the standard euler scheme using the
    time grid ``t``. At each step, ``apply_constraints`` (if supplied)
    is applied to the new state before the next step begins.

    Parameters
    ----------
    substeps : int, optional (default 1)
        Number of internal euler substeps between each pair of output
        time points.  Increasing this reduces the effective timestep
        without changing the output grid, which helps stability for
        stiff systems (e.g. diffusion on fine spatial grids).

    Note
    ----
    Euler is explicit and conditionally stable. For stiff systems,
    either increase ``substeps`` or use an implicit integrator such
    as :class:`pysindy.integrator.ScipyIntegrator` with
    ``method="BDF"`` or ``method="Radau"``.
    """

    _default_kwargs = {"substeps": 1}

    def solve_ivp(
        self, rhs, t, x0, apply_constraints=None, callback=None, **kwargs
    ) -> IntegratorResult:
        if apply_constraints is None:
            apply_constraints = lambda _, x: x
        
        t = np.asarray(t, dtype=float)
        kwargs = {**self._default_kwargs, **kwargs}
        substeps = kwargs.pop("substeps", 1)
        if substeps < 1:
            raise ValueError("substeps must be a positive integer")

        X = np.zeros((len(t), *x0.shape), dtype=x0.dtype)

        def _diverged(i_step, n_step):
            X[i_step:] = np.nan
            self.n_steps_ = n_step
            return IntegratorResult(
                t=t,
                x=X,
                success=False,
                message=(
                    "Euler integration diverged (non-finite state) "
                    f"at t={t[i_step - 1]:.6g} after {n_step} steps. "
                    "Consider increasing `substeps` or using an "
                    "implicit integrator (e.g. method='Radau')."
                ),
                n_steps=n_step,
            )

        X[0] = x0
        n_steps = 0
        for i in range(1, len(t)):
            t_start = t[i - 1]
            t_end = t[i]
            x_i = apply_constraints(t_start, X[i - 1])
            h = (t_end - t_start) / substeps

            for step in range(substeps):
                t_curr = t_start + step * h
                t_next = t_start + (step + 1) * h
                try:
                    x_dot = rhs(t_curr, x_i)
                    x_i = apply_constraints(t_next, x_i + h * x_dot)
                except (ValueError, FloatingPointError, OverflowError):
                    n_steps += 1
                    return _diverged(i, n_steps)
                n_steps += 1

                if not np.all(np.isfinite(x_i)):
                    return _diverged(i, n_steps)

            if callback is not None:
                x_i = callback(i, t_end, x_i)
            X[i] = x_i

        self.n_steps_ = n_steps
        return IntegratorResult(
            t=t,
            x=X,
            success=True,
            message="The solver successfully reached the end of the "
            "integration interval.",
            n_steps=n_steps,
        )
