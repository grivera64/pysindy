"""
Fixed-step Implicit integrator using Jacobian-Free Newton Krylov.
"""
import numpy as np
from scipy.optimize import root

from ._result import IntegratorResult
from .base import BaseIntegrator


class BackwardEulerIntegrator(BaseIntegrator):
    """
    Fixed-step Implicit integrator using Jacobian-Free Newton Krylov.

    Advances the state with the generalized implicit theta-scheme:
        x_{i+1} - x_i - dt * ((1 - alpha) * f(x_i) + alpha * f(x_{i+1})) = 0

    If alpha=1, this is the unconditionally stable Backward Euler method.
    If alpha=0.5, this is the non-dissipative Crank-Nicolson method.

    This equation is solved at each step using scipy.optimize.root with
    method='krylov'. Because Krylov methods are Jacobian-free, this is
    highly efficient for large PDE systems and unconditionally stable for
    stiff systems.

    At each step, ``apply_constraints`` (if supplied) is applied to the
    new state before the next step begins.

    Parameters
    ----------
    substeps : int, optional (default 1)
        Number of internal substeps between each pair of output time points.
        While implicit methods are unconditionally stable, increasing substeps
        can improve the time-stepping accuracy (truncation error) for highly
        non-linear dynamics.
    alpha : float, optional (default 1.0)
        Implicit scheme weighting parameter. alpha=1.0 for Backward Euler,
        alpha=0.5 for Crank-Nicolson.
    root_kws : dict, optional
        Keyword arguments passed to ``scipy.optimize.root``.
    """

    _default_kwargs = {"substeps": 1, "alpha": 1.0, "root_kws": {}}

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

        alpha = kwargs.pop("alpha", 1.0)
        root_kws = kwargs.pop("root_kws", {})
        
        # Merge default root_kws with user provided
        root_opts = {"method": "krylov"}
        root_opts.update(root_kws)

        X = np.zeros((len(t), *x0.shape), dtype=x0.dtype)

        def _diverged(i_step, n_step, message=""):
            X[i_step:] = np.nan
            self.n_steps_ = n_step
            return IntegratorResult(
                t=t,
                x=X,
                success=False,
                message=(
                    f"Backward Euler integration failed at t={t[i_step - 1]:.6g} "
                    f"after {n_step} steps. {message}"
                ),
                n_steps=n_step,
            )

        X[0] = apply_constraints(t[0], x0)
        n_steps = 0
        for i in range(1, len(t)):
            t_start = t[i - 1]
            t_end = t[i]
            x_curr = X[i - 1]
            h = (t_end - t_start) / substeps

            for step in range(substeps):
                t_curr = t_start + step * h
                t_next = t_start + (step + 1) * h

                f_curr = 0.0
                def residual(x_next_flat):
                    x_next = apply_constraints(t_next, x_next_flat.reshape(x_curr.shape))
                    if alpha == 1.0:
                        error_vector = x_next - x_curr - h * rhs(t_next, x_next)
                    else:
                        error_vector = x_next - x_curr - h * ((1.0 - alpha) * f_curr + alpha * rhs(t_next, x_next))
                    return error_vector.ravel()
                
                try:
                    if alpha < 1.0:
                        f_curr = rhs(t_curr, x_curr)
                    sol = root(residual, x_curr.ravel(), **root_opts)
                    if not sol.success:
                        n_steps += 1
                        return _diverged(i, n_steps, message=sol.message)
                    
                    x_curr = apply_constraints(t_next, sol.x.reshape(x_curr.shape))
                except (ValueError, FloatingPointError, OverflowError) as e:
                    n_steps += 1
                    return _diverged(i, n_steps, message=str(e))
                n_steps += 1

                if not np.all(np.isfinite(x_curr)):
                    return _diverged(i, n_steps, message="Non-finite state produced.")

            if callback is not None:
                x_curr = callback(i, t[i], x_curr)
            X[i] = x_curr

        self.n_steps_ = n_steps
        return IntegratorResult(
            t=t,
            x=X,
            success=True,
            message="The solver successfully reached the end of the integration interval.",
            n_steps=n_steps,
        )
