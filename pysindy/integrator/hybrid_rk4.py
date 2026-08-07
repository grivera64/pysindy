"""
Adaptive Method-Switching Integrator (Pseudo-LSODA) for PDEs.
"""
import numpy as np
from scipy.optimize import root

from ._result import IntegratorResult
from .base import BaseIntegrator


class HybridRK4Integrator(BaseIntegrator):
    """
    Fixed-step integrator that automatically switches between Explicit (RK4) 
    and Implicit (Backward Euler JFNK) based on dynamic stiffness detection.

    Captures the spirit of LSODA without the memory overhead of multi-step 
    history arrays, making it ideal for spatially extended PDE grids.

    Parameters
    ----------
    substeps : int, optional (default 1)
        Number of internal substeps between each output time point.
    stability_limit : float, optional (default 2.5)
        The maximum allowable value of h * L (where L is the estimated local 
        Lipschitz constant) before the solver switches to the implicit method. 
        RK4's theoretical limit is ~2.78.
    root_kws : dict, optional
        Keyword arguments passed to ``scipy.optimize.root`` for the implicit steps.
    """

    _default_kwargs = {
        "substeps": 1,
        "stability_limit": 2.5, 
        "root_kws": {"method": "krylov"}
    }

    def solve_ivp(
        self, rhs, t, x0, apply_constraints=None, callback=None, **kwargs
    ) -> IntegratorResult:
        kwargs = {**self._default_kwargs, **kwargs}
        substeps = kwargs.pop("substeps", 1)
        if substeps < 1:
            raise ValueError("substeps must be a positive integer")
        
        if apply_constraints is None:
            apply_constraints = lambda _, x: x

        stability_limit = kwargs.pop("stability_limit", 2.5)
        alpha = kwargs.pop("alpha", 1.0)
        root_kws = kwargs.pop("root_kws", {})

        root_opts = {"method": "krylov"}
        root_opts.update(root_kws)

        X = np.zeros((len(t), *x0.shape), dtype=x0.dtype)
        t = np.asarray(t, dtype=float)

        def _diverged(i_step, n_step, fallback_state, message=""):
            # Preserve the last finite state instead of introducing NaNs into the
            # returned trajectory. This keeps downstream visualizations and any
            # post-processing code from blanking out the tail of the rollout.
            X[i_step:] = fallback_state
            self.n_steps_ = n_step
            return IntegratorResult(
                t=t,
                x=X,
                success=False,
                message=(
                    f"Integration failed at t={t[i_step - 1]:.6g}. {message}"
                ),
                n_steps=n_step,
            )

        X[0] = x0
        n_steps = 0
        implicit_steps_taken = 0
        for i in range(1, len(t)):
            t_start = t[i - 1]
            t_end = t[i]
            x_curr = apply_constraints(t_start, X[i - 1])
            h = (t_end - t_start) / substeps
            recovered = False

            t_curr = t_start
            for _ in range(substeps):
                t_mid = t_curr + h / 2
                t_next = t_curr + h
                
                try:
                    xdot_curr = rhs(t_curr, x_curr)
                except (ValueError, FloatingPointError, OverflowError) as e:
                    if callback is None:
                        return _diverged(
                            i,
                            n_steps,
                            x_curr,
                            message=f"Math error on base state: {str(e)}",
                        )
                    x_curr = callback(t_next, x_curr)
                    recovered = True
                    break

                is_stiff = False
                
                # 1. Attempt Explicit Step (RK4)
                try:
                    k1 = xdot_curr
                    x_k1 = apply_constraints(t_mid, x_curr + h * k1 / 2)
                    
                    k2 = rhs(t_mid, x_k1)
                    x_k2 = apply_constraints(t_mid, x_curr + h * k2 / 2)

                    k3 = rhs(t_mid, x_k2)
                    x_k3 = apply_constraints(t_next, x_curr + h * k3)

                    k4 = rhs(t_next, x_k3)
                    x_next = apply_constraints(t_next, x_curr + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4))                    
                    xdot_next = rhs(t_next, x_next)
                    
                    # Dynamic Stiffness Detection
                    dx_norm = np.max(np.abs(x_next - x_curr))
                    df_norm = np.max(np.abs(xdot_next - xdot_curr))
                    
                    lipschitz = df_norm / (dx_norm + 1e-15)
                    stiffness_factor = h * lipschitz
                    
                    if stiffness_factor > stability_limit or not np.all(np.isfinite(x_next)):
                        is_stiff = True

                except (ValueError, FloatingPointError, OverflowError, RuntimeWarning):
                    is_stiff = True

                if not is_stiff:
                    x_curr = x_next
                else:
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
                            f_curr = xdot_curr
                        sol = root(residual, x_curr.ravel(), **root_opts)
                        if not sol.success:
                            if callback is None:
                                return _diverged(
                                    i,
                                    n_steps,
                                    x_curr,
                                    message=f"Implicit solver failed: {sol.message}",
                                )
                            x_curr = callback(t_next, x_curr)
                            recovered = True
                            break
                        
                        x_curr = apply_constraints(t_next, sol.x.reshape(x_curr.shape))
                    except (ValueError, FloatingPointError, OverflowError) as e:
                        if callback is None:
                            return _diverged(
                                i,
                                n_steps,
                                x_curr,
                                message=f"Implicit solver math error: {str(e)}",
                            )
                        x_curr = callback(t_next, x_curr)
                        recovered = True
                        break
                    implicit_steps_taken += 1
                n_steps += 1

                if not np.all(np.isfinite(x_curr)):
                    if callback is None:
                        return _diverged(i, n_steps, x_curr, message="Non-finite state produced.")
                    x_curr = callback(t_next, x_curr)
                    recovered = True
                    break
                t_curr = t_next

            if callback is not None and not recovered:
                x_curr = callback(t[i], x_curr)
            X[i] = x_curr

        self.n_steps_ = n_steps
        self.implicit_steps_ = implicit_steps_taken
        
        message = (
            f"Successfully reached the end. "
            f"Took {n_steps} total steps ({implicit_steps_taken} required implicit stiff solver)."
        )
        return IntegratorResult(
            t=t,
            x=X,
            success=True,
            message=message,
            n_steps=n_steps,
        )