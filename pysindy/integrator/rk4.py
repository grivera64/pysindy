import typing

import numpy as np

from .integrator import Integrator

class RK4Integrator(Integrator):
    def integrate(
        self,
        f: typing.Callable[[float, np.ndarray], np.ndarray],
        x0: np.ndarray,
        t: np.ndarray,
        substeps: int = 1,
        **kwargs: typing.Any,
    ) -> np.ndarray:
        X = np.empty((*x0.shape[:-1], len(t), x0.shape[-1]))
        x = x0.copy()

        X[..., 0, :] = x[...]
        for index in range(1, len(t)):
            prev_t = t[index - 1]
            curr_t = t[index]
            dt = curr_t - prev_t
            for sub_index in range(substeps):
                sub_dt = dt / substeps

                offset = sub_dt * sub_index
                k1 = sub_dt * f(prev_t + offset, x)
                k2 = sub_dt * f(prev_t + offset + (0.5 * sub_dt), x + (0.5 * k1))
                k3 = sub_dt * f(prev_t + offset + (0.5 * sub_dt), x + (0.5 * k2))
                k4 = sub_dt * f(prev_t + offset + sub_dt, x + k3)
                
                x += (k1 + (2 * k2) + (2 * k3) + k4) / 6
            X[..., index, :] = x[...]

        return X


