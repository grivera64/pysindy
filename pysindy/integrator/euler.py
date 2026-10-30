import typing

import numpy as np

from .integrator import Integrator

class EulerIntegrator(Integrator):
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
            sub_dt = dt / substeps
            for sub_index in range(substeps):
                sub_t = prev_t + sub_dt * sub_index
                x += sub_dt * f(sub_t, x)
            X[..., index, :] = x[...]

        return X


