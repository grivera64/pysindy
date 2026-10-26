import abc
import typing

import numpy as np


class Integrator(abc.ABC):
    @abc.abstractmethod
    def integrate(
        self,
        f: typing.Callable[[float, np.ndarray], np.ndarray],
        x0: np.ndarray,
        t: np.ndarray,
        **kwargs: typing.Any,
    ) -> np.ndarray:
        r"""
        Integrates the following Initial Value Problem (IVP) over discrete time starting at (t0, x0).

        .. math::
            x(t) = x_0 + \int _ {t_0} ^ {t} {f(\tau, x) d\tau}

        where x0 is a 1D vector of states at the starting state of time and f(t, x) returns
        a 1D vector of time derivatives of each state x at time t.

        Parameters
        ----------
        f: callable
            The right-hand side vector field of the system for the state vector x at time t.
            It returns a 1D numpy array of time derivatives :math:`dx/dt = f(t, x)` of shape
            `(n_features,)`

        x0: numpy array, shape (n_features,)
            The initial state vector at time t0

        t: numpy array,  shape (n_samples,)
            The times for which to return samples of the solution.

        **kwargs: dict, optional (default {})
            Optional keyword arguments to pass into the integrator.

        Returns
        -------
        x: numpy array, shape (n_samples, n_features)
            Integration results
        """
        pass

    def __call__(
        self,
        f: typing.Callable[[float, np.ndarray], np.ndarray],
        x0: np.ndarray,
        t: np.ndarray,
        **kwargs: typing.Any,
    ) -> np.ndarray:
        """
        An alias for :meth:`integrate`.

        Parameters
        ----------
        f: callable
            The right-hand side vector field of the system for the state vector x at time t.
            It returns a 1D numpy array of time derivatives :math:`dx/dt = f(t, x)` of shape
            `(n_features,)`

        x0: numpy array, shape (n_features,)
            The initial state vector at time t0

        t: numpy array,  shape (n_samples,)
            The times for which to return samples of the solution.

        **kwargs: dict, optional (default {})
            Optional keyword arguments to pass into the integrator.

        Returns
        -------
        x: numpy array, shape (n_samples, n_features)
            Integration results
        """
        return self.integrate(f, x0, t, **kwargs)
