"""
Integrators for continuous-time dynamical systems.
"""
import inspect

from .base import BaseIntegrator
from .odeint import OdeintIntegrator
from .rk4 import RK4Integrator
from .scipy_ivp import ScipyIntegrator

from .backward_euler import BackwardEulerIntegrator
from .hybrid_rk4 import HybridRK4Integrator

#: Registry mapping integrator names to classes.
INTEGRATORS = {
    "solve_ivp": ScipyIntegrator,
    "odeint": OdeintIntegrator,
    "rk4": RK4Integrator,
    "backward_euler": BackwardEulerIntegrator,
    "hybrid_rk4": HybridRK4Integrator,
}

__all__ = [
    "BaseIntegrator",
    "ScipyIntegrator",
    "OdeintIntegrator",
    "RK4Integrator",
    "BackwardEulerIntegrator",
    "HybridRK4Integrator"
    "INTEGRATORS",
    "get_integrator",
]


def get_integrator(integrator):
    """
    Resolve an integrator from a name or class.

    Parameters
    ----------
    integrator: str or BaseIntegrator subclass
        Name of a registered integrator (e.g. ``"solve_ivp"`` or
        ``"odeint"``) or a subclass of
        :class:`pysindy.integrator.BaseIntegrator`.

    Returns
    -------
    integrator_cls: BaseIntegrator subclass
        The integrator class to instantiate.

    Raises
    ------
    ValueError
        If ``integrator`` is a string not in the registry.
    TypeError
        If ``integrator`` is neither a string nor a
        :class:`pysindy.integrator.BaseIntegrator` subclass.
    """
    if isinstance(integrator, str):
        if integrator not in INTEGRATORS:
            raise ValueError(
                f"Integrator {integrator!r} not supported. "
                f"Choose from {list(INTEGRATORS)} or pass a "
                "BaseIntegrator subclass."
            )
        return INTEGRATORS[integrator]
    if inspect.isclass(integrator) and issubclass(integrator, BaseIntegrator):
        return integrator
    raise TypeError(
        "integrator must be a string or a BaseIntegrator subclass, "
        f"got {type(integrator).__name__}"
    )