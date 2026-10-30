from .integrator import Integrator
from .scipy import ODEIntegrator, SolveIVPIntegrator

from . import scipy

__all__ = ["Integrator", "ODEIntegrator", "SolveIVPIntegrator"]


def _assert_integrator(integrator):
    """
    Validates and returns a callable integrator.
    """
    # For backwards compatibility with existing SINDy code,
    # handle the string-named integrators.
    if isinstance(integrator, str):
        if integrator == "solve_ivp":
            return scipy.SolveIVPIntegrator()

        if integrator == "odeint":
            return scipy.ODEIntegrator()

        raise ValueError(f'Integrator "{integrator}" is not supported, exiting')

    # Uninstantiated Integrator subclasses are not valid integrators:
    # calling the class would construct an instance rather than integrate.
    if isinstance(integrator, type) and issubclass(integrator, Integrator):
        raise TypeError(
            f"integrator must be an instance, not the class "
            f"{integrator.__name__}; pass {integrator.__name__}() instead"
        )

    # All integrators are callable.
    if isinstance(integrator, Integrator) or callable(integrator):
        return integrator

    raise ValueError("Integrator is an invalid type, exiting")
