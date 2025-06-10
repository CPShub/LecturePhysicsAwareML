# TODO: Add alternative interpolation methods
# TODO: Add option to pass u as a function.
# TODO: Check the use of u=None for consistency and how to handle optional inputs better.

import equinox as eqx
import diffrax
from diffrax import backward_hermite_coefficients, CubicInterpolation

import jax
import jax.numpy as jnp

from jaxtyping import Array, PyTree
from typing import cast, Callable, TypeVar, Generic


T = TypeVar("T", bound=Callable[..., Array])

class ODESolver(eqx.Module, Generic[T]):
    R"""Integrate a function or submodule.

    This class integrates :math:`\dot{y} = \text{func}(t, y, u, [\text{funcargs}])`
    defined by a function or submodule ``func``. The integration is performed
    using `diffrax <https://docs.kidger.site/diffrax/>`_.
    This makes the ``ODESolver`` differentiable.

    :Example:
        >>> import jax.numpy as jnp
        >>> from dynax import ODESolver
        >>> def func(t, y, u):
        ...     return -y + u
        >>> model = ODESolver(func)
        >>> ts = jnp.linspace(0, 1, 100)
        >>> y0 = jnp.array([0.5, 1.0, 2.0])
        >>> us = jnp.sin(ts)  # Example input
        >>> solution = model(ts, y0, us)
        >>> print(solution.shape)
        (100, 3)
    """

    func: T
    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractStepSizeController
    max_steps: int = eqx.field(static=True)
    is_augmented: bool = eqx.field(static=True)
    augmented_ic: Array
    augmented_ic_learnable: bool = eqx.field(static=True)

    def __init__(
        self,
        func: T,
        augmentation: int | Array = 0,
        augmented_ic_learnable: bool = False,
        solver: diffrax.AbstractSolver = diffrax.Tsit5(),
        stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
            rtol=1e-6, atol=1e-6
        ),
        max_steps: int = 4096,
    ):
        """
        Args:
            func: Function or submodel to integrate. The function arguments are
                ``(t, y, u, [funcargs])`` with a scalar time ``t``, a ``N``-dimensional
                state vector ``y``, a ``m``-dimensional input vector ``u`` and an optional
                patree ``funcargs``. The function must return an ``N``-dimensional vector
                representing the state derivative.
            augmentation: If ``augmentation`` is an array, it describes the vector
                of augmented states that are added to the inital state ``y0`` before
                passing it to ``func``. If augmentation is an integer, it describes how many
                extra dimensions are to be added to the state and initializes the augmented
                initial condition to all zeros. Defaults to 0.
            augmented_ic_learnable: If ``True`` the initial condition of the augmented
                state is updated during training. Defaults to False.
            solver: Specifies the diffax solver to use for numerical integration.
                Defaults to diffrax.Tsit5().
            stepsize_controller: The diffrax stepsize controller to use for integration.
                Defaults to diffrax.PIDController( rtol=1e-6, atol=1e-6 ).
            max_steps: The maximum number of steps to take before quitting the computation
                unconditionally. Defaults to 4096.

        Raises:
            ValueError: If provided augmentation is neither an array or integer.
        """
        self.func = func
        self.augmented_ic_learnable = augmented_ic_learnable
        if isinstance(augmentation, int):
            self.augmented_ic = jnp.zeros(augmentation)
        elif eqx.is_array(augmentation):
            assert augmentation.ndim == 1, (
                "Initial condition for the augmented state must be 1-dimensional."
            )
            self.augmented_ic = augmentation
        else:
            raise ValueError(
                f"'augmentation' must be an int or an array but got {augmentation}"
            )

        self.is_augmented = self.augmented_ic.size != 0

        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.max_steps = max_steps

    def __call__(
        self, ts: Array, y0: Array, us: Array | None = None, funcargs: PyTree = None
    ) -> Array:
        """
        Args:
            ts: Array of timesteps at which the solution is evaluated, with shape ``(k,)``.
            y0: Initial condition of the system state at time ``ts[0]``.
            us: Two dimensional array of stacked input vectors at each timestep.
                Shape ``(k, m)``. Defaults to None.

        Returns:
            Solution of the ODE excluding the augmented states. Shape ``(k, n)``.
        """
        ys = self.get_augmented_trajectory(ts, y0, us, funcargs)
        if self.is_augmented:
            # Remove the augmentation dimension and return
            return ys[:, : -self.augmented_ic.size]
        else:
            return ys

    def get_solution(
        self, ts: Array, y0: Array, us: Array | None = None, funcargs: PyTree = None
    ) -> diffrax.Solution:
        """
        Args:
            ts: Array of timesteps at which the solution is evaluated, with shape ``(k,)``.
            y0: Initial condition of the system state at time ``ts[0]``.
            us: Two dimensional array of stacked input vectors at each timestep.
                Shape ``(k, m)``. Defaults to None.

        Returns:
            diffrax solution object.
        """
        # Add the augmentation dimensions to the inital state
        y0_aug = self.augmented_ic
        if not self.augmented_ic_learnable:
            y0_aug = jax.lax.stop_gradient(y0_aug)
        y0 = jnp.concat([y0, y0_aug])

        # If inputs are supplied, then interpolate them
        if us is not None:
            coeffs = backward_hermite_coefficients(ts, us)
            u_interp = CubicInterpolation(ts, coeffs)
        else:
            u_interp = None

        # Define the funtion to be intergrated
        def _func(t, y, args):
            u_interp, funcargs = args
            if u_interp is None:
                u = None
            else:
                u = u_interp.evaluate(t)

            if funcargs is None:
                return self.func(t, y, u)
            return self.func(t, y, u, funcargs)

        # Solve the ODE using diffrax
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(_func),
            self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            args=(u_interp, funcargs),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=self.stepsize_controller,
            max_steps=self.max_steps,
        )

        return solution

    def get_augmented_trajectory(
        self, ts: Array, y0: Array, us: Array | None = None, funcargs: PyTree = None
    ) -> Array:
        """
        Args:
            ts: Array of timesteps at which the solution is evaluated, with shape ``(k,)``.
            y0: Initial condition of the system state at time ``ts[0]``.
            us: Two dimensional array of stacked input vectors at each timestep.
                Shape ``(k, m)``. Defaults to None.

        Returns:
            Solution of the ODE including the augmented states. Shape ``(k, N)``.
        """
        ys = self.get_solution(ts, y0, us, funcargs).ys
        return cast(Array, ys)
