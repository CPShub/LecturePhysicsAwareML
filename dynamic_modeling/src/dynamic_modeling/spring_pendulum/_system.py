import equinox as eqx
import jax.numpy as jnp
from typing import Literal
from jaxtyping import Array

# TODO: Add dissipation


class SpringPendulum(eqx.Module):
    """
    Derivative function (i.e., the right-hand side of the ODE) for the
    spring pendulum system in cartesian or polar coordinates.
    This computes the derivatives of the state variables given the current state.
    """

    k: float = eqx.field(static=True)  #: Spring constant
    m: float = eqx.field(static=True)  #: Mass of the pendulum bob
    g: float = eqx.field(static=True)  #: Gravitational acceleration
    l0: float = eqx.field(static=True)  #: Natural length of the spring
    c: float = eqx.field(static=True)  #: Damping coefficient
    coordinates: Literal[
        "cartesian", "polar"
    ]  #: Coordinate system used ('cartesian' or 'polar')

    def __init__(
        self,
        k: float,
        m: float,
        g: float,
        l0: float,
        c: float = 0.0,
        coordinates: Literal["cartesian", "polar"] = "cartesian",
    ):
        self.k = k
        self.m = m
        self.g = g
        self.l0 = l0
        self.c = c
        self.coordinates = coordinates

    def cartesian_derivative(self, t, y, u):
        qx, qy, vx, vy = y
        length = jnp.sqrt(qx**2 + qy**2)
        qx_t = vx
        qy_t = vy
        vx_t = -self.k / self.m * (1 - self.l0 / length) * qx
        vy_t = -self.k / self.m * (1 - self.l0 / length) * qy - self.g
        return jnp.stack([qx_t, qy_t, vx_t, vy_t])

    def polar_derivative(self, t, y, u):
        r, θ, vr, vθ = y
        r_t = vr
        θ_t = vθ
        vr_t = -(self.k / self.m) * (r - self.l0) + self.g * jnp.cos(θ) + r * vθ**2
        vθ_t = -(self.g / r) * jnp.sin(θ) - 2 * vr * vθ / r
        return jnp.stack([r_t, θ_t, vr_t, vθ_t])

    def __call__(self, t: Array | None, y: Array, u: Array | None = None) -> Array:
        if self.coordinates == "cartesian":
            return self.cartesian_derivative(t, y, u)
        elif self.coordinates == "polar":
            return self.polar_derivative(t, y, u)
        else:
            raise ValueError("Invalid coordinate system. Use 'cartesian' or 'polar'.")

    def get_kinetic_energy(self, y: Array) -> float:
        """
        Compute the total kinetic energy of the spring pendulum system.

        Args:
            ys: State array with ``shape (4,)``.

        Returns:
            Total kinetic energy of the state.
        """

        assert y.shape == (4,), "State array must be 1D with shape (4,)."

        if self.coordinates == "cartesian":
            qx, qy, vx, vy = y
            kinetic_energy = 0.5 * self.m * (vx**2 + vy**2)
        elif self.coordinates == "polar":
            r, θ, vr, vθ = y
            kinetic_energy = 0.5 * self.m * (vr**2 + (r * vθ) ** 2)
        else:
            raise ValueError("Invalid coordinate system. Use 'cartesian' or 'polar'.")

        return kinetic_energy

    def get_potential_energy(self, y: Array) -> float:
        """
        Compute the total potential energy of the spring pendulum system.
        (Sum of spring and gravitational potential energies.)

        Args:
            ys: State array with ``shape (4,)``.

        Returns:
            Total potential energy of the state.
        """

        assert y.shape == (4,), "State array must be 1D with shape (4,)."

        if self.coordinates == "cartesian":
            qx, qy, vx, vy = y
            gravitational_energy = self.m * self.g * qy
            spring_energy = 0.5 * self.k * ((qx**2 + qy**2) ** 0.5 - self.l0) ** 2
        elif self.coordinates == "polar":
            r, θ, vr, vθ = y
            gravitational_energy = -self.m * self.g * r * jnp.cos(θ)
            spring_energy = 0.5 * self.k * (r - self.l0) ** 2
        else:
            raise ValueError("Invalid coordinate system. Use 'cartesian' or 'polar'.")

        return gravitational_energy + spring_energy

    def get_energy(self, y: Array) -> float:
        """
        Compute the total mechanical energy of the spring pendulum system.

        Args:
            ys: State array with ``shape (4,)``.

        Returns:
            Total mechanical energy of the state.
        """
        kin = self.get_kinetic_energy(y)
        pot = self.get_potential_energy(y)

        return kin + pot

    def get_lagrangian(self, y: Array) -> float:
        """
        Compute the Lagrangian of the state.
        (Kinetic minus potential energy.)

        Args:
            ys: State array with ``shape (4,)``.

        Returns:
            Lagrangian of the state.
        """
        kin = self.get_kinetic_energy(y)
        pot = self.get_potential_energy(y)

        return kin - pot