import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array


class TwoMassOscillator(eqx.Module):
    """
    Derivative function (i.e., the right-hand side of the ODE) for the
    two mass oscillator system.
    This computes the derivatives of the state variables given the current state.
    """

    k1: float = eqx.field(static=True)  #: Spring constant 1
    k2: float = eqx.field(static=True)  #: Spring constant 2
    m1: float = eqx.field(static=True)  #: Mass 1
    m2: float = eqx.field(static=True)  #: Mass 2
    c1: float = eqx.field(static=True)  #: Damping coefficient 1
    c2: float = eqx.field(static=True)  #: Damping coefficient 2

    def __init__(
        self,
        k1: float,
        k2: float,
        m1: float,
        m2: float,
        c1: float = 0.0,
        c2: float = 0.0,
    ):
        self.k1 = k1
        self.k2 = k2
        self.m1 = m1
        self.m2 = m2
        self.c1 = c1
        self.c2 = c2

    def __call__(self, t: Array | None, y: Array, u: Array | None = None) -> Array:
        q1, q2, v1, v2 = y
        q1_t = v1
        q2_t = v2
        v1_t = (
            -(self.k1 + self.k2) * q1
            + self.k2 * q2
            - (self.c1 + self.c2) * v1
            + self.c2 * v2
        ) / self.m1
        v2_t = (self.k2 * q1 - self.k2 * q2 + self.c2 * v1 - self.c2 * v2) / self.m2
        return jnp.stack([q1_t, q2_t, v1_t, v2_t])

    def get_kinetic_energy(self, y: Array) -> float:
        """
        Compute the total kinetic energy of the spring pendulum system.

        Args:
            ys: State array with ``shape (4,)``.

        Returns:
            Total kinetic energy of the state.
        """

        assert y.shape == (4,), "State array must be 1D with shape (4,)."

        q1, q2, v1, v2 = y
        kinetic_energy = 0.5 * self.m1 * v1**2 + 0.5 * self.m2 * v2**2

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

        q1, q2, v1, v2 = y
        potential_energy = 0.5 * self.k1 * q1**2 + 0.5 * self.k2 * (q2 - q1) ** 2

        return potential_energy

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
