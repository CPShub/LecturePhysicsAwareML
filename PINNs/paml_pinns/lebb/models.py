from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import paramax as px
import equinox as eqx


class PINN(eqx.Module):
    """ A PINN for the linear elastic Euler-Bernoulli beam."""
    nn: eqx.Module
    EI: float
    L: float
    q: float
    w_bc_coords: px.NonTrainable | None
    w_bc_values: px.NonTrainable | None
    w_x_bc_coords: px.NonTrainable | None
    w_x_bc_values: px.NonTrainable | None
    M_bc_coords: px.NonTrainable | None
    M_bc_values: px.NonTrainable | None
    Q_bc_coords: px.NonTrainable | None
    Q_bc_values: px.NonTrainable | None

    def __init__(
        self,
        EI: float,
        L: float,
        q: float,
        bc: dict[str, Array],
        *,
        key: PRNGKeyArray,
    ):
        self.nn = eqx.nn.MLP("scalar", "scalar", 8, 2, jax.nn.tanh, key=key)
        self.EI = EI
        self.L = L
        self.q = q

        if bc["w_bc_coords"] is None:
            self.w_bc_coords = None
            self.w_bc_values = None
        else:
            self.w_bc_coords = px.NonTrainable(bc["w_bc_coords"])
            self.w_bc_values = px.NonTrainable(bc["w_bc_values"])

        if bc["w_x_bc_coords"] is None:
            self.w_x_bc_coords = None
            self.w_x_bc_values = None
        else:
            self.w_x_bc_coords = px.NonTrainable(bc["w_x_bc_coords"])
            self.w_x_bc_values = px.NonTrainable(bc["w_x_bc_values"])

        if bc["M_bc_coords"] is None:
            self.M_bc_coords = None
            self.M_bc_values = None
        else:
            self.M_bc_coords = px.NonTrainable(bc["M_bc_coords"])
            self.M_bc_values = px.NonTrainable(bc["M_bc_values"])

        if bc["Q_bc_coords"] is None:
            self.Q_bc_coords = None
            self.Q_bc_values = None
        else:
            self.Q_bc_coords = px.NonTrainable(bc["Q_bc_coords"])
            self.Q_bc_values = px.NonTrainable(bc["Q_bc_values"])

    def __call__(self, x: Array) -> Tuple[Array, ...]:
        w = self.w(x)
        w_x = self.w_x(x)
        M = self.M(x)
        Q = self.Q(x)
        w_xxxx = self.w_xxxx(x)

        return w, w_x, M, Q, w_xxxx

    def forward(self, x: Array) -> Array:
        x = x / self.L
        return self.nn(x)

    def w(self, x: Array) -> Array:
        return self.forward(x)

    def w_x(self, x: Array) -> Array:
        return jax.grad(self.w)(x)

    def w_xx(self, x: Array) -> Array:
        return jax.grad(self.w_x)(x)

    def M(self, x: Array) -> Array:
        return - self.EI * self.w_xx(x)

    def w_xxx(self, x: Array) -> Array:
        return jax.grad(self.w_xx)(x)

    def Q(self, x: Array) -> Array:
        return - self.EI * self.w_xxx(x) 

    def w_xxxx(self, x: Array) -> Array:
        return jax.grad(self.w_xxx)(x)

    def res(self, x: Array):
        w_xxxx = self.w_xxxx(x)
        # The following line defines the residual for a constant line load.
        # To change that in the future, the right hand side of this equation
        # must be modified.
        rw = self.EI * w_xxxx - self.q

        return rw

    def losses(self, x):
        if self.w_bc_coords is None:
            w_bc_loss = jnp.array(0.)
        else:
            w_bc_pred = jax.vmap(self.w)(self.w_bc_coords)
            w_bc_loss = jnp.mean((w_bc_pred - self.w_bc_values)**2)

        if self.w_x_bc_coords is None:
            w_x_bc_loss = jnp.array(0.)
        else:
            w_x_bc_pred = jax.vmap(self.w_x)(self.w_x_bc_coords)
            w_x_bc_loss = jnp.mean((w_x_bc_pred - self.w_x_bc_values)**2)

        if self.M_bc_coords is None:
            M_bc_loss = jnp.array(0.)
        else:
            M_bc_pred = jax.vmap(self.M)(self.M_bc_coords)
            M_bc_loss = jnp.mean((M_bc_pred - self.M_bc_values)**2)

        if self.Q_bc_coords is None:
            Q_bc_loss = jnp.array(0.)
        else:
            Q_bc_pred = jax.vmap(self.Q)(self.Q_bc_coords)
            Q_bc_loss = jnp.mean((Q_bc_pred - self.Q_bc_values)**2)

        rw_pred = jax.vmap(self.res)(x)
        rw_loss = jnp.mean(rw_pred**2)

        loss_dict = {
            "w_bc": w_bc_loss,
            "w_x_bc": w_x_bc_loss,
            "M_bc": M_bc_loss,
            "Q_bc": Q_bc_loss,
            "rw": rw_loss
        }
        return loss_dict

    def loss(self, weights, x):
        losses = self.losses(x)
        weighted_losses = jax.tree.map(lambda x, y: x * y, losses, weights)
        loss = jax.tree.reduce(lambda x, y: x + y, weighted_losses)
        return loss