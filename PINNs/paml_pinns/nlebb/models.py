from typing import Tuple, Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import paramax
import equinox as eqx


# === -------------------------------------------------------------------- === #
# PINN
# === -------------------------------------------------------------------- === #

class PINN(eqx.Module):
    """ A PINN for the linear elastic Euler-Bernoulli beam."""
    nn: eqx.Module
    EI: float
    EA: float
    L: float
    f: float
    q: float
    u_bc_coords: paramax.NonTrainable | None
    u_bc_values: paramax.NonTrainable | None
    w_bc_coords: paramax.NonTrainable | None
    w_bc_values: paramax.NonTrainable | None
    w_x_bc_coords: paramax.NonTrainable | None
    w_x_bc_values: paramax.NonTrainable | None
    N_bc_coords: paramax.NonTrainable | None
    N_bc_values: paramax.NonTrainable | None
    M_bc_coords: paramax.NonTrainable | None
    M_bc_values: paramax.NonTrainable | None
    Q_bc_coords: paramax.NonTrainable | None
    Q_bc_values: paramax.NonTrainable | None

    def __init__(
        self,
        EI: float,
        EA: float,
        L: float,
        f: float,
        q: float,
        bc: dict[str, Array | None],
        *,
        key: PRNGKeyArray
    ):
        self.nn = eqx.nn.MLP(1, 2, 16, 2, jax.nn.tanh, key=key)
        self.EI = EI
        self.EA = EA
        self.L = L
        self.f = f
        self.q = q

        if bc["u_bc_coords"] is None:
            self.u_bc_coords = None
            self.u_bc_values = None
        else:
            self.u_bc_coords = paramax.NonTrainable(bc["u_bc_coords"].reshape(-1, 1))
            self.u_bc_values = paramax.NonTrainable(bc["u_bc_values"].reshape(-1, 1))

        if bc["w_bc_coords"] is None:
            self.w_bc_coords = None
            self.w_bc_values = None
        else:
            self.w_bc_coords = paramax.NonTrainable(bc["w_bc_coords"].reshape(-1, 1))
            self.w_bc_values = paramax.NonTrainable(bc["w_bc_values"].reshape(-1, 1))

        if bc["w_x_bc_coords"] is None:
            self.w_x_bc_coords = None
            self.w_x_bc_values = None
        else:
            self.w_x_bc_coords = paramax.NonTrainable(bc["w_x_bc_coords"].reshape(-1, 1))
            self.w_x_bc_values = paramax.NonTrainable(bc["w_x_bc_values"].reshape(-1, 1))

        if bc["N_bc_coords"] is None:
            self.N_bc_coords = None
            self.N_bc_values = None
        else:
            self.N_bc_coords = paramax.NonTrainable(bc["N_bc_coords"].reshape(-1, 1))
            self.N_bc_values = paramax.NonTrainable(bc["N_bc_values"].reshape(-1, 1))

        if bc["M_bc_coords"] is None:
            self.M_bc_coords = None
            self.M_bc_values = None
        else:
            self.M_bc_coords = paramax.NonTrainable(bc["M_bc_coords"].reshape(-1, 1))
            self.M_bc_values = paramax.NonTrainable(bc["M_bc_values"].reshape(-1, 1))
        
        if bc["Q_bc_coords"] is None:
            self.Q_bc_coords = None
            self.Q_bc_values = None
        else:
            self.Q_bc_coords = paramax.NonTrainable(bc["Q_bc_coords"].reshape(-1, 1))
            self.Q_bc_values = paramax.NonTrainable(bc["Q_bc_values"].reshape(-1, 1))

    def __call__(self, x: Array) -> Array:
        u = self.u(x)
        w = self.w(x)
        w_x = self.w_x(x)
        N = self.N(x)
        M = self.M(x)
        Q = self.Q(x)

        return u, w, w_x, N, M, Q

    def forward(self, x: Array) -> Tuple[Array, ...]:
        x = x / self.L
        output = self.nn(x)
        u = output[0]
        w = output[1]
        return u, w

    def u(self, x: Array) -> Array:
        u, _ = self.forward(x)
        return u
    
    def u_x(self, x: Array) -> Array:
        return jax.jacfwd(self.u)(x)[0]

    def u_xx(self, x: Array) -> Array:
        return jax.jacfwd(self.u_x)(x)[0]

    def w(self, x: Array) -> Array:
        _, w = self.forward(x)
        return w

    def w_x(self, x: Array) -> Array:
        return jax.jacfwd(self.w)(x)[0]

    def w_xx(self, x: Array) -> Array:
        return jax.jacfwd(self.w_x)(x)[0]

    def w_xxx(self, x: Array) -> Array:
        return jax.jacfwd(self.w_xx)(x)[0]

    def w_xxxx(self, x: Array) -> Array:
        return jax.jacfwd(self.w_xxx)(x)[0]

    def N(self, x: Array) -> Array:
        return self.EA * (self.u_x(x) + 0.5 * self.w_x(x)**2)

    def M(self, x: Array) -> Array:
        return - self.EI * self.w_xx(x)

    def Q(self, x: Array) -> Array:
        return - self.EI * self.w_xxx(x) + self.N(x) * self.w_x(x)

    def res(self, x: Array):
        u_x = self.u_x(x)
        u_xx = self.u_xx(x)
        w_x = self.w_x(x)
        w_xx = self.w_xx(x)
        w_xxxx = self.w_xxxx(x)
        
        ru = self.EA * (u_xx + 0.5 * w_x**2) + self.f # Assumes constant line load
        rw = - self.EA * (u_xx * w_x + u_x * w_xx + 1.5 * w_x**2 * w_xx) + self.EI *  w_xxxx - self.q

        return ru, rw

    def losses(self, x):
        if self.u_bc_coords is None:
            u_bc_loss = jnp.array(0.)
        else:
            u_bc_pred = jax.vmap(self.u)(self.u_bc_coords)
            u_bc_loss = jnp.mean((u_bc_pred - self.u_bc_values)**2)

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

        if self.N_bc_coords is None:
            N_bc_loss = jnp.array(0.)
        else:
            N_bc_pred = jax.vmap(self.N)(self.N_bc_coords)
            N_bc_loss = jnp.mean((N_bc_pred - self.N_bc_values)**2)

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

        ru_pred, rw_pred = jax.vmap(self.res)(x)
        ru_loss = jnp.mean(ru_pred**2)
        rw_loss = jnp.mean(rw_pred**2)

        loss_dict = {
            "u_bc": u_bc_loss,
            "w_bc": w_bc_loss,
            "w_x_bc": w_x_bc_loss,
            "N_bc": N_bc_loss,
            "M_bc": M_bc_loss,
            "Q_bc": Q_bc_loss,
            "ru": ru_loss,
            "rw": rw_loss
        }
        return loss_dict

    def loss(self, weights, x):
        losses = self.losses(x)
        weighted_losses = jax.tree.map(lambda x, y: x * y, losses, weights)
        loss = jax.tree.reduce(lambda x, y: x + y, weighted_losses)
        return loss