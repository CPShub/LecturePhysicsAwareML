from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import paramax as px
import equinox as eqx


class PINN(eqx.Module):
    """A PINN for the nonlinear Euler-Bernoulli beam."""

    nn: eqx.Module
    params: dict[float]
    u_bc_coords: px.NonTrainable | None
    u_bc_values: px.NonTrainable | None
    w_bc_coords: px.NonTrainable | None
    w_bc_values: px.NonTrainable | None
    w_x_bc_coords: px.NonTrainable | None
    w_x_bc_values: px.NonTrainable | None
    N_bc_coords: px.NonTrainable | None
    N_bc_values: px.NonTrainable | None
    M_bc_coords: px.NonTrainable | None
    M_bc_values: px.NonTrainable | None
    Q_bc_coords: px.NonTrainable | None
    Q_bc_values: px.NonTrainable | None

    def __init__(
        self,
        params: dict[float],
        bc: dict[str, Array | None],
        *,
        key: PRNGKeyArray,
    ):
        self.nn = eqx.nn.MLP("scalar", 2, 16, 2, jax.nn.tanh, key=key)
        self.params = params

        if bc["u_bc_coords"] is None:
            self.u_bc_coords = None
            self.u_bc_values = None
        else:
            self.u_bc_coords = px.NonTrainable(bc["u_bc_coords"])
            self.u_bc_values = px.NonTrainable(bc["u_bc_values"])

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

        if bc["N_bc_coords"] is None:
            self.N_bc_coords = None
            self.N_bc_values = None
        else:
            self.N_bc_coords = px.NonTrainable(bc["N_bc_coords"])
            self.N_bc_values = px.NonTrainable(bc["N_bc_values"])

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

    def __call__(self, x: Array) -> Array:
        u = self.u(x)
        w = self.w(x)
        w_x = self.w_x(x)
        N = self.N(x)
        M = self.M(x)
        Q = self.Q(x)

        return u, w, w_x, N, M, Q

    def forward(self, x: Array) -> Tuple[Array, ...]:
        x = x / self.params["L"]
        output = self.nn(x)
        u = output[0]
        w = output[1]
        return u, w

    def u(self, x: Array) -> Array:
        u, _ = self.forward(x)
        return u

    def u_x(self, x: Array) -> Array:
        return jax.grad(self.u)(x)

    def u_xx(self, x: Array) -> Array:
        return jax.grad(self.u_x)(x)

    def w(self, x: Array) -> Array:
        _, w = self.forward(x)
        return w

    def w_x(self, x: Array) -> Array:
        return jax.grad(self.w)(x)

    def w_xx(self, x: Array) -> Array:
        return jax.grad(self.w_x)(x)

    def w_xxx(self, x: Array) -> Array:
        return jax.grad(self.w_xx)(x)

    def w_xxxx(self, x: Array) -> Array:
        return jax.grad(self.w_xxx)(x)

    def N(self, x: Array) -> Array:
        return self.params["EA"] * (self.u_x(x) + 0.5 * self.w_x(x) ** 2)

    def M(self, x: Array) -> Array:
        return -self.params["EI"] * self.w_xx(x)

    def Q(self, x: Array) -> Array:
        return -self.params["EI"] * self.w_xxx(x) + self.N(x) * self.w_x(x)

    def res(self, x: Array):
        u_x = self.u_x(x)
        u_xx = self.u_xx(x)
        w_x = self.w_x(x)
        w_xx = self.w_xx(x)
        w_xxxx = self.w_xxxx(x)

        # Residuals assuming constant line loads
        ru = -self.params["EA"] * (u_xx + 0.5 * w_x**2) - self.params["f0"]
        rw = (
            -self.params["EA"] * (u_xx * w_x + u_x * w_xx + 1.5 * w_x**2 * w_xx)
            + self.params["EI"] * w_xxxx
            - self.params["q0"]
        )

        return ru, rw

    def losses(self, x):
        if self.u_bc_coords is None:
            u_bc_loss = jnp.array(0.0)
        else:
            u_bc_pred = jax.vmap(self.u)(self.u_bc_coords)
            u_bc_loss = jnp.mean((u_bc_pred - self.u_bc_values) ** 2)

        if self.w_bc_coords is None:
            w_bc_loss = jnp.array(0.0)
        else:
            w_bc_pred = jax.vmap(self.w)(self.w_bc_coords)
            w_bc_loss = jnp.mean((w_bc_pred - self.w_bc_values) ** 2)

        if self.w_x_bc_coords is None:
            w_x_bc_loss = jnp.array(0.0)
        else:
            w_x_bc_pred = jax.vmap(self.w_x)(self.w_x_bc_coords)
            w_x_bc_loss = jnp.mean((w_x_bc_pred - self.w_x_bc_values) ** 2)

        if self.N_bc_coords is None:
            N_bc_loss = jnp.array(0.0)
        else:
            N_bc_pred = jax.vmap(self.N)(self.N_bc_coords)
            N_bc_loss = jnp.mean((N_bc_pred - self.N_bc_values) ** 2)

        if self.M_bc_coords is None:
            M_bc_loss = jnp.array(0.0)
        else:
            M_bc_pred = jax.vmap(self.M)(self.M_bc_coords)
            M_bc_loss = jnp.mean((M_bc_pred - self.M_bc_values) ** 2)

        if self.Q_bc_coords is None:
            Q_bc_loss = jnp.array(0.0)
        else:
            Q_bc_pred = jax.vmap(self.Q)(self.Q_bc_coords)
            Q_bc_loss = jnp.mean((Q_bc_pred - self.Q_bc_values) ** 2)

        ru_pred, rw_pred = jax.vmap(self.res)(x)
        ru_loss = jnp.mean(ru_pred**2)
        rw_loss = jnp.mean(rw_pred**2)

        loss_dict = {
            "u": ru_loss + u_bc_loss,
            "w": rw_loss + w_bc_loss,
            "w_x": w_x_bc_loss,
            "N": N_bc_loss,
            "M": M_bc_loss,
            "Q": Q_bc_loss,
        }
        return loss_dict

    def loss(self, weights, x):
        losses = self.losses(x)
        weighted_losses = jax.tree.map(lambda x, y: x * y, losses, weights)
        loss = jax.tree.reduce(lambda x, y: x + y, weighted_losses)
        return loss


class MixedPINN(eqx.Module):
    """A PINN for the nonlinear Euler-Bernoulli beam using a mixed formulation."""

    nn_u: eqx.Module
    nn_w: eqx.Module
    nn_w_x: eqx.Module
    nn_N: eqx.Module
    nn_M: eqx.Module
    nn_Q: eqx.Module
    params: dict[float]
    u_bc_coords: px.NonTrainable | None
    u_bc_values: px.NonTrainable | None
    w_bc_coords: px.NonTrainable | None
    w_bc_values: px.NonTrainable | None
    w_x_bc_coords: px.NonTrainable | None
    w_x_bc_values: px.NonTrainable | None
    N_bc_coords: px.NonTrainable | None
    N_bc_values: px.NonTrainable | None
    M_bc_coords: px.NonTrainable | None
    M_bc_values: px.NonTrainable | None
    Q_bc_coords: px.NonTrainable | None
    Q_bc_values: px.NonTrainable | None

    def __init__(
        self,
        params: dict[float],
        bc: dict[str, Array | None],
        *,
        key: PRNGKeyArray,
    ):
        self.nn_u = eqx.nn.MLP("scalar", "scalar", 8, 1, jax.nn.tanh, key=key)
        self.nn_w = eqx.nn.MLP("scalar", "scalar", 8, 1, jax.nn.tanh, key=key)
        self.nn_w_x = eqx.nn.MLP("scalar", "scalar", 8, 1, jax.nn.tanh, key=key)
        self.nn_N = eqx.nn.MLP("scalar", "scalar", 8, 1, jax.nn.tanh, key=key)
        self.nn_M = eqx.nn.MLP("scalar", "scalar", 8, 1, jax.nn.tanh, key=key)
        self.nn_Q = eqx.nn.MLP("scalar", "scalar", 8, 1, jax.nn.tanh, key=key)
        self.params = params

        if bc["u_bc_coords"] is None:
            self.u_bc_coords = None
            self.u_bc_values = None
        else:
            self.u_bc_coords = px.NonTrainable(bc["u_bc_coords"])
            self.u_bc_values = px.NonTrainable(bc["u_bc_values"])

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

        if bc["N_bc_coords"] is None:
            self.N_bc_coords = None
            self.N_bc_values = None
        else:
            self.N_bc_coords = px.NonTrainable(bc["N_bc_coords"])
            self.N_bc_values = px.NonTrainable(bc["N_bc_values"])

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

    def __call__(self, x: Array) -> Array:
        u = self.u(x)
        w = self.w(x)
        w_x = self.w_x(x)
        N = self.N(x)
        M = self.M(x)
        Q = self.Q(x)

        return u, w, w_x, N, M, Q

    def forward(self, x: Array) -> Tuple[Array, ...]:
        x = x / self.params["L"]
        u = self.nn_u(x)
        w = self.nn_w(x)
        w_x = self.nn_w_x(x)
        N = self.nn_N(x)
        M = self.nn_M(x)
        Q = self.nn_Q(x)
        return u, w, w_x, N, M, Q

    def u(self, x: Array) -> Array:
        u, _, _, _, _, _ = self.forward(x)
        return u

    def u_x(self, x: Array) -> Array:
        return jax.grad(self.u)(x)

    def w(self, x: Array) -> Array:
        _, w, _, _, _, _ = self.forward(x)
        return w

    def w_x(self, x: Array) -> Array:
        _, _, w_x, _, _, _ = self.forward(x)
        return w_x

    def w_xx(self, x: Array) -> Array:
        return jax.grad(self.w_x)(x)

    def N(self, x: Array) -> Array:
        _, _, _, N, _, _ = self.forward(x)
        return N

    def N_x(self, x: Array) -> Array:
        return jax.grad(self.N)(x)

    def Nw_x(self, x: Array) -> Array:
        return self.N(x) * self.w_x(x)

    def Nw_x_x(self, x: Array) -> Array:
        return jax.grad(self.Nw_x)(x)

    def M(self, x: Array) -> Array:
        _, _, _, _, M, _ = self.forward(x)
        return M

    def M_x(self, x: Array) -> Array:
        return jax.grad(self.M)(x)

    def M_xx(self, x: Array) -> Array:
        return jax.grad(self.M_x)(x)

    def Q(self, x: Array) -> Array:
        _, _, _, _, _, Q = self.forward(x)
        return Q

    def res(self, x: Array):
        N = self.params["EA"] * (self.u_x(x) + 0.5 * self.w_x(x) ** 2)
        M = -self.params["EI"] * self.w_xx(x)
        Q = self.M_x(x) + self.N(x) * self.w_x(x)

        # Residuals assuming constant line loads
        ru = -self.N_x(x) - self.params["f0"]
        rw = -self.Nw_x_x(x) - self.M_xx(x) - self.params["q0"]
        rw_x = self.w_x(x) - jax.grad(self.w)(x)
        rN = N - self.N(x)
        rM = M - self.M(x)
        rQ = Q - self.Q(x)

        return ru, rw, rw_x, rN, rM, rQ

    def losses(self, x):
        if self.u_bc_coords is None:
            u_bc_loss = jnp.array(0.0)
        else:
            u_bc_pred = jax.vmap(self.u)(self.u_bc_coords)
            u_bc_loss = jnp.mean((u_bc_pred - self.u_bc_values) ** 2)

        if self.w_bc_coords is None:
            w_bc_loss = jnp.array(0.0)
        else:
            w_bc_pred = jax.vmap(self.w)(self.w_bc_coords)
            w_bc_loss = jnp.mean((w_bc_pred - self.w_bc_values) ** 2)

        if self.w_x_bc_coords is None:
            w_x_bc_loss = jnp.array(0.0)
        else:
            w_x_bc_pred = jax.vmap(self.w_x)(self.w_x_bc_coords)
            w_x_bc_loss = jnp.mean((w_x_bc_pred - self.w_x_bc_values) ** 2)

        if self.N_bc_coords is None:
            N_bc_loss = jnp.array(0.0)
        else:
            N_bc_pred = jax.vmap(self.N)(self.N_bc_coords)
            N_bc_loss = jnp.mean((N_bc_pred - self.N_bc_values) ** 2)

        if self.M_bc_coords is None:
            M_bc_loss = jnp.array(0.0)
        else:
            M_bc_pred = jax.vmap(self.M)(self.M_bc_coords)
            M_bc_loss = jnp.mean((M_bc_pred - self.M_bc_values) ** 2)

        if self.Q_bc_coords is None:
            Q_bc_loss = jnp.array(0.0)
        else:
            Q_bc_pred = jax.vmap(self.Q)(self.Q_bc_coords)
            Q_bc_loss = jnp.mean((Q_bc_pred - self.Q_bc_values) ** 2)

        r_pred = jax.vmap(self.res)(x)
        ru_loss = jnp.mean(r_pred[0] ** 2)
        rw_loss = jnp.mean(r_pred[1] ** 2)
        rw_x_loss = jnp.mean(r_pred[2] ** 2)
        rN_loss = jnp.mean(r_pred[3] ** 2)
        rM_loss = jnp.mean(r_pred[4] ** 2)
        rQ_loss = jnp.mean(r_pred[5] ** 2)

        loss_dict = {
            "u": ru_loss + u_bc_loss,
            "w": rw_loss + w_bc_loss,
            "w_x": rw_x_loss + w_x_bc_loss,
            "N": rN_loss + N_bc_loss,
            "M": rM_loss + M_bc_loss,
            "Q": rQ_loss + Q_bc_loss,
        }
        return loss_dict

    def loss(self, weights, x):
        losses = self.losses(x)
        weighted_losses = jax.tree.map(lambda x, y: x * y, losses, weights)
        loss = jax.tree.reduce(lambda x, y: x + y, weighted_losses)
        return loss


def create_pinn(*args, mixed_formulation: bool = False, **kwargs) -> eqx.Module:
    """Factory method for that returns a PINN instance."""
    if mixed_formulation:
        return MixedPINN(*args, **kwargs)
    else:
        return PINN(*args, **kwargs)
