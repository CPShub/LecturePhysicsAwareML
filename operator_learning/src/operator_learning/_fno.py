import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx

from klax.nn import MLP, Linear

from jaxtyping import PRNGKeyArray, Array
from typing import Callable, Literal


class SpectralConvolution1D(eqx.Module):
    weight_real: Array
    weight_imag: Array
    num_modes: int  #: Number of nodes to retain

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_modes: int,
        *,
        key: PRNGKeyArray,
    ):
        weight_shape = (num_modes, in_channels, out_channels)

        scale = 1 / (in_channels * out_channels)
        weight_real_key, weight_imag_key = jr.split(key)

        self.weight_real = jr.uniform(
            weight_real_key, weight_shape, minval=-scale, maxval=scale
        )
        self.weight_imag = jr.uniform(
            weight_imag_key, weight_shape, minval=-scale, maxval=scale
        )

        self.num_modes = num_modes

    def __call__(self, x: Array):
        """_summary_

        Args:
            x: shape=(n, in_channels)
        """

        num_space, in_channels = x.shape

        if num_space // 2 + 1 < self.num_modes:
            raise ValueError(
                f"The data discretization is to coarse for the given number of modes. (Modes: {self.num_modes}, data points: {num_space})"
            )

        # Apply real valued fft. x_ft.shape = (num_space // 2 + 1)
        x_ft = jnp.fft.rfft(x, axis=0)
        # Truncate to num_modes
        x_ft = x_ft[: self.num_modes, :]
        # Multiply by complex weights
        weight = self.weight_real + 1j * self.weight_imag
        out_ft = jnp.einsum("xc,xio->xo", x_ft, weight)
        # Inverse FFT
        out_ift = jnp.fft.irfft(out_ft, n=num_space, axis=0)

        return out_ift


class FNOLayer(eqx.Module):
    spectral_conv: SpectralConvolution1D
    linear: Linear  #: Linear part of the FNO layer
    activation: Callable

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_modes: int,
        activation: Callable = jax.nn.softplus,
        *,
        key: PRNGKeyArray,
    ):
        spectral_key, linear_key = jr.split(key)
        self.spectral_conv = SpectralConvolution1D(
            in_channels, out_channels, num_modes, key=spectral_key
        )
        self.linear = Linear(
            in_channels,
            out_channels,
            weight_init=jax.nn.initializers.variance_scaling(
                0.1, "fan_avg", "truncated_normal"
            ),
            key=linear_key,
        )
        self.activation = activation

    def __call__(self, x: Array):
        return self.activation(
            self.spectral_conv(x) + jax.vmap(self.linear)(x)
        )


class FNO(eqx.Module):
    lifting: MLP
    projection: MLP
    layers: list[FNOLayer]

    def __init__(
        self,
        in_channels: int | Literal["scalar"],
        out_channels: int | Literal["scalar"],
        num_modes: int,
        hidden_channels: int,
        depth: int = 3,
        *,
        key: PRNGKeyArray,
    ):
        p_key, q_key, l_key = jr.split(key, 3)
        self.lifting = MLP(
            in_size=in_channels,
            out_size=hidden_channels,
            width_sizes=[16, 16],
            activation=jax.nn.softplus,
            key=p_key,
        )
        self.projection = MLP(
            in_size=hidden_channels,
            out_size=out_channels,
            width_sizes=[16, 16],
            activation=jax.nn.softplus,
            key=q_key,
        )
        self.layers = [
            FNOLayer(hidden_channels, hidden_channels, num_modes, key=k)
            for k in jr.split(l_key, depth)
        ]

    def __call__(self, x):
        # Pointwise application of the lifting function
        x = jax.vmap(self.lifting)(x)
        # Iterate over FNO layers
        for layer in self.layers:
            x = layer(x)
        # Pointwise application of the projection function
        return jax.vmap(self.projection)(x)
