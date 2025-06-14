import klax
import jax
import equinox as eqx

from jaxtyping import PRNGKeyArray, Array
from typing import Literal, Callable

ACTIVATIONS: dict[str, Callable] = dict(
    softplus=jax.nn.softplus,
    relu=jax.nn.relu,
    sigmoid=jax.nn.sigmoid,
)


class NODEDerivative(eqx.Module):
    mlp: klax.nn.MLP

    def __init__(
        self,
        *,
        state_size: int = 4,
        hidden_layer_sizes: list[int] = [16, 16],
        activation: Literal["softplus", "relu", "sigmoid"] = "softplus",
        key: PRNGKeyArray,
    ):
        self.mlp = klax.nn.MLP(
            in_size=state_size,
            out_size=state_size,
            width_sizes=hidden_layer_sizes,
            weight_init=jax.nn.initializers.variance_scaling(
                0.2, mode="fan_avg", distribution="truncated_normal"
            ),
            activation=ACTIVATIONS[activation],
            key=key,
        )

    def __call__(self, t: Array, y: Array, u: Array | None = None) -> Array:
        return self.mlp(y)
