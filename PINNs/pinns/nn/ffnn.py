from typing import Union, Literal, Optional, Tuple
from collections.abc import Callable

import jax
from jaxtyping import Array, PRNGKeyArray

import equinox as eqx

from .dense import Dense


class FFNN(eqx.Module):
    layers: Tuple[Dense, ...]
    activations: Tuple[Callable, ...]
    final_activation: Callable
    use_bias: bool
    use_final_bias: bool
    in_features: Union[int, Literal["scalar"]]
    hidden_features: Tuple[Union[int, Literal["scalar"]], ...]
    out_features: Union[int, Literal["scalar"]]
    depth: int

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        hidden_features: Tuple[Union[int, Literal["scalar"]], ...],
        out_features: Union[int, Literal["scalar"]],
        activations: Tuple[Callable, ...],
        final_activation: Callable,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        depth = len(activations)
        assert depth == len(hidden_features), \
            f"The number of activations {len(activations)} does not match the" \
            f" number of hidden layers {len(hidden_features)}."


        keys = jax.random.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(Dense(
                in_features, 
                out_features,
                final_activation,
                use_final_bias,
                dtype=dtype,
                key=keys[0]
            ))
        else:
            layers.append(Dense(
                in_features, 
                hidden_features[0],
                activations[0],
                use_bias,
                dtype=dtype,
                key=keys[0]
            ))
            for i in range(depth - 1):
                layers.append(Dense(
                    hidden_features[i], 
                    hidden_features[i + 1],
                    activations[i + 1],
                    use_bias,
                    dtype=dtype,
                    key=keys[i + 1]
                ))
            layers.append(Dense(
                hidden_features[-1], 
                out_features,
                final_activation,
                use_final_bias,
                dtype=dtype,
                key=keys[-1]
            ))

        self.layers = tuple(layers)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.depth = depth
        self.activations = activations
        self.final_activation = final_activation

        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activations[i](x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x