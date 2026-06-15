import numpy as np
from jaxtyping import Array
import jax

def compute_POD_basis_reference_implementation(snapshots: Array, m: int) -> Array:
    """Compute a POD basis `Q` with `m` modes from a snapshot matrix.

    Args:
        snapshots: Matrix of snapshots with shape `(n, s)`, where
            n is the number of DOFs (nodes) comprising each snapshot and
            s is the number of snapshots.
        m: Number of modes, i.e., number of basis vectors to keep.
            If `m=5`, then the first 5 left singular vectors will be used to
            construct the basis matrix `Q`

    Returns:
        Tuple (Q, s) where
            Q: Matrix with shape (n, m), where each column represents one
                mode/basis vector.
    """

    U, s, V = np.linalg.svd(snapshots, full_matrices=False)
    Q = U[:, :m]
    return Q


def check_solution(func):

    for _ in range(5):
        snapshots = np.random.uniform(-10, 10, (100, 69))
        m = np.random.randint(1, 40)
        Q = func(snapshots=snapshots, m=m)

        if not isinstance(Q, (np.ndarray, jax.Array)):
            raise ValueError(f"The output of your function is not an array! I expected a numpy or jax array but got: {type(Q)}")

        if Q.shape != (100, m):
            raise ValueError(f"The shape of the output array is not what I expected. The shape of the snapshot matrix I use for testing was (100, 69) and m={m}, so I expected a basis matrix of shape (100, {m}), but I got {Q.shape}.")

        Q_true = compute_POD_basis_reference_implementation(snapshots, m)
        if np.allclose(Q, Q_true):
            print("Your implementation seems to be correct! :)")
            return
        print(
            "It appears your implementation is not correct... Try to find the issue or look up the reference implementation."
        )