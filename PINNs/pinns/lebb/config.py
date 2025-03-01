from dataclasses import dataclass

# === -------------------------------------------------------------------- === #
# Config
# === -------------------------------------------------------------------- === #

@dataclass
class Config:
    """Configuration for data generation and PINN training."""
    EI: float
    L: float
    F: float
    q: float
    bc_case: int
    dataset_size: int
    steps: int
    learning_rate: float
    batch_size: int
    weights: dict[str, float]