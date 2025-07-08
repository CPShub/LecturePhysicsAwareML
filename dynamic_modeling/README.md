# Tutorial: Physics aware dynamics

This folder contains two notebooks:
1. [Neural ODEs and training methods](.\notebooks\1-NODE_and_training.ipynb)
2. [Energy based models for dynamics](.\notebooks\2-Energy_based_models.ipynb)

And two additional files for further reading:
+ [A closer look at autoparametric resonance in the spring pendulum](.\notebooks\bonus-autoparametric_resonance.ipynb)
+ [An investigation of the gauge invariance of the Lagranigan](.\notebooks\gauge_invariance.py)

## Installation
If you are using `uv`, you can create a virtual environment with all dependencies and the `dynamic_modeling` package by running

```bash
uv sync
```

If you are using `pip`, you can install `dynamic_modeling` in your virtual environment by running

```bash
pip install -e .
```