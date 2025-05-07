# Lecture "Physics-aware Machine Learning"

Lecturer: Prof. Oliver Weeger

Assistants: Dr.-Ing. Maximilian Kannapinn, Jasper O. Schommartz, Dominik K. Klein

Lecture website: [link](https://www.maschinenbau.tu-darmstadt.de/cps/cps_teaching/cps_courses/vorlesung_physikbewusstes_ml/paml_1.en.jsp)

Research group website: [link](https://www.maschinenbau.tu-darmstadt.de/cps/department_cps/index.en.jsp) 

## Description

This repository offers a collection of lecture materials and code examples accompanying the lecture "Physics-aware Machine Learning" at TU Darmstadt (module no. 16-73-4144). 

## Lecture material

### Chapter 01: Introduction
* `FFNN_introduction`: Demo of regression of simple functions with FFNNs

### Chapter 02: Physics-based modeling & simulation
* `ode_examples`: Numerical integration of ODEs in MATLAB
* `nlebb`: Finite element discretization and numerical integration of dynamic nonlinear Euler Bernoulli beam in MATLAB

### Chapter 05: Physics-aware losses
* `PINNs`: Physics-informed neural networks for the geometrically linear and nonlinear Euler-Bernoulli beam using Python and Jax

### Chapter 06: PAML for dynamic systems
* `nlebb_dynamic_batch`: Automated data generation with the non-linear Euler Bernoulli beam in MATLAB
* `NeuralODE`: Demo of augmented neural ODEs in JAX


## Additional Information

### Reproducing results for Task FFNN ROM

To reproduce the results for Task FFNN ROM in the `FFNN_ROM` directory, the following steps must be followed.

#### Step 1: Generation of SVD modes
Run the `nlebb/nlebb_dynamic_svd.m` MATLAB file for the following loads and make sure the `svdUU1.mat` and `svdUS1.mat` output files for the SVD modes are produced (They will be required in the next step.):

```MATLAB
% -------------------------------------------------------------------------
% --- DEFINE LOADS 

load_f = 0;
load_q = 0;
load = @(x,t) [load_f, load_q];    

tPer = .1;
Nx = @(t) [0 0 0 0];
Qz = @(t) [0 0 0 multiphase_multisin(2,0.085,100,1,3,t)];
My = @(t) 0;
```

#### Step 2: Data generation

Next, run `nlebb/nlebb_dynamic_proj.m` several times for different exitations `Qz`. Save the produced `q0all.txt`, `Qfall.txt`, and `QKQall.txt` files to `FFNN_ROM/data/${loadcase}` accoding to their respective load case. The excitations are:


1. Multisine train 1: `Qz = @(t) [0 0 0 multiphase_multisin(2,0.085,100,1,3,t)];`
2. Multisine train 2: `Qz = @(t) [0 0 0 multiphase_multisin(2,0.085,100,2,3,t)];`
3. Multisine train 3: `Qz = @(t) [0 0 0 multiphase_multisin(2,0.085,100,3,3,t)];`
4. Multisine test: `Qz = @(t) [0 0 0 multiphase_multisin(1,0.103,78,1,1,t)];` 
5. Sine test: `Qz = @(t) [0 0 0 1.5*sin(2*pi*6*t)];`
6. Step test: `Qz = @(t) [0 0 0 -10*(t>.2)];`
7. Dirac test: `Qz = @(t) [0 0 0 -10*(t>.5)*(t<.51)];`
8. Quasi-static test: `Qz = @(t) [0 0 0 -2*t];`

The remaining inputs are:

```MATLAB
% -------------------------------------------------------------------------
% --- INPUTS

qmode = 2;
qm = 4;
qdeim = 0;

qnn = 0;

load_f = 0;
load_q = 0;
load = @(x,t) [load_f, load_q];    

tPer = .1;
Nx = @(t) [0 0 0 0];
Qz = @(t) % INSERT EXCITATION HERE
My = @(t) 0;
```

#### Step 3: Neural network model calibration

Execute the `force.ipynb` and `potential.ipynb` notebooks in `FFNN_ROM/notebooks` with default settings and save the obtained model weights in `FFNN_ROM/data/force.weights.txt` and `FFNN_ROM/data/potential.weights.txt`, respectively.

#### Step 4: Run simulation with NN models

To run the calibrated NN models in `nlebb/nlebb_dynamic_proj.m`, set `qnn` to the desired model and run for any of the excitations in (2).