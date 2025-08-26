# PDE–ML Framework for ICF Capsule Heat Transport

This repository implements a computational testbed for modeling laser-driven heat transport in inertial confinement fusion (ICF) capsules. It demonstrates an end-to-end workflow that combines physics-based models, uncertainty quantification, and machine learning surrogates.

## Features
1. **PDE Solver in C (`ICF-PDE.c`)**
   - Finite-difference discretization of the heat equation.
   - Parallelized for multi-threading and GPU acceleration
   - Boundary conditions for laser-induced surface heating.
   - Outputs temperature evolution for the outer shell and fuel interior.
   - Parameters: thermal diffusivity, laser heat flux.

2. **Gaussian Process UQ in Python (`ICF-GP.py`)**
   - Trains a Gaussian process on PDE simulation data.
   - Quantifies predictive uncertainty across parameter regimes.

3. **Neural Network Surrogate in JAX (`ICF-NN.py`)**
   - Fully connected neural network for fast prediction of capsule temperatures.
   - Enables scalable surrogate modeling of PDE outputs.

## Repository Structure
- ICF-PDE.c # Parallel finite difference PDE solver
- ICF-GP.py # Gaussian process for uncertainty quantification
- ICF-NN.py # Neural network surrogate model (JAX)
- ICFdata.txt # Simulation dataset
- Report.pdf # Technical report with results and figures
- README.md


## Getting Started
1. **Compile and run PDE Solver**
   ```bash
   gcc -fopenmp ICF-PDE.c
   ./a.out
   
2. **Run UQ analysis**
   ```bash
   python ICF-GP.py
   
4. **Train Neural Network**
   ```bash
   python ICF-NN.py

## Note:
The physics solver produces sample data of no relevance to actual ICF simulations. The physics model is a demonstrator only and should not be used for design studies.
