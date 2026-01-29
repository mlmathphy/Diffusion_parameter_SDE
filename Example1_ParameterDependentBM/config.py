"""
Example 1: Configuration
========================

All parameters for the Parameter-Dependent Brownian Motion example.
SDE: dX_t = mu * dt + dW_t, where mu in [-1, 1]
"""

import numpy as np

# =============================================================================
# Directories
# =============================================================================
DATA_DIR = 'data_example1'
FIG_DIR = '../../manuscript/figs/example1'  # Output directly to paper figures

# =============================================================================
# SDE Parameters
# =============================================================================
X_DIM = 1                      # State dimension
MU_DIM = 1                     # Parameter dimension
X_RANGE = (-5.0, 5.0)          # State domain
MU_RANGE = (-1.0, 1.0)         # Parameter domain
DT = 0.1                       # Time step for recording (Delta t)
T = 1.0                        # Total simulation time

# =============================================================================
# Stage 1: Training Data Generation
# =============================================================================
N_MU = 21                      # Number of parameter values to sample
N_TRAJ = 5000                  # Number of trajectories per parameter value
DATA_SEED = 12345              # Random seed for data generation

# =============================================================================
# Stage 2: Label Generation (Training-Free Diffusion)
# =============================================================================
N_LABELS = 50000               # Number of labeled samples to generate
N_NEIGHBORS = 2000             # Number of nearest neighbors for score estimation
ODE_STEPS = 1000               # Number of ODE discretization steps
NU_X = 1.0                     # Bandwidth for spatial (x) Gaussian kernel
NU_MU = 0.5                    # Bandwidth for parameter (mu) Gaussian kernel
DIFF_SCALE = 3.0               # Scaling factor for displacements
LABEL_SEED = 12312414          # Random seed for label generation

# =============================================================================
# Stage 3: Neural Network Training
# =============================================================================
HIDDEN_SIZE = 128              # Hidden layer size
N_HIDDEN_LAYERS = 3            # Number of hidden layers
N_EPOCHS = 5000                # Maximum training epochs
BATCH_SIZE = 1024              # Batch size
LEARNING_RATE = 0.001          # Initial learning rate
PATIENCE = 50                  # Early stopping patience
TRAIN_RATIO = 0.9              # Train/validation split ratio
TRAIN_SEED = 12345             # Random seed for training

# =============================================================================
# Stage 4: Evaluation
# =============================================================================
EVAL_SEED = 42                 # Random seed for evaluation
N_SAMPLES_PLOT = 50000         # Number of samples for distribution plots
N_SAMPLES_STAT = 10000         # Number of samples for statistics


# =============================================================================
# SDE Definition
# =============================================================================
def drift(x, mu):
    """Drift coefficient: a(x, mu) = mu"""
    return mu * np.ones_like(x)


def diffusion(x, mu):
    """Diffusion coefficient: b(x, mu) = 1"""
    return np.ones_like(x)
