"""
Example 2: Configuration
========================

All parameters for the Ornstein-Uhlenbeck Process example.
SDE: dX_t = -mu * X_t * dt + sqrt(1 + mu^2) * dW_t, where mu in [0.5, 2.0]

Key properties:
- Mean-reversion rate: mu
- Diffusion coefficient: sigma(mu) = sqrt(1 + mu^2)
- Stationary distribution: N(0, (1 + mu^2) / (2*mu))
- Stationary variance is non-monotonic: minimized at mu = 1
"""

import numpy as np

# =============================================================================
# Directories
# =============================================================================
DATA_DIR = 'data_example2'
FIG_DIR = '../../manuscript/figs/example2'  # Output directly to paper figures

# =============================================================================
# SDE Parameters
# =============================================================================
X_DIM = 1                      # State dimension
MU_DIM = 1                     # Parameter dimension
X_RANGE = (-4.0, 4.0)          # State domain
MU_RANGE = (0.5, 2.0)          # Parameter domain
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
N_LABELS = 20000               # Number of labeled samples to generate
N_NEIGHBORS = 1000             # Number of nearest neighbors for score estimation
ODE_STEPS = 500                # Number of ODE discretization steps
NU_X = 1.0                     # Bandwidth for spatial (x) Gaussian kernel
NU_MU = 0.3                    # Bandwidth for parameter (mu) Gaussian kernel
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
N_SAMPLES_STAT = 20000         # Number of samples for statistics


# =============================================================================
# SDE Definition
# =============================================================================
def drift(x, mu):
    """Drift coefficient: a(x, mu) = -mu * x"""
    return -mu * x


def diffusion(x, mu):
    """Diffusion coefficient: b(x, mu) = sqrt(1 + mu^2)"""
    return np.sqrt(1 + mu**2) * np.ones_like(x)


# =============================================================================
# Analytical Solutions
# =============================================================================
def stationary_variance(mu):
    """Stationary variance: (1 + mu^2) / (2*mu)"""
    return (1 + mu**2) / (2 * mu)


def exact_conditional_mean(x, mu, dt):
    """Exact E[X_{n+1} | X_n = x, mu] = x * exp(-mu * dt)"""
    return x * np.exp(-mu * dt)


def exact_conditional_var(mu, dt):
    """
    Exact Var[X_{n+1} | X_n = x, mu]
    = (1 + mu^2) / (2*mu) * (1 - exp(-2*mu*dt))
    """
    return (1 + mu**2) / (2 * mu) * (1 - np.exp(-2 * mu * dt))


def exact_conditional_std(mu, dt):
    """Exact Std[X_{n+1} | X_n = x, mu]"""
    return np.sqrt(exact_conditional_var(mu, dt))
