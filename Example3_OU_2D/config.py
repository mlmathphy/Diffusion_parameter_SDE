"""
Example 3: Configuration
========================

2D Ornstein-Uhlenbeck Process with Rotation.

SDE: dX_t = -A * X_t * dt + sigma * dW_t

where A = [mu  -omega]    (rotation matrix with decay)
          [omega  mu ]

Parameter: mu in [0.5, 2.0] (decay rate / mean reversion strength)
Fixed: omega = 1.0 (rotation frequency)
Fixed: sigma = 0.5 (diffusion coefficient)

Key features:
- mu > 0: stable spiral (converges to origin)
- Larger mu: faster decay, tighter spiral
- omega controls rotation speed

Exact solution:
    X_t = exp(-A*t) * X_0 + integral of exp(-A*(t-s)) * sigma * dW_s

Matrix exponential:
    exp(-A*t) = exp(-mu*t) * [cos(omega*t)   sin(omega*t)]
                             [-sin(omega*t)  cos(omega*t)]

Conditional mean:
    E[X_{n+1} | X_n = x] = exp(-A*dt) * x

Stationary distribution (for mu > 0):
    X_infty ~ N(0, Sigma_infty)
    where Sigma_infty = (sigma^2 / 2mu) * I
"""

import numpy as np

# =============================================================================
# Directories
# =============================================================================
DATA_DIR = 'data_example3'
FIG_DIR = '../../manuscript/figs/example3'  # Output directly to paper figures

# =============================================================================
# SDE Parameters
# =============================================================================
X_DIM = 2                      # State dimension (2D)
MU_DIM = 1                     # Parameter dimension
X_RANGE = (-3.0, 3.0)          # State domain (per dimension)
MU_RANGE = (0.5, 2.0)          # Parameter domain (decay rate, must be positive for stability)
OMEGA = 1.0                    # Fixed rotation frequency
SIGMA = 0.5                    # Fixed diffusion coefficient
DT = 0.1                       # Time step for recording
T = 2.0                        # Simulation time per segment

# =============================================================================
# Stage 1: Training Data Generation
# =============================================================================
N_MU = 16                      # Number of parameter values to sample
N_TRAJ = 3000                  # Number of trajectories per parameter value
DATA_SEED = 12345              # Random seed for data generation

# =============================================================================
# Stage 2: Label Generation (Training-Free Diffusion)
# =============================================================================
N_LABELS = 50000               # Number of labeled samples to generate
N_NEIGHBORS = 2000             # Number of nearest neighbors for score estimation
ODE_STEPS = 500                # Number of ODE discretization steps
NU_X = 0.8                     # Bandwidth for spatial (x) Gaussian kernel (2D needs larger)
NU_MU = 0.3                    # Bandwidth for parameter (mu) Gaussian kernel
DIFF_SCALE = 3.0               # Scaling factor for displacements
LABEL_SEED = 12312414          # Random seed for label generation

# =============================================================================
# Stage 3: Neural Network Training
# =============================================================================
HIDDEN_SIZE = 256              # Hidden layer size (larger for 2D)
N_HIDDEN_LAYERS = 4            # Number of hidden layers
N_EPOCHS = 5000                # Maximum training epochs
BATCH_SIZE = 1024              # Batch size
LEARNING_RATE = 0.001          # Initial learning rate
PATIENCE = 100                 # Early stopping patience
TRAIN_RATIO = 0.9              # Train/validation split ratio
TRAIN_SEED = 12345             # Random seed for training

# =============================================================================
# Stage 4: Evaluation
# =============================================================================
EVAL_SEED = 42                 # Random seed for evaluation
N_SAMPLES_PLOT = 10000         # Number of samples for distribution plots
N_SAMPLES_STAT = 5000          # Number of samples for statistics
N_TRAJ_PLOT = 5                # Number of trajectories to plot


# =============================================================================
# Matrix A Definition
# =============================================================================
def get_A_matrix(mu):
    """
    Return the drift matrix A for given mu.

    A = [mu   -omega]
        [omega  mu  ]
    """
    return np.array([[mu, -OMEGA],
                     [OMEGA, mu]])


def get_exp_neg_A_dt(mu, dt):
    """
    Compute exp(-A*dt) analytically.

    exp(-A*t) = exp(-mu*t) * R(-omega*t)

    where R(theta) = [cos(theta)   sin(theta)]
                     [-sin(theta)  cos(theta)]
    """
    decay = np.exp(-mu * dt)
    angle = -OMEGA * dt
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    return decay * np.array([[cos_angle, sin_angle],
                             [-sin_angle, cos_angle]])


# =============================================================================
# SDE Definition
# =============================================================================
def drift(x, mu):
    """
    Drift coefficient: a(x, mu) = -A * x

    For 2D input x of shape (n, 2), returns shape (n, 2).
    """
    x = np.atleast_2d(x)
    mu = np.atleast_1d(mu).flatten()[0] if hasattr(mu, '__len__') else mu

    A = get_A_matrix(mu)
    return -x @ A.T  # -A @ x^T transposed = -x @ A^T


def diffusion(x, mu):
    """
    Diffusion coefficient: b(x, mu) = sigma (constant, isotropic)

    Returns shape (n, 2) with constant sigma.
    """
    x = np.atleast_2d(x)
    return SIGMA * np.ones_like(x)


# =============================================================================
# Analytical Solutions
# =============================================================================
def exact_conditional_mean(x, mu, dt):
    """
    Exact E[X_{n+1} | X_n = x, mu] for 2D OU with rotation.

    E[X_{n+1}] = exp(-A*dt) * x

    Args:
        x: Starting position, shape (n, 2) or (2,)
        mu: Decay rate parameter
        dt: Time step

    Returns:
        Conditional mean, same shape as x
    """
    x = np.atleast_2d(x)
    exp_neg_A = get_exp_neg_A_dt(mu, dt)
    return x @ exp_neg_A.T


def exact_conditional_cov(mu, dt):
    """
    Exact Cov[X_{n+1} | X_n = x, mu] for 2D OU with rotation.

    For this system with isotropic noise:
    Cov = (sigma^2 / 2mu) * (1 - exp(-2*mu*dt)) * I

    (The rotation doesn't affect the covariance structure due to isotropy)

    Returns:
        2x2 covariance matrix
    """
    var = (SIGMA**2 / (2 * mu)) * (1 - np.exp(-2 * mu * dt))
    return var * np.eye(2)


def exact_conditional_var(mu, dt):
    """Exact marginal variance (same for both dimensions due to isotropy)."""
    return (SIGMA**2 / (2 * mu)) * (1 - np.exp(-2 * mu * dt))


def exact_stationary_cov(mu):
    """
    Exact stationary covariance for mu > 0.

    Sigma_infty = (sigma^2 / 2mu) * I
    """
    if mu <= 0:
        return np.inf * np.eye(2)
    return (SIGMA**2 / (2 * mu)) * np.eye(2)


def exact_stationary_var(mu):
    """Exact stationary marginal variance."""
    if mu <= 0:
        return np.inf
    return SIGMA**2 / (2 * mu)


def sample_exact_transition(x0, mu, dt, n_samples):
    """
    Sample from the exact conditional distribution.

    X_{n+1} | X_n = x0 ~ N(exp(-A*dt)*x0, Cov)

    Args:
        x0: Starting position, shape (2,) or (n, 2)
        mu: Decay rate
        dt: Time step
        n_samples: Number of samples

    Returns:
        Samples of shape (n_samples, 2)
    """
    x0 = np.atleast_2d(x0)
    if x0.shape[0] == 1:
        x0 = np.tile(x0, (n_samples, 1))

    mean = exact_conditional_mean(x0, mu, dt)
    cov = exact_conditional_cov(mu, dt)

    # Sample from multivariate normal
    noise = np.random.multivariate_normal(np.zeros(2), cov, size=n_samples)
    return mean + noise


def simulate_trajectory(x0, mu, dt, n_steps, n_traj=1):
    """
    Simulate exact trajectories using the analytical transition.

    Args:
        x0: Initial position, shape (2,)
        mu: Decay rate
        dt: Time step
        n_steps: Number of steps
        n_traj: Number of trajectories

    Returns:
        Trajectories of shape (n_traj, n_steps+1, 2)
    """
    x0 = np.atleast_1d(x0).flatten()
    trajectories = np.zeros((n_traj, n_steps + 1, 2))
    trajectories[:, 0, :] = x0

    exp_neg_A = get_exp_neg_A_dt(mu, dt)
    cov = exact_conditional_cov(mu, dt)

    for t in range(n_steps):
        mean = trajectories[:, t, :] @ exp_neg_A.T
        noise = np.random.multivariate_normal(np.zeros(2), cov, size=n_traj)
        trajectories[:, t + 1, :] = mean + noise

    return trajectories
