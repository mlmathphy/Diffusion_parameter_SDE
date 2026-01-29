"""
Example 3: Data Generation (Stages 1 & 2)
==========================================

2D Ornstein-Uhlenbeck Process with Rotation.

SDE: dX_t = -A * X_t * dt + sigma * dW_t

where A = [mu  -omega]
          [omega  mu ]

Stage 1: Generate SDE trajectory training data
Stage 2: Generate labeled data using training-free diffusion

Usage:
    python generate_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from flowmap import FlowMapModel
import config as cfg

# =============================================================================
# Setup
# =============================================================================
def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

make_folder(cfg.DATA_DIR)
make_folder(cfg.FIG_DIR)

# =============================================================================
# Custom 2D Training Data Generation
# =============================================================================
def generate_2d_training_data(mu_values, n_traj, dt, T, x_range, seed=None):
    """
    Generate training data for 2D OU with rotation.

    Uses exact sampling from the analytical transition distribution.
    """
    if seed is not None:
        np.random.seed(seed)

    all_x, all_y, all_mu = [], [], []

    n_steps = int(T / dt)

    for mu in tqdm(mu_values, desc="Generating trajectories"):
        # Sample initial conditions uniformly in the state space
        x0 = np.random.uniform(x_range[0], x_range[1], (n_traj, 2))

        # Simulate trajectories using exact transitions
        x_current = x0.copy()

        exp_neg_A = cfg.get_exp_neg_A_dt(mu, dt)
        cov = cfg.exact_conditional_cov(mu, dt)

        for step in range(n_steps):
            x_start = x_current.copy()

            # Exact transition: X_{n+1} = exp(-A*dt)*X_n + noise
            mean = x_current @ exp_neg_A.T
            noise = np.random.multivariate_normal(np.zeros(2), cov, size=n_traj)
            x_current = mean + noise

            # Store transition pairs
            all_x.append(x_start)
            all_y.append(x_current)
            all_mu.append(mu * np.ones((n_traj, 1)))

    x_data = np.vstack(all_x)
    y_data = np.vstack(all_y)
    mu_data = np.vstack(all_mu)

    return x_data, y_data, mu_data


# =============================================================================
# Main
# =============================================================================
print("=" * 60)
print("Example 4: 2D Ornstein-Uhlenbeck with Rotation")
print("SDE: dX_t = -A * X_t * dt + sigma * dW_t")
print(f"A = [mu, -omega; omega, mu], omega = {cfg.OMEGA}")
print("=" * 60)

# =============================================================================
# Stage 1: Generate Training Data
# =============================================================================
print("\n" + "=" * 60)
print("Stage 1: Generating Training Data")
print("=" * 60)
print(f"  Parameter values: {cfg.N_MU}")
print(f"  Trajectories per parameter: {cfg.N_TRAJ}")
print(f"  Time step: dt = {cfg.DT}")
print(f"  Rotation frequency: omega = {cfg.OMEGA}")
print(f"  Diffusion coefficient: sigma = {cfg.SIGMA}")

mu_values = np.linspace(cfg.MU_RANGE[0], cfg.MU_RANGE[1], cfg.N_MU)

x_data, y_data, mu_data = generate_2d_training_data(
    mu_values=mu_values,
    n_traj=cfg.N_TRAJ,
    dt=cfg.DT,
    T=cfg.T,
    x_range=cfg.X_RANGE,
    seed=cfg.DATA_SEED
)

print(f"\nTotal data pairs: {len(x_data)}")
print(f"  x_data shape: {x_data.shape}")
print(f"  y_data shape: {y_data.shape}")
print(f"  mu_data shape: {mu_data.shape}")
print(f"  x range: [{x_data.min():.3f}, {x_data.max():.3f}]")

# Save training data
np.save(os.path.join(cfg.DATA_DIR, 'x_data.npy'), x_data)
np.save(os.path.join(cfg.DATA_DIR, 'y_data.npy'), y_data)
np.save(os.path.join(cfg.DATA_DIR, 'mu_data.npy'), mu_data)
np.save(os.path.join(cfg.DATA_DIR, 'sde_dt.npy'), np.array([cfg.DT]))

print(f"Training data saved to {cfg.DATA_DIR}/")

# =============================================================================
# Stage 2: Generate Labeled Data
# =============================================================================
print("\n" + "=" * 60)
print("Stage 2: Generating Labeled Data (Training-Free Diffusion)")
print("=" * 60)
print(f"  Number of labels: {cfg.N_LABELS}")
print(f"  Neighbors: {cfg.N_NEIGHBORS}")
print(f"  ODE steps: {cfg.ODE_STEPS}")
print(f"  Bandwidths: nu_x = {cfg.NU_X}, nu_mu = {cfg.NU_MU}")

# Create model for label generation
model = FlowMapModel(
    drift_fn=cfg.drift,
    diffusion_fn=cfg.diffusion,
    x_dim=cfg.X_DIM,
    mu_dim=cfg.MU_DIM,
    x_range=cfg.X_RANGE,
    mu_range=cfg.MU_RANGE,
    dt=cfg.DT,
    T=cfg.T,
    hidden_size=cfg.HIDDEN_SIZE,
    n_hidden_layers=cfg.N_HIDDEN_LAYERS,
    diff_scale=cfg.DIFF_SCALE
)

# Inject training data
model.training_data = {
    'x_data': x_data,
    'y_data': y_data,
    'mu_data': mu_data,
    'dt': cfg.DT
}

labeled_data = model.generate_labels(
    n_labels=cfg.N_LABELS,
    n_neighbors=cfg.N_NEIGHBORS,
    ode_steps=cfg.ODE_STEPS,
    nu_x=cfg.NU_X,
    nu_mu=cfg.NU_MU,
    seed=cfg.LABEL_SEED
)

# Save labeled data
np.save(os.path.join(cfg.DATA_DIR, 'x_train.npy'), labeled_data['x_train'])
np.save(os.path.join(cfg.DATA_DIR, 'mu_train.npy'), labeled_data['mu_train'])
np.save(os.path.join(cfg.DATA_DIR, 'z_train.npy'), labeled_data['z_train'])
np.save(os.path.join(cfg.DATA_DIR, 'y_train.npy'), labeled_data['y_train'])
np.save(os.path.join(cfg.DATA_DIR, 'diff_scale.npy'), np.array([labeled_data['diff_scale']]))

print(f"Labeled data saved to {cfg.DATA_DIR}/")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Data Generation Complete!")
print("=" * 60)
print(f"\nGenerated files in {cfg.DATA_DIR}/:")
print(f"  - x_data.npy, y_data.npy, mu_data.npy  (training pairs)")
print(f"  - x_train.npy, mu_train.npy, z_train.npy, y_train.npy  (labeled data)")
print(f"\nNext step: python train.py")
