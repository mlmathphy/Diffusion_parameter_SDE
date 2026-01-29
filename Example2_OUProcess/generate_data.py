"""
Example 2: Data Generation (Stages 1 & 2)
==========================================

Stage 1: Generate SDE trajectory training data
Stage 2: Generate labeled data using training-free diffusion

This is the slowest part of the pipeline. Run once, then train multiple times.

Usage:
    python generate_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

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
# Create Model
# =============================================================================
print("=" * 60)
print("Example 2: Data Generation")
print("SDE: dX_t = -mu * X_t * dt + sqrt(1 + mu^2) * dW_t")
print("=" * 60)

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

# =============================================================================
# Stage 1: Generate Training Data
# =============================================================================
print("\n" + "=" * 60)
print("Stage 1: Generating Training Data")
print("=" * 60)
print(f"  Parameter values: {cfg.N_MU}")
print(f"  Trajectories per parameter: {cfg.N_TRAJ}")
print(f"  Time step: dt = {cfg.DT}")
print(f"  Total time: T = {cfg.T}")

# Custom training data generation for OU process (sample from stationary distribution)
np.random.seed(cfg.DATA_SEED)

mu_values = np.linspace(cfg.MU_RANGE[0], cfg.MU_RANGE[1], cfg.N_MU)
all_x, all_y, all_mu = [], [], []

print("Generating trajectories...")
from tqdm import tqdm
for mu in tqdm(mu_values):
    # Sample initial conditions from stationary distribution
    stat_var = cfg.stationary_variance(mu)
    x_init = np.random.normal(0, np.sqrt(stat_var), (cfg.N_TRAJ, 1))

    # Simulate SDE using fine time step
    mc_dt = 0.001
    n_steps_per_dt = int(cfg.DT / mc_dt)
    n_snapshots = int(cfg.T / cfg.DT) + 1

    sigma = np.sqrt(1 + mu**2)
    x_current = x_init.copy()
    trajectory = [x_current.copy()]

    for snap in range(1, n_snapshots):
        for _ in range(n_steps_per_dt):
            dW = np.random.randn(cfg.N_TRAJ, 1) * np.sqrt(mc_dt)
            x_current = x_current + (-mu * x_current) * mc_dt + sigma * dW
        trajectory.append(x_current.copy())

    # Extract consecutive pairs
    for t_idx in range(n_snapshots - 1):
        all_x.append(trajectory[t_idx])
        all_y.append(trajectory[t_idx + 1])
        all_mu.append(mu * np.ones((cfg.N_TRAJ, 1)))

x_data = np.vstack(all_x)
y_data = np.vstack(all_y)
mu_data = np.vstack(all_mu)

print(f"\nTotal data pairs: {len(x_data)}")
print(f"  x_data shape: {x_data.shape}")
print(f"  y_data shape: {y_data.shape}")
print(f"  mu_data shape: {mu_data.shape}")

# Inject training data into model
model.training_data = {
    'x_data': x_data,
    'y_data': y_data,
    'mu_data': mu_data,
    'dt': cfg.DT
}

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
