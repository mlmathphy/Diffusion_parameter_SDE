"""
Example 3: Neural Network Training (Stage 3)
=============================================

2D Ornstein-Uhlenbeck Process with Rotation.

Train neural network to predict displacement given (x, mu, z).

Usage:
    python train.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from flowmap import FlowMapModel
import config as cfg

# =============================================================================
# Check Data Exists
# =============================================================================
data_files = ['x_train.npy', 'mu_train.npy', 'z_train.npy', 'y_train.npy']
for f in data_files:
    path = os.path.join(cfg.DATA_DIR, f)
    if not os.path.exists(path):
        print(f"Error: {path} not found. Please run generate_data.py first.")
        sys.exit(1)

# =============================================================================
# Main
# =============================================================================
print("=" * 60)
print("Example 4: Neural Network Training")
print("=" * 60)

# =============================================================================
# Load Labeled Data
# =============================================================================
print("\nLoading labeled data...")

x_train = np.load(os.path.join(cfg.DATA_DIR, 'x_train.npy'))
mu_train = np.load(os.path.join(cfg.DATA_DIR, 'mu_train.npy'))
z_train = np.load(os.path.join(cfg.DATA_DIR, 'z_train.npy'))
y_train = np.load(os.path.join(cfg.DATA_DIR, 'y_train.npy'))

print(f"  x_train: {x_train.shape}")
print(f"  y_train: {y_train.shape}")

# =============================================================================
# Create and Train Model
# =============================================================================
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

# Inject labeled data
model.labeled_data = {
    'x_train': x_train,
    'mu_train': mu_train,
    'z_train': z_train,
    'y_train': y_train
}

print("\n" + "=" * 60)
print("Training Neural Network")
print("=" * 60)
print(f"  Architecture: {cfg.X_DIM + cfg.MU_DIM + cfg.X_DIM} -> {cfg.HIDDEN_SIZE} x {cfg.N_HIDDEN_LAYERS} -> {cfg.X_DIM}")
print(f"  Learning rate: {cfg.LEARNING_RATE}")
print(f"  Batch size: {cfg.BATCH_SIZE}")
print(f"  Max epochs: {cfg.N_EPOCHS}")
print(f"  Early stopping patience: {cfg.PATIENCE}")

model.train_network(
    n_epochs=cfg.N_EPOCHS,
    batch_size=cfg.BATCH_SIZE,
    learning_rate=cfg.LEARNING_RATE,
    patience=cfg.PATIENCE,
    train_ratio=cfg.TRAIN_RATIO
)

# =============================================================================
# Save Model
# =============================================================================
model_path = os.path.join(cfg.DATA_DIR, 'acse_model.pt')
model.save(model_path)
print(f"Model saved to {model_path}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"\nSaved files:")
print(f"  - {cfg.DATA_DIR}/acse_model.pt")
print(f"  - {cfg.DATA_DIR}/training_curves.png")
print(f"\nNext step: python eval.py")
