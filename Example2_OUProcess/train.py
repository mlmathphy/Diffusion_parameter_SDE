"""
Example 2: Neural Network Training (Stage 3)
=============================================

Train the neural network using labeled data from generate_data.py.

Usage:
    python train.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt

from flowmap import FlowMapModel
import config as cfg

# =============================================================================
# Setup
# =============================================================================
np.random.seed(cfg.TRAIN_SEED)
torch.manual_seed(cfg.TRAIN_SEED)

# =============================================================================
# Check Data Exists
# =============================================================================
required_files = ['x_train.npy', 'mu_train.npy', 'z_train.npy', 'y_train.npy', 'diff_scale.npy']
for f in required_files:
    if not os.path.exists(os.path.join(cfg.DATA_DIR, f)):
        print(f"Error: {f} not found in {cfg.DATA_DIR}/")
        print("Please run generate_data.py first.")
        sys.exit(1)

# =============================================================================
# Create Model
# =============================================================================
print("=" * 60)
print("Example 2: Neural Network Training")
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
# Load Labeled Data
# =============================================================================
print("\nLoading labeled data...")

x_train = np.load(os.path.join(cfg.DATA_DIR, 'x_train.npy'))
mu_train = np.load(os.path.join(cfg.DATA_DIR, 'mu_train.npy'))
z_train = np.load(os.path.join(cfg.DATA_DIR, 'z_train.npy'))
y_train = np.load(os.path.join(cfg.DATA_DIR, 'y_train.npy'))
diff_scale = np.load(os.path.join(cfg.DATA_DIR, 'diff_scale.npy'))[0]

print(f"  x_train: {x_train.shape}")
print(f"  y_train: {y_train.shape}")

# Inject labeled data into model
model.labeled_data = {
    'x_train': x_train,
    'mu_train': mu_train,
    'z_train': z_train,
    'y_train': y_train,
    'diff_scale': diff_scale
}

# =============================================================================
# Train Neural Network
# =============================================================================
print("\n" + "=" * 60)
print("Training Neural Network")
print("=" * 60)
print(f"  Architecture: {cfg.X_DIM + cfg.MU_DIM + cfg.X_DIM} -> {cfg.HIDDEN_SIZE} x {cfg.N_HIDDEN_LAYERS} -> {cfg.X_DIM}")
print(f"  Learning rate: {cfg.LEARNING_RATE}")
print(f"  Batch size: {cfg.BATCH_SIZE}")
print(f"  Max epochs: {cfg.N_EPOCHS}")
print(f"  Early stopping patience: {cfg.PATIENCE}")

history = model.train_network(
    n_epochs=cfg.N_EPOCHS,
    batch_size=cfg.BATCH_SIZE,
    learning_rate=cfg.LEARNING_RATE,
    patience=cfg.PATIENCE,
    train_ratio=cfg.TRAIN_RATIO
)

# =============================================================================
# Save Model
# =============================================================================
model.save(os.path.join(cfg.DATA_DIR, 'acse_model.pt'))

# =============================================================================
# Plot Training Curves
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(history['train_loss'], label='Train Loss', alpha=0.8)
ax.plot(history['val_loss'], label='Val Loss', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Curves')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(history['train_loss'], label='Train Loss', alpha=0.8)
ax.plot(history['val_loss'], label='Val Loss', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_yscale('log')
ax.set_title('Training Curves (Log Scale)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(cfg.DATA_DIR, 'training_curves.png'), dpi=150)
plt.close()

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
