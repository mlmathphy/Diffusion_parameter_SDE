"""
Example 3: Evaluation and Visualization (Stage 4)
==================================================

2D Ornstein-Uhlenbeck Process with Rotation.

SDE: dX_t = -A * X_t * dt + sigma * dW_t

Figures:
1. Sample trajectories showing spiral behavior
2. One-step conditional distribution (2D scatter)
3. Conditional mean comparison
4. Multi-step distribution and stationary convergence

Usage:
    python eval.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import niceplots

from flowmap import FlowMapModel
import config as cfg

# =============================================================================
# Setup
# =============================================================================
np.random.seed(cfg.EVAL_SEED)
torch.manual_seed(cfg.EVAL_SEED)

# Setup niceplots with CMU fonts
plt.style.use(niceplots.get_style())
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif', 'Computer Modern Roman', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 14  # Base font size for figures
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# =============================================================================
# Check Model Exists
# =============================================================================
model_path = os.path.join(cfg.DATA_DIR, 'acse_model.pt')
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Please run train.py first.")
    sys.exit(1)

# =============================================================================
# Load Model
# =============================================================================
print("=" * 60)
print("Example 4: 2D OU with Rotation Evaluation")
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

model.load(model_path)
print("Model loaded successfully!")

# =============================================================================
# Get niceplots colors
# =============================================================================
colors = niceplots.get_colors_list()

# =============================================================================
# Figure 1: Sample Trajectories
# =============================================================================
print("\nGenerating Figure 1: Sample trajectories...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

mu_values_traj = [0.5, 1.0, 2.0]
x0 = np.array([2.0, 0.0])
n_steps = 50

for idx, mu in enumerate(mu_values_traj):
    ax = axes[idx]

    # Store final points for marking
    final_points_exact = []
    final_points_learned = []

    # Exact trajectories: wider, more transparent
    for i in range(cfg.N_TRAJ_PLOT):
        traj_exact = cfg.simulate_trajectory(x0, mu, cfg.DT, n_steps, n_traj=1)[0]
        ax.plot(traj_exact[:, 0], traj_exact[:, 1], '-', color=colors[0],
                alpha=0.4, linewidth=2)
        final_points_exact.append(traj_exact[-1])

    # Learned trajectories: thinner, darker
    for i in range(cfg.N_TRAJ_PLOT):
        traj_learned = model.generate_trajectory(x0=x0, mu=mu, n_steps=n_steps, n_samples=1)
        # Reshape to (n_steps+1, 2)
        traj_learned_full = np.zeros((n_steps + 1, 2))
        traj_learned_full[0] = x0
        x_current = x0.reshape(1, 2)
        for t in range(n_steps):
            x_next = model.sample(x=x_current, mu=mu, n_samples=1)
            traj_learned_full[t + 1] = x_next.flatten()
            x_current = x_next
        ax.plot(traj_learned_full[:, 0], traj_learned_full[:, 1], '-',
                color=colors[1], alpha=0.7, linewidth=1)
        final_points_learned.append(traj_learned_full[-1])

    # Plot final points
    final_points_exact = np.array(final_points_exact)
    final_points_learned = np.array(final_points_learned)
    ax.scatter(final_points_exact[:, 0], final_points_exact[:, 1],
               c=colors[0], s=30, marker='s', alpha=0.6, zorder=5)
    ax.scatter(final_points_learned[:, 0], final_points_learned[:, 1],
               c=colors[1], s=20, marker='s', alpha=0.8, zorder=5)

    ax.plot(x0[0], x0[1], 'ko', markersize=8, label='Start')
    # Origin: use filled circle with edge color for visibility
    ax.plot(0, 0, 'o', markersize=10, markerfacecolor='gray',
            markeredgecolor='black', markeredgewidth=1.5, label='Origin', zorder=10)
    ax.set_xlabel(r'$X_1$', fontsize=16)
    if idx == 0:
        ax.set_ylabel(r'$X_2$', fontsize=16, rotation=0, ha='right')
    else:
        # Remove y-axis for non-left panels (shared y-axis)
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_aspect('equal')
    if idx == 2:  # Show legend in rightmost panel
        ax.plot([], [], '-', color=colors[0], alpha=0.4, linewidth=2, label='Exact')
        ax.plot([], [], '-', color=colors[1], alpha=0.7, linewidth=1, label='Learned')
        ax.scatter([], [], c=colors[0], s=30, marker='s', label='End (Exact)')
        ax.scatter([], [], c=colors[1], s=20, marker='s', label='End (Learned)')
        ax.legend(fontsize=12, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(cfg.FIG_DIR, 'fig1_trajectories.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cfg.FIG_DIR}/fig1_trajectories.pdf")

# =============================================================================
# Figure 2: One-Step Conditional Distribution (2D Scatter)
# =============================================================================
print("Generating Figure 2: One-step conditional distribution...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

x0 = np.array([1.5, 0.5])
mu_values_scatter = [0.5, 1.0, 2.0]
n_samples = 2000

for idx, mu in enumerate(mu_values_scatter):
    # Top row: Exact samples
    ax = axes[0, idx]
    samples_exact = cfg.sample_exact_transition(x0, mu, cfg.DT, n_samples)
    ax.scatter(samples_exact[:, 0], samples_exact[:, 1], s=1, alpha=0.3, c=colors[0])
    ax.plot(x0[0], x0[1], 'ko', markersize=8)

    # Draw exact mean
    mean_exact = cfg.exact_conditional_mean(x0, mu, cfg.DT).flatten()
    ax.plot(mean_exact[0], mean_exact[1], 'k*', markersize=12)

    # Top row: no x-axis (shared with bottom)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)
    ax.spines['bottom'].set_visible(False)
    if idx == 0:
        ax.set_ylabel(r'$X_2$', fontsize=16, rotation=0, ha='right')
    else:
        # Remove y-axis for non-left columns
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
    ax.set_xlim([0, 2.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_aspect('equal')

    # Bottom row: Learned samples
    ax = axes[1, idx]
    samples_learned = model.sample(x=x0, mu=mu, n_samples=n_samples)
    ax.scatter(samples_learned[:, 0], samples_learned[:, 1], s=1, alpha=0.3, c=colors[1])
    ax.plot(x0[0], x0[1], 'ko', markersize=8)

    # Learned mean
    mean_learned = np.mean(samples_learned, axis=0)
    ax.plot(mean_learned[0], mean_learned[1], 'k*', markersize=12)

    # Bottom row: show x-axis labels
    ax.set_xlabel(r'$X_1$', fontsize=16)
    if idx == 0:
        ax.set_ylabel(r'$X_2$', fontsize=16, rotation=0, ha='right')
    else:
        # Remove y-axis for non-left columns
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
    ax.set_xlim([0, 2.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(cfg.FIG_DIR, 'fig2_conditional_scatter.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cfg.FIG_DIR}/fig2_conditional_scatter.pdf")

# =============================================================================
# Figure 3: Conditional Mean Comparison
# =============================================================================
print("Generating Figure 3: Conditional mean comparison...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Test multiple starting points and mu values
x0_values = [np.array([2.0, 0.0]), np.array([0.0, 2.0]), np.array([1.5, 1.5])]
mu_test_range = np.linspace(cfg.MU_RANGE[0], cfg.MU_RANGE[1], 11)

# Left: X1 component of conditional mean
ax = axes[0]
for x0_idx, x0 in enumerate(x0_values):
    means_exact_x1 = [cfg.exact_conditional_mean(x0, mu, cfg.DT).flatten()[0] for mu in mu_test_range]
    means_learned_x1 = []
    for mu in mu_test_range:
        samples = model.sample(x=x0, mu=mu, n_samples=cfg.N_SAMPLES_STAT)
        means_learned_x1.append(np.mean(samples[:, 0]))

    ax.plot(mu_test_range, means_exact_x1, color=colors[x0_idx], linewidth=3, alpha=0.5,
            label=f'Exact $x_0$=({x0[0]:.1f},{x0[1]:.1f})')
    ax.plot(mu_test_range, means_learned_x1, 'o', color=colors[x0_idx], markersize=5)

ax.set_xlabel(r'$\mu$', fontsize=16)
ax.set_ylabel(r'$\mathbb{E}[X_1^{n+1}]$', fontsize=16, rotation=0, ha='right', labelpad=40)

# Right: X2 component of conditional mean
ax = axes[1]
for x0_idx, x0 in enumerate(x0_values):
    means_exact_x2 = [cfg.exact_conditional_mean(x0, mu, cfg.DT).flatten()[1] for mu in mu_test_range]
    means_learned_x2 = []
    for mu in mu_test_range:
        samples = model.sample(x=x0, mu=mu, n_samples=cfg.N_SAMPLES_STAT)
        means_learned_x2.append(np.mean(samples[:, 1]))

    ax.plot(mu_test_range, means_exact_x2, color=colors[x0_idx], linewidth=3, alpha=0.5,
            label=f'Exact $x_0$=({x0[0]:.1f},{x0[1]:.1f})')
    ax.plot(mu_test_range, means_learned_x2, 'o', color=colors[x0_idx], markersize=5)

ax.set_xlabel(r'$\mu$', fontsize=16)
ax.set_ylabel(r'$\mathbb{E}[X_2^{n+1}]$', fontsize=16, rotation=0, ha='right', labelpad=40)
# Remove y-axis for right panel (shared y-axis)
ax.set_yticklabels([])
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)
ax.legend(fontsize=14)  # Show legend in right panel

plt.tight_layout()
plt.savefig(os.path.join(cfg.FIG_DIR, 'fig3_conditional_mean.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cfg.FIG_DIR}/fig3_conditional_mean.pdf")

# =============================================================================
# Figure 4: Multi-step Distribution and Variance
# =============================================================================
print("Generating Figure 4: Multi-step distribution and variance...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Multi-step distribution at T=2.0 (starting from fixed x0)
ax = axes[0]
x0 = np.array([2.0, 0.0])
mu_values_terminal = [0.5, 1.0, 2.0]
n_steps = 20
n_samples_terminal = 5000

for idx, mu in enumerate(mu_values_terminal):
    # Exact: simulate trajectories
    traj_exact = cfg.simulate_trajectory(x0, mu, cfg.DT, n_steps, n_traj=n_samples_terminal)
    x_T_exact = traj_exact[:, -1, :]  # Shape (n_samples, 2)

    # Learned: multi-step using generate_trajectory
    # Tile x0 to match n_samples since generate_trajectory checks shape
    x0_tiled = np.tile(x0.reshape(1, 2), (n_samples_terminal, 1))
    x_T_learned = model.generate_trajectory(x0=x0_tiled, mu=mu, n_steps=n_steps, n_samples=n_samples_terminal)

    # Plot marginal X1 distribution
    kde_exact = gaussian_kde(x_T_exact[:, 0])
    kde_learned = gaussian_kde(x_T_learned[:, 0])
    x_grid = np.linspace(-2, 2.5, 200)

    # Exact: wider, more transparent
    ax.plot(x_grid, kde_exact(x_grid), color=colors[idx], linewidth=3, alpha=0.5,
            label=f'Exact ($\\mu={mu}$)')
    # Learned: thinner, darker (same color)
    ax.plot(x_grid, kde_learned(x_grid), color=colors[idx], linewidth=1.5)

    # Print statistics
    print(f"\nmu={mu}, T={n_steps*cfg.DT}:")
    print(f"  Exact mean: ({np.mean(x_T_exact[:,0]):.4f}, {np.mean(x_T_exact[:,1]):.4f})")
    print(f"  Learned mean: ({np.mean(x_T_learned[:,0]):.4f}, {np.mean(x_T_learned[:,1]):.4f})")
    print(f"  Exact var: ({np.var(x_T_exact[:,0]):.4f}, {np.var(x_T_exact[:,1]):.4f})")
    print(f"  Learned var: ({np.var(x_T_learned[:,0]):.4f}, {np.var(x_T_learned[:,1]):.4f})")

ax.set_xlabel(r'$X_1$', fontsize=16)
ax.set_ylabel('Density', fontsize=16, rotation=0, ha='right')

# Right: Variance evolution over time for different mu
ax = axes[1]
x0 = np.array([2.0, 0.0])
mu_values_var = [0.5, 1.0, 2.0]
n_steps_var = 40  # Up to T=4.0
times = np.arange(0, n_steps_var + 1) * cfg.DT

for idx, mu in enumerate(mu_values_var):
    # Exact: compute variance analytically
    # For starting from deterministic x0, variance grows from 0
    # Var[X_t] = Σ_∞ * (1 - exp(-2μt)) where Σ_∞ = σ²/(2μ) is stationary variance
    sigma_infty = cfg.SIGMA**2 / (2 * mu)
    exact_var_t = sigma_infty * (1 - np.exp(-2 * mu * times))

    # Learned: simulate trajectories using exact simulation
    # (since we need full time series, use exact simulation)
    n_samples_var = 5000
    traj_exact_full = cfg.simulate_trajectory(x0, mu, cfg.DT, n_steps_var, n_traj=n_samples_var)
    # Shape: (n_samples, n_steps+1, 2)

    # For learned, use the multi-step approach but collect variances at each time
    learned_var_t = []
    for t_step in range(n_steps_var + 1):
        if t_step == 0:
            learned_var_t.append(0.0)  # Deterministic initial condition
        else:
            # Generate samples at this time by calling model t_step times
            x0_tiled = np.tile(x0.reshape(1, 2), (n_samples_var, 1))
            x_t = model.generate_trajectory(x0=x0_tiled, mu=mu, n_steps=t_step, n_samples=n_samples_var)
            learned_var_t.append(np.var(x_t[:, 0]))  # Variance of X1
    learned_var_t = np.array(learned_var_t)

    # Plot: Exact wider/transparent, Learned thinner/darker
    ax.plot(times, exact_var_t, color=colors[idx], linewidth=3, alpha=0.5,
            label=f'Exact ($\\mu={mu}$)')
    ax.plot(times, learned_var_t, color=colors[idx], linewidth=1.5)

ax.set_xlabel(r'Time $t$', fontsize=16)
# Remove y-axis for right panel (shared y-axis)
ax.set_yticklabels([])
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)
ax.set_xlim(0, n_steps_var * cfg.DT)
ax.legend(fontsize=14)  # Show legend in right panel

plt.tight_layout()
plt.savefig(os.path.join(cfg.FIG_DIR, 'fig4_multistep_variance.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {cfg.FIG_DIR}/fig4_multistep_variance.pdf")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Evaluation Complete!")
print("=" * 60)
print(f"\nGenerated figures in {cfg.FIG_DIR}/:")
print(f"  - fig1_trajectories.pdf")
print(f"  - fig2_conditional_scatter.pdf")
print(f"  - fig3_conditional_mean.pdf")
print(f"  - fig4_multistep_variance.pdf")
print(f"\nExact formulas for 2D OU with rotation:")
print(f"  A = [mu, -omega; omega, mu], omega = {cfg.OMEGA}")
print(f"  exp(-A*t) = exp(-mu*t) * R(-omega*t)")
print(f"  Conditional mean: E[X_{{n+1}}|X_n=x] = exp(-A*dt) * x")
print(f"  Conditional var: (sigma^2 / 2mu) * (1 - exp(-2*mu*dt))")
