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
plt.rcParams['font.size'] = 18  # Base font size for figures
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

# Line and marker settings
LINEWIDTH = 2.5
MARKERSIZE = 8

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
print("Example 3: 2D OU with Rotation Evaluation")
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
n_traj_stats = 100  # Trajectories for statistics
n_traj_plot = 10    # Trajectories to actually plot

for idx, mu in enumerate(mu_values_traj):
    ax = axes[idx]

    # Store final points for statistics
    final_points_exact = []
    final_points_learned = []

    # Exact trajectories: generate all for statistics, plot only n_traj_plot
    for i in range(n_traj_stats):
        traj_exact = cfg.simulate_trajectory(x0, mu, cfg.DT, n_steps, n_traj=1)[0]
        if i < n_traj_plot:
            ax.plot(traj_exact[:, 0], traj_exact[:, 1], '-', color=colors[0],
                    alpha=0.5, linewidth=LINEWIDTH)
        final_points_exact.append(traj_exact[-1])

    # Learned trajectories: generate all for statistics, plot only n_traj_plot
    for i in range(n_traj_stats):
        traj_learned_full = np.zeros((n_steps + 1, 2))
        traj_learned_full[0] = x0
        x_current = x0.reshape(1, 2)
        for t in range(n_steps):
            x_next = model.sample(x=x_current, mu=mu, n_samples=1)
            traj_learned_full[t + 1] = x_next.flatten()
            x_current = x_next
        if i < n_traj_plot:
            ax.plot(traj_learned_full[:, 0], traj_learned_full[:, 1], '--',
                    color=colors[1], alpha=0.7, linewidth=LINEWIDTH*0.8)
        final_points_learned.append(traj_learned_full[-1])

    # Compute statistics for final positions
    final_points_exact = np.array(final_points_exact)
    final_points_learned = np.array(final_points_learned)

    # Expected final radius: |x0| * exp(-mu * T) where T = n_steps * dt
    T_final = n_steps * cfg.DT
    expected_radius = np.linalg.norm(x0) * np.exp(-mu * T_final)
    exact_radius = np.mean(np.linalg.norm(final_points_exact, axis=1))
    learned_radius = np.mean(np.linalg.norm(final_points_learned, axis=1))

    # Plot start and origin
    ax.plot(x0[0], x0[1], 'k*', markersize=15, markeredgewidth=1.5, zorder=10)
    ax.plot(0, 0, 'o', markersize=12, markerfacecolor='white',
            markeredgecolor='black', markeredgewidth=2, zorder=10)

    ax.set_xlabel(r'$X_1$')
    ax.set_title(f'$\\mu = {mu}$\n$\\bar{{r}}_{{exact}}={exact_radius:.3f}$, $\\bar{{r}}_{{learned}}={learned_radius:.3f}$')
    if idx == 0:
        ax.set_ylabel(r'$X_2$', rotation=0, ha='right')
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

# Single legend at top spanning all panels
handles = [
    plt.Line2D([0], [0], color=colors[0], linewidth=LINEWIDTH, label='Exact'),
    plt.Line2D([0], [0], color=colors[1], linewidth=LINEWIDTH*0.8, linestyle='--', label='Learned'),
    plt.Line2D([0], [0], marker='*', color='k', markersize=12, linestyle='None', label='Start'),
    plt.Line2D([0], [0], marker='o', color='white', markeredgecolor='black',
               markeredgewidth=2, markersize=10, linestyle='None', label='Origin'),
]
fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=14,
           bbox_to_anchor=(0.5, 1.02), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(cfg.FIG_DIR, 'fig1_trajectories.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cfg.FIG_DIR}/fig1_trajectories.pdf")

# Print decay rate statistics
print("\nFigure 1 Statistics (Final radius after T={:.1f}):".format(n_steps * cfg.DT))
print("=" * 60)
for idx, mu in enumerate(mu_values_traj):
    T_final = n_steps * cfg.DT
    expected_radius = np.linalg.norm(x0) * np.exp(-mu * T_final)
    print(f"  mu={mu}: Expected r = {expected_radius:.4f}")

# =============================================================================
# Figure 2: One-Step Conditional Distribution (2D Scatter)
# =============================================================================
print("\nGenerating Figure 2: One-step conditional distribution...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

x0 = np.array([1.5, 0.5])
mu_values_scatter = [0.5, 1.0, 2.0]
n_samples_scatter = 5000

print("\nFigure 2 Statistics (One-step conditional from x0 = (1.5, 0.5)):")
print("=" * 60)

for idx, mu in enumerate(mu_values_scatter):
    # Compute exact statistics
    mean_exact = cfg.exact_conditional_mean(x0, mu, cfg.DT).flatten()
    cov_exact = cfg.exact_conditional_cov(mu, cfg.DT)
    var_exact = cov_exact[0, 0]  # Same for both dimensions

    # Generate samples
    samples_exact = cfg.sample_exact_transition(x0, mu, cfg.DT, n_samples_scatter)
    samples_learned = model.sample(x=x0, mu=mu, n_samples=n_samples_scatter)

    # Compute learned statistics
    mean_learned = np.mean(samples_learned, axis=0)
    var_learned = np.mean(np.var(samples_learned, axis=0))

    # Mean error
    mean_error = np.linalg.norm(mean_learned - mean_exact)
    var_error = np.abs(var_learned - var_exact) / var_exact * 100

    print(f"  mu={mu}: Mean error = {mean_error:.4f}, Var rel error = {var_error:.1f}%")

    # Determine good axis limits based on samples
    all_samples = np.vstack([samples_exact, samples_learned])
    x_margin = 0.3
    y_margin = 0.3
    x_min, x_max = all_samples[:, 0].min() - x_margin, all_samples[:, 0].max() + x_margin
    y_min, y_max = all_samples[:, 1].min() - y_margin, all_samples[:, 1].max() + y_margin

    # Top row: Exact samples
    ax = axes[0, idx]
    ax.scatter(samples_exact[:, 0], samples_exact[:, 1], s=2, alpha=0.3, c=colors[0])
    ax.plot(x0[0], x0[1], 'ko', markersize=MARKERSIZE, zorder=10)
    ax.plot(mean_exact[0], mean_exact[1], 'k*', markersize=15, zorder=10)

    ax.set_title(f'Exact ($\\mu={mu}$)\n$\\bar{{x}}$=({mean_exact[0]:.2f},{mean_exact[1]:.2f}), $\\sigma^2$={var_exact:.4f}')
    if idx == 0:
        ax.set_ylabel(r'$X_2$', rotation=0, ha='right')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Bottom row: Learned samples
    ax = axes[1, idx]
    ax.scatter(samples_learned[:, 0], samples_learned[:, 1], s=2, alpha=0.3, c=colors[1])
    ax.plot(x0[0], x0[1], 'ko', markersize=MARKERSIZE, zorder=10)
    ax.plot(mean_learned[0], mean_learned[1], 'k*', markersize=15, zorder=10)

    ax.set_title(f'Learned ($\\mu={mu}$)\n$\\bar{{x}}$=({mean_learned[0]:.2f},{mean_learned[1]:.2f}), $\\sigma^2$={var_learned:.4f}')
    ax.set_xlabel(r'$X_1$')
    if idx == 0:
        ax.set_ylabel(r'$X_2$', rotation=0, ha='right')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

# Single legend at top
handles = [
    plt.Line2D([0], [0], marker='o', color='k', markersize=10, linestyle='None', label='Start $x_0$'),
    plt.Line2D([0], [0], marker='*', color='k', markersize=15, linestyle='None', label='Mean'),
    plt.Line2D([0], [0], marker='o', color=colors[0], markersize=8, linestyle='None', label='Exact samples'),
    plt.Line2D([0], [0], marker='o', color=colors[1], markersize=8, linestyle='None', label='Learned samples'),
]
fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=14,
           bbox_to_anchor=(0.5, 1.02), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(cfg.FIG_DIR, 'fig2_conditional_scatter.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cfg.FIG_DIR}/fig2_conditional_scatter.pdf")

# =============================================================================
# Figure 3: Conditional Mean Comparison
# =============================================================================
print("\nGenerating Figure 3: Conditional mean comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Test multiple starting points and mu values
x0_values = [np.array([2.0, 0.0]), np.array([0.0, 2.0]), np.array([1.5, 1.5])]
x0_labels = ['(2,0)', '(0,2)', '(1.5,1.5)']
mu_test_range = np.linspace(cfg.MU_RANGE[0], cfg.MU_RANGE[1], 15)

print("\nFigure 3 Statistics (Conditional mean errors):")
print("=" * 60)

# Left: X1 component of conditional mean
ax = axes[0]
for x0_idx, x0 in enumerate(x0_values):
    means_exact_x1 = np.array([cfg.exact_conditional_mean(x0, mu, cfg.DT).flatten()[0] for mu in mu_test_range])
    means_learned_x1 = []
    for mu in mu_test_range:
        samples = model.sample(x=x0, mu=mu, n_samples=cfg.N_SAMPLES_STAT)
        means_learned_x1.append(np.mean(samples[:, 0]))
    means_learned_x1 = np.array(means_learned_x1)

    # Exact: solid line
    ax.plot(mu_test_range, means_exact_x1, color=colors[x0_idx], linewidth=LINEWIDTH,
            label=f'Exact $x_0$={x0_labels[x0_idx]}')
    # Learned: markers (same color)
    ax.plot(mu_test_range, means_learned_x1, 'o', color=colors[x0_idx], markersize=MARKERSIZE,
            label=f'Learned $x_0$={x0_labels[x0_idx]}')

    # Print error
    max_err = np.max(np.abs(means_learned_x1 - means_exact_x1))
    print(f"  x0={x0_labels[x0_idx]}, X1 component: max abs error = {max_err:.4f}")

ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\mathbb{E}[X_1^{n+1}]$', rotation=90)
ax.set_title(r'$X_1$ Component')
ax.grid(True, alpha=0.3)

# Right: X2 component of conditional mean
ax = axes[1]
for x0_idx, x0 in enumerate(x0_values):
    means_exact_x2 = np.array([cfg.exact_conditional_mean(x0, mu, cfg.DT).flatten()[1] for mu in mu_test_range])
    means_learned_x2 = []
    for mu in mu_test_range:
        samples = model.sample(x=x0, mu=mu, n_samples=cfg.N_SAMPLES_STAT)
        means_learned_x2.append(np.mean(samples[:, 1]))
    means_learned_x2 = np.array(means_learned_x2)

    # Exact: solid line
    ax.plot(mu_test_range, means_exact_x2, color=colors[x0_idx], linewidth=LINEWIDTH,
            label=f'Exact $x_0$={x0_labels[x0_idx]}')
    # Learned: markers (same color)
    ax.plot(mu_test_range, means_learned_x2, 'o', color=colors[x0_idx], markersize=MARKERSIZE,
            label=f'Learned $x_0$={x0_labels[x0_idx]}')

    # Print error
    max_err = np.max(np.abs(means_learned_x2 - means_exact_x2))
    print(f"  x0={x0_labels[x0_idx]}, X2 component: max abs error = {max_err:.4f}")

ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\mathbb{E}[X_2^{n+1}]$', rotation=90)
ax.set_title(r'$X_2$ Component')
ax.grid(True, alpha=0.3)

# Single legend at top
handles = [
    plt.Line2D([0], [0], color='gray', linewidth=LINEWIDTH, label='Exact (solid)'),
    plt.Line2D([0], [0], marker='o', color='gray', markersize=MARKERSIZE, linestyle='None', label='Learned (markers)'),
]
fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=14,
           bbox_to_anchor=(0.5, 1.02), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(cfg.FIG_DIR, 'fig3_conditional_mean.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cfg.FIG_DIR}/fig3_conditional_mean.pdf")

# =============================================================================
# Figure 4: Multi-step Distribution and Variance
# =============================================================================
print("\nGenerating Figure 4: Multi-step distribution and variance...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Multi-step distribution at T=2.0 (starting from fixed x0)
ax = axes[0]
x0 = np.array([2.0, 0.0])
mu_values_terminal = [0.5, 1.0, 2.0]
n_steps_term = 20
n_samples_terminal = 10000

print("\nFigure 4 Left: Terminal distribution at T={:.1f}".format(n_steps_term * cfg.DT))
print("=" * 60)

for idx, mu in enumerate(mu_values_terminal):
    # Exact: simulate trajectories
    traj_exact = cfg.simulate_trajectory(x0, mu, cfg.DT, n_steps_term, n_traj=n_samples_terminal)
    x_T_exact = traj_exact[:, -1, :]  # Shape (n_samples, 2)

    # Learned: multi-step
    x0_tiled = np.tile(x0.reshape(1, 2), (n_samples_terminal, 1))
    x_T_learned = model.generate_trajectory(x0=x0_tiled, mu=mu, n_steps=n_steps_term, n_samples=n_samples_terminal)

    # Plot marginal X1 distribution
    kde_exact = gaussian_kde(x_T_exact[:, 0])
    kde_learned = gaussian_kde(x_T_learned[:, 0])
    x_grid = np.linspace(-2, 2.5, 200)

    # Exact: solid line
    ax.plot(x_grid, kde_exact(x_grid), color=colors[idx], linewidth=LINEWIDTH,
            label=f'Exact ($\\mu={mu}$)')
    # Learned: markers (same color, subsampled)
    subsample = 10
    ax.plot(x_grid[::subsample], kde_learned(x_grid)[::subsample], 'o', color=colors[idx],
            markersize=MARKERSIZE*0.7, label=f'Learned ($\\mu={mu}$)')

    # Print statistics
    mean_err = np.abs(np.mean(x_T_learned[:, 0]) - np.mean(x_T_exact[:, 0]))
    var_err = np.abs(np.var(x_T_learned[:, 0]) - np.var(x_T_exact[:, 0])) / np.var(x_T_exact[:, 0]) * 100
    print(f"  mu={mu}: Mean error = {mean_err:.4f}, Var rel error = {var_err:.1f}%")

ax.set_xlabel(r'$X_1$')
ax.set_ylabel('Density', rotation=90)
ax.set_title(f'Terminal Distribution at $T={n_steps_term * cfg.DT}$')
ax.grid(True, alpha=0.3)

# Right: Variance evolution over time for different mu
ax = axes[1]
x0 = np.array([2.0, 0.0])
mu_values_var = [0.5, 1.0, 2.0]
n_steps_var = 30  # Up to T=3.0
times = np.arange(0, n_steps_var + 1) * cfg.DT
n_samples_var = 5000

print("\nFigure 4 Right: Variance evolution")
print("=" * 60)

for idx, mu in enumerate(mu_values_var):
    # Exact: compute variance analytically
    sigma_infty = cfg.SIGMA**2 / (2 * mu)
    exact_var_t = sigma_infty * (1 - np.exp(-2 * mu * times))

    # Learned: collect variances at each time step
    learned_var_t = [0.0]  # t=0: deterministic
    for t_step in range(1, n_steps_var + 1):
        x0_tiled = np.tile(x0.reshape(1, 2), (n_samples_var, 1))
        x_t = model.generate_trajectory(x0=x0_tiled, mu=mu, n_steps=t_step, n_samples=n_samples_var)
        learned_var_t.append(np.var(x_t[:, 0]))
    learned_var_t = np.array(learned_var_t)

    # Exact: solid line
    ax.plot(times, exact_var_t, color=colors[idx], linewidth=LINEWIDTH,
            label=f'Exact ($\\mu={mu}$)')
    # Learned: markers
    ax.plot(times[::2], learned_var_t[::2], 'o', color=colors[idx],
            markersize=MARKERSIZE*0.7, label=f'Learned ($\\mu={mu}$)')

    # Print final variance error
    var_err_final = np.abs(learned_var_t[-1] - exact_var_t[-1]) / exact_var_t[-1] * 100
    print(f"  mu={mu}: Stationary var = {sigma_infty:.4f}, Final var error = {var_err_final:.1f}%")

ax.set_xlabel(r'Time $t$')
ax.set_ylabel('Variance', rotation=90)
ax.set_title('Variance Evolution')
ax.set_xlim(0, n_steps_var * cfg.DT)
ax.grid(True, alpha=0.3)

# Add horizontal lines for stationary variances
for idx, mu in enumerate(mu_values_var):
    sigma_infty = cfg.SIGMA**2 / (2 * mu)
    ax.axhline(y=sigma_infty, color=colors[idx], linestyle=':', alpha=0.5)

# Single legend at top
handles = [
    plt.Line2D([0], [0], color='gray', linewidth=LINEWIDTH, label='Exact (solid)'),
    plt.Line2D([0], [0], marker='o', color='gray', markersize=MARKERSIZE*0.7, linestyle='None', label='Learned (markers)'),
]
fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=14,
           bbox_to_anchor=(0.5, 1.02), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
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
print(f"  exp(-A*t) = exp(-mu*t) * R(omega*t)  (clockwise rotation)")
print(f"  Conditional mean: E[X_{{n+1}}|X_n=x] = exp(-A*dt) * x")
print(f"  Conditional var: (sigma^2 / 2mu) * (1 - exp(-2*mu*dt))")
