"""
Example 1: Evaluation and Visualization (Stage 4)
==================================================

Load trained model and generate figures.

Figures:
1. Conditional distribution p(X_{n+1}|X_n, mu) and conditional mean
2. Heatmap of conditional distribution
3. Terminal distribution with Gaussian initial condition

Usage:
    python eval.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
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
plt.rcParams['text.usetex'] = False  # Set True if LaTeX is available
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
print("Example 1: Evaluation and Visualization")
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
# Exact Distribution Functions
# =============================================================================
def exact_conditional_pdf(y, x, mu, dt):
    """Exact p(y|x, mu) for one-step transition"""
    mean = x + mu * dt
    std = np.sqrt(dt)
    return norm.pdf(y, loc=mean, scale=std)

def exact_conditional_mean(x, mu, dt):
    """Exact E[X_{n+1} | X_n = x, mu]"""
    return x + mu * dt

def exact_conditional_std(dt):
    """Exact Std[X_{n+1} | X_n = x, mu]"""
    return np.sqrt(dt)

def exact_terminal_pdf_gaussian_init(y, m0, sigma0, mu, T):
    """Exact terminal distribution when X_0 ~ N(m0, sigma0^2)"""
    mean = m0 + mu * T
    std = np.sqrt(sigma0**2 + T)
    return norm.pdf(y, loc=mean, scale=std)

# =============================================================================
# Figure 1: Conditional Distribution and Mean
# =============================================================================
print("\nGenerating Figure 1: Conditional distribution and mean...")

# Get niceplots colors
colors = niceplots.get_colors_list()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x_fixed = 2.0
mu_values_plot = [-0.5, 0.5]

# Left: Conditional distributions
ax = axes[0]
for idx, mu_test in enumerate(mu_values_plot):
    samples_learned = model.sample(x=x_fixed, mu=mu_test, n_samples=cfg.N_SAMPLES_PLOT).flatten()

    exact_mean = exact_conditional_mean(x_fixed, mu_test, cfg.DT)
    exact_std = exact_conditional_std(cfg.DT)
    y_range = np.linspace(exact_mean - 4*exact_std, exact_mean + 4*exact_std, 300)

    pdf_exact = exact_conditional_pdf(y_range, x_fixed, mu_test, cfg.DT)
    # Exact: wider, more transparent
    ax.plot(y_range, pdf_exact, color=colors[idx], linewidth=3, alpha=0.5,
            label=f'Exact ($\\mu={mu_test}$)')

    kde_learned = gaussian_kde(samples_learned)
    pdf_learned = kde_learned(y_range)
    # Learned: thinner, solid
    ax.plot(y_range, pdf_learned, color=colors[idx], linewidth=1.5,
            label=f'Learned ($\\mu={mu_test}$)')

ax.set_xlabel(r'$X_{n+1}$', fontsize=16)
ax.set_ylabel('Density', fontsize=16, rotation=0, ha='right')
ax.legend(fontsize=14)

# Right: Conditional mean vs mu
ax = axes[1]
mu_test_range = np.linspace(cfg.MU_RANGE[0], cfg.MU_RANGE[1], 21)

means_exact = []
means_learned = []

for mu_test in mu_test_range:
    means_exact.append(exact_conditional_mean(x_fixed, mu_test, cfg.DT))
    samples = model.sample(x=x_fixed, mu=mu_test, n_samples=cfg.N_SAMPLES_STAT).flatten()
    means_learned.append(np.mean(samples))

# Exact: wider, more transparent
ax.plot(mu_test_range, means_exact, color=colors[0], linewidth=3, alpha=0.5, label='Exact')
# Learned: markers
ax.plot(mu_test_range, means_learned, 'o', color=colors[1], markersize=5, label='Learned')
ax.set_xlabel(r'$\mu$', fontsize=16)
ax.set_ylabel(r'$\mathbb{E}[X_{n+1}]$', fontsize=16, rotation=0, ha='right')
ax.legend(fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(cfg.FIG_DIR, 'fig1_conditional_combined.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cfg.FIG_DIR}/fig1_conditional_combined.pdf")

# =============================================================================
# Figure 2: Heatmap of Conditional Distribution
# =============================================================================
print("Generating Figure 2: Heatmap comparison...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=False)

x_fixed = 0.0
mu_grid = np.linspace(cfg.MU_RANGE[0], cfg.MU_RANGE[1], 50)
y_grid = np.linspace(-1.5, 1.5, 100)
MU, Y = np.meshgrid(mu_grid, y_grid)

# Exact heatmap
pdf_exact_grid = exact_conditional_pdf(Y, x_fixed, MU, cfg.DT)

# Get common color scale
vmin = min(pdf_exact_grid.min(), 0)
vmax = max(pdf_exact_grid.max(), 1)

ax = axes[0]
im = ax.contourf(MU, Y, pdf_exact_grid, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
ax.set_xlabel(r'$\mu$', fontsize=16)
ax.set_ylabel(r'$X_{n+1}$', fontsize=16, rotation=0, ha='right')

# Learned heatmap
n_samples_heatmap = 5000
pdf_learned_grid = np.zeros_like(pdf_exact_grid)

for i, mu_val in enumerate(mu_grid):
    samples = model.sample(x=x_fixed, mu=mu_val, n_samples=n_samples_heatmap).flatten()
    if len(samples) > 1 and np.std(samples) > 1e-6:
        kde = gaussian_kde(samples)
        pdf_learned_grid[:, i] = kde(y_grid)

# Update vmax based on learned data
vmax = max(vmax, pdf_learned_grid.max())

# Re-plot exact with updated scale
axes[0].clear()
im = axes[0].contourf(MU, Y, pdf_exact_grid, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_xlabel(r'$\mu$', fontsize=16)
axes[0].set_ylabel(r'$X_{n+1}$', fontsize=16, rotation=0, ha='right')

ax = axes[1]
im = ax.contourf(MU, Y, pdf_learned_grid, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
ax.set_xlabel(r'$\mu$', fontsize=16)
# Remove y-axis for right panel (shared y-axis)
ax.set_yticklabels([])
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)

# Single shared colorbar at the bottom
fig.subplots_adjust(bottom=0.25)
cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Density', fontsize=14)

plt.savefig(os.path.join(cfg.FIG_DIR, 'fig2_heatmap.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cfg.FIG_DIR}/fig2_heatmap.pdf")

# =============================================================================
# Figure 3: Terminal Distribution
# =============================================================================
print("Generating Figure 3: Terminal distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

T_final = cfg.T
n_steps = int(T_final / cfg.DT)
mu_values = [-0.5, 0.5]

# Gaussian initial distribution X_0 ~ N(0, 0.5^2)
m0_gauss, sigma0_gauss = 0.0, 0.5

for col, mu in enumerate(mu_values):
    ax = axes[col]

    x0_samples = np.random.normal(m0_gauss, sigma0_gauss, cfg.N_SAMPLES_PLOT)

    # Learned model
    x_T_learned = model.generate_trajectory(
        x0=x0_samples.copy(),
        mu=mu,
        n_steps=n_steps,
        n_samples=cfg.N_SAMPLES_PLOT
    ).flatten()

    # Exact (Monte Carlo)
    W_T = np.random.randn(cfg.N_SAMPLES_PLOT) * np.sqrt(T_final)
    x_T_exact = x0_samples + mu * T_final + W_T

    # Plot range
    all_samples = np.concatenate([x_T_learned, x_T_exact])
    y_min, y_max = np.percentile(all_samples, [1, 99])
    y_range = np.linspace(y_min - 0.5, y_max + 0.5, 300)

    # MC ground truth: wider, more transparent
    kde_exact = gaussian_kde(x_T_exact)
    pdf_mc = kde_exact(y_range)
    ax.plot(y_range, pdf_mc, color=colors[0], linewidth=3, alpha=0.5, label='MC')

    # Analytical: wider, more transparent
    pdf_analytical = exact_terminal_pdf_gaussian_init(y_range, m0_gauss, sigma0_gauss, mu, T_final)
    ax.plot(y_range, pdf_analytical, color=colors[1], linewidth=3, alpha=0.5, label='Analytical')

    # Our Method: thinner, solid
    kde_learned = gaussian_kde(x_T_learned)
    pdf_learned = kde_learned(y_range)
    ax.plot(y_range, pdf_learned, color=colors[2], linewidth=1.5, label='Our Method')

    ax.set_xlabel(r'$X_T$', fontsize=16)
    if col == 0:
        ax.set_ylabel('Density', fontsize=16, rotation=0, ha='right')
    else:
        # Remove y-axis for right panel (shared y-axis)
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        ax.legend(fontsize=14)  # Show legend in right panel

    # Statistics
    print(f"\nGaussian initial, mu={mu}:")
    print(f"  MC mean: {np.mean(x_T_exact):.4f}, std: {np.std(x_T_exact):.4f}")
    print(f"  Analytical mean: {m0_gauss + mu * T_final:.4f}, std: {np.sqrt(sigma0_gauss**2 + T_final):.4f}")
    print(f"  Our Method mean: {np.mean(x_T_learned):.4f}, std: {np.std(x_T_learned):.4f}")

plt.tight_layout()
plt.savefig(os.path.join(cfg.FIG_DIR, 'fig3_terminal_distribution.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {cfg.FIG_DIR}/fig3_terminal_distribution.pdf")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Evaluation Complete!")
print("=" * 60)
print(f"\nGenerated figures in {cfg.FIG_DIR}/:")
print(f"  - fig1_conditional_combined.pdf")
print(f"  - fig2_heatmap.pdf")
print(f"  - fig3_terminal_distribution.pdf")
