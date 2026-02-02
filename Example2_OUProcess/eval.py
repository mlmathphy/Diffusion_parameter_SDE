"""
Example 2: Evaluation and Visualization (Stage 4)
==================================================

Load trained model and generate figures.

SDE: dX_t = -mu * X_t * dt + sqrt(1 + mu^2) * dW_t

Exact formulas:
- One-step conditional: p(X_{n+1}|X_n=x, mu) = N(x*exp(-mu*dt), (1+mu^2)/(2*mu)*(1-exp(-2*mu*dt)))
- Stationary distribution: N(0, (1+mu^2)/(2*mu))
- Terminal with Gaussian initial X_0 ~ N(m0, s0^2):
  X_T ~ N(m0*exp(-mu*T), s0^2*exp(-2*mu*T) + (1+mu^2)/(2*mu)*(1-exp(-2*mu*T)))

Figures:
1. Conditional distribution p(X_{n+1}|X_n, mu) and conditional mean
2. Heatmap of conditional distribution
3. Terminal distribution with Gaussian initial condition
4. Variance analysis (stationary and conditional)

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
plt.rcParams['font.size'] = 18  # Base font size for figures
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 12

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
print("Example 2: Evaluation and Visualization")
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
    mean = cfg.exact_conditional_mean(x, mu, dt)
    std = cfg.exact_conditional_std(mu, dt)
    return norm.pdf(y, loc=mean, scale=std)

def exact_terminal_mean_gaussian_init(m0, mu, T):
    """Exact E[X_T] when X_0 ~ N(m0, sigma0^2)"""
    return m0 * np.exp(-mu * T)

def exact_terminal_var_gaussian_init(sigma0, mu, T):
    """Exact Var[X_T] when X_0 ~ N(m0, sigma0^2)"""
    return sigma0**2 * np.exp(-2 * mu * T) + cfg.stationary_variance(mu) * (1 - np.exp(-2 * mu * T))

def exact_terminal_std_gaussian_init(sigma0, mu, T):
    """Exact Std[X_T] when X_0 ~ N(m0, sigma0^2)"""
    return np.sqrt(exact_terminal_var_gaussian_init(sigma0, mu, T))

def exact_terminal_pdf_gaussian_init(y, m0, sigma0, mu, T):
    """Exact terminal distribution when X_0 ~ N(m0, sigma0^2)"""
    mean = exact_terminal_mean_gaussian_init(m0, mu, T)
    std = exact_terminal_std_gaussian_init(sigma0, mu, T)
    return norm.pdf(y, loc=mean, scale=std)

# =============================================================================
# Figure 1: Conditional Distribution and Mean
# =============================================================================
print("\nGenerating Figure 1: Conditional distribution and mean...")

# Get niceplots colors
colors = niceplots.get_colors_list()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x_fixed = 1.0  # Non-zero to see mean-reversion effect
mu_values_plot = [0.5, 2.0]

# Left: Conditional distributions
ax = axes[0]
for idx, mu_test in enumerate(mu_values_plot):
    samples_learned = model.sample(x=x_fixed, mu=mu_test, n_samples=cfg.N_SAMPLES_PLOT).flatten()

    exact_mean = cfg.exact_conditional_mean(x_fixed, mu_test, cfg.DT)
    exact_std = cfg.exact_conditional_std(mu_test, cfg.DT)
    y_range = np.linspace(exact_mean - 4*exact_std, exact_mean + 4*exact_std, 300)

    pdf_exact = exact_conditional_pdf(y_range, x_fixed, mu_test, cfg.DT)
    # Exact: solid line
    ax.plot(y_range, pdf_exact, color=colors[idx], linewidth=LINEWIDTH,
            label=f'Exact ($\\mu={mu_test}$)')

    kde_learned = gaussian_kde(samples_learned)
    pdf_learned = kde_learned(y_range)
    # Learned: same color, markers
    subsample = 15
    ax.plot(y_range[::subsample], pdf_learned[::subsample], 'o', color=colors[idx],
            markersize=MARKERSIZE, label=f'Learned ($\\mu={mu_test}$)')

ax.set_xlabel(r'$X_{n+1}$')
ax.set_xlim([None,5])
ax.set_ylabel('Density', rotation=0, ha='right')
ax.legend(loc='upper right')

# Right: Conditional mean vs mu
ax = axes[1]
mu_test_range = np.linspace(cfg.MU_RANGE[0], cfg.MU_RANGE[1], 21)

means_exact = []
means_learned = []
vars_exact = []
vars_learned = []

for mu_test in mu_test_range:
    means_exact.append(cfg.exact_conditional_mean(x_fixed, mu_test, cfg.DT))
    vars_exact.append(cfg.exact_conditional_var(mu_test, cfg.DT))
    samples = model.sample(x=x_fixed, mu=mu_test, n_samples=cfg.N_SAMPLES_STAT).flatten()
    means_learned.append(np.mean(samples))
    vars_learned.append(np.var(samples))

means_exact = np.array(means_exact)
means_learned = np.array(means_learned)
vars_exact = np.array(vars_exact)
vars_learned = np.array(vars_learned)

# Exact: solid line
ax.plot(mu_test_range, means_exact, color=colors[0], linewidth=LINEWIDTH, label='Exact')
# Learned: same color, markers
ax.plot(mu_test_range, means_learned, 'o', color=colors[0], markersize=MARKERSIZE, label='Learned')
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\mathbb{E}[X_{n+1}]$', rotation=0, ha='right')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(cfg.FIG_DIR, 'fig1_conditional_combined.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cfg.FIG_DIR}/fig1_conditional_combined.pdf")

# Print error statistics for Figure 1
print("\n" + "=" * 60)
print(f"Figure 1 Error Statistics (X_n = {x_fixed})")
print("=" * 60)
# Relative error for mean (since values are not near zero)
mean_rel_errors = np.abs(means_learned - means_exact) / np.abs(means_exact) * 100
# Relative error for variance
var_rel_errors = np.abs(vars_learned - vars_exact) / vars_exact * 100

print(f"Conditional Mean (exact = x*exp(-mu*dt)):")
print(f"  Max relative error: {np.max(mean_rel_errors):.2f}%")
print(f"  Mean relative error: {np.mean(mean_rel_errors):.2f}%")
print(f"  Exact mean range: [{np.min(means_exact):.4f}, {np.max(means_exact):.4f}]")
print(f"  Learned mean range: [{np.min(means_learned):.4f}, {np.max(means_learned):.4f}]")
print(f"Conditional Variance:")
print(f"  Max relative error: {np.max(var_rel_errors):.2f}%")
print(f"  Mean relative error: {np.mean(var_rel_errors):.2f}%")
print(f"  Exact variance range: [{np.min(vars_exact):.6f}, {np.max(vars_exact):.6f}]")
print(f"  Learned variance range: [{np.min(vars_learned):.6f}, {np.max(vars_learned):.6f}]")

# =============================================================================
# Figure 2: Heatmap of Conditional Distribution
# =============================================================================
print("Generating Figure 2: Heatmap comparison...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=False)

x_fixed = 0.0  # Start from equilibrium to see variance effect
mu_grid = np.linspace(cfg.MU_RANGE[0], cfg.MU_RANGE[1], 50)
y_grid = np.linspace(-2.5, 2.5, 100)
MU, Y = np.meshgrid(mu_grid, y_grid)

# Exact heatmap
pdf_exact_grid = np.zeros_like(MU)
for i, mu_val in enumerate(mu_grid):
    pdf_exact_grid[:, i] = exact_conditional_pdf(y_grid, x_fixed, mu_val, cfg.DT)

# Get common color scale
vmin = min(pdf_exact_grid.min(), 0)
vmax = pdf_exact_grid.max()

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

ax = axes[0]
im = ax.contourf(MU, Y, pdf_exact_grid, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$X_{n+1}$', rotation=0, ha='right')
ax.set_title('Exact')

ax = axes[1]
im = ax.contourf(MU, Y, pdf_learned_grid, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
ax.set_xlabel(r'$\mu$')
ax.set_title('Learned')
# Remove y-axis for right panel (shared y-axis)
ax.set_yticklabels([])
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)

# Single shared colorbar at the bottom
fig.subplots_adjust(bottom=0.25)
cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Density')
cbar.ax.tick_params(labelsize=14)

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
mu_values = [0.5, 2.0]

# Gaussian initial distribution X_0 ~ N(1.0, 0.5^2) - start away from equilibrium
m0_gauss, sigma0_gauss = 1.0, 0.5

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

    # Exact (Monte Carlo using Euler-Maruyama)
    mc_dt = 0.001
    mc_steps = int(T_final / mc_dt)
    sigma = np.sqrt(1 + mu**2)
    x_T_exact = x0_samples.copy()
    for _ in range(mc_steps):
        dW = np.random.randn(cfg.N_SAMPLES_PLOT) * np.sqrt(mc_dt)
        x_T_exact = x_T_exact - mu * x_T_exact * mc_dt + sigma * dW

    # Plot range
    all_samples = np.concatenate([x_T_learned, x_T_exact])
    y_min, y_max = np.percentile(all_samples, [1, 99])
    y_range = np.linspace(y_min - 0.5, y_max + 0.5, 300)

    # MC ground truth: solid line
    kde_exact = gaussian_kde(x_T_exact)
    pdf_mc = kde_exact(y_range)
    ax.plot(y_range, pdf_mc, color=colors[0], linewidth=LINEWIDTH, label='MC')

    # Analytical: dashed line
    pdf_analytical = exact_terminal_pdf_gaussian_init(y_range, m0_gauss, sigma0_gauss, mu, T_final)
    ax.plot(y_range, pdf_analytical, '--', color=colors[1], linewidth=LINEWIDTH, label='Analytical')

    # Our Method: markers
    kde_learned = gaussian_kde(x_T_learned)
    pdf_learned = kde_learned(y_range)
    subsample = 15
    ax.plot(y_range[::subsample], pdf_learned[::subsample], 'o', color=colors[2],
            markersize=MARKERSIZE, label='Learned')

    # Add vertical line for mean
    analytical_mean = exact_terminal_mean_gaussian_init(m0_gauss, mu, T_final)
    ax.axvline(x=analytical_mean, color='gray', linestyle=':', linewidth=2, alpha=0.7)

    # Set y-axis limit with padding
    all_pdfs = np.concatenate([pdf_mc, pdf_analytical, pdf_learned])
    ax.set_ylim(0, np.max(all_pdfs) * 1.15)

    ax.set_xlabel(r'$X_T$')
    ax.set_title(f'$\\mu = {mu}$')
    if col == 0:
        ax.set_ylabel('Density', rotation=0, ha='right')
    else:
        ax.legend(loc='upper right')

    # Statistics
    analytical_std = exact_terminal_std_gaussian_init(sigma0_gauss, mu, T_final)
    print(f"\nGaussian initial, mu={mu}:")
    print(f"  MC mean: {np.mean(x_T_exact):.4f}, std: {np.std(x_T_exact):.4f}")
    print(f"  Analytical mean: {analytical_mean:.4f}, std: {analytical_std:.4f}")
    print(f"  Our Method mean: {np.mean(x_T_learned):.4f}, std: {np.std(x_T_learned):.4f}")

plt.tight_layout()
plt.savefig(os.path.join(cfg.FIG_DIR, 'fig3_terminal_distribution.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {cfg.FIG_DIR}/fig3_terminal_distribution.pdf")

# =============================================================================
# Figure 4: Variance Analysis
# =============================================================================
print("Generating Figure 4: Variance analysis...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Stationary variance as function of mu (non-monotonic!)
ax = axes[0]
mu_range_dense = np.linspace(cfg.MU_RANGE[0], cfg.MU_RANGE[1], 100)
stat_var_analytical = np.array([cfg.stationary_variance(mu) for mu in mu_range_dense])

# Analytical: solid line
ax.plot(mu_range_dense, stat_var_analytical, color=colors[0], linewidth=LINEWIDTH, label='Analytical')

# Compute learned stationary variance (from x=0 after many steps)
mu_test_vals = np.linspace(cfg.MU_RANGE[0], cfg.MU_RANGE[1], 21)
learned_stat_var = []
exact_stat_var = []
n_samples_stat_var = cfg.N_SAMPLES_PLOT
n_steps_stationary = 50  # Increase for better convergence, especially for small mu
for mu_test in mu_test_vals:
    # Start from x=0, take many steps to reach stationarity
    x0 = np.zeros(n_samples_stat_var)
    x_final = model.generate_trajectory(x0=x0, mu=mu_test, n_steps=n_steps_stationary, n_samples=n_samples_stat_var).flatten()
    learned_stat_var.append(np.var(x_final))
    exact_stat_var.append(cfg.stationary_variance(mu_test))

learned_stat_var = np.array(learned_stat_var)
exact_stat_var = np.array(exact_stat_var)

# Learned: markers
ax.plot(mu_test_vals, learned_stat_var, 'o', color=colors[0], markersize=MARKERSIZE, label=f'Learned ({n_steps_stationary} steps)')

# Add vertical line at mu=1 (minimum of stationary variance)
ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=2, alpha=0.7)

ax.set_xlabel(r'$\mu$')
ax.set_ylabel('Stationary Variance', rotation=90)
ax.set_ylim([0.8,1.4])
ax.set_title('Stationary Variance (min at $\\mu=1$)')
ax.legend(loc='upper right')

# Right: Conditional variance as function of mu
ax = axes[1]
cond_var_analytical = np.array([cfg.exact_conditional_var(mu, cfg.DT) for mu in mu_range_dense])

# Analytical: solid line
ax.plot(mu_range_dense, cond_var_analytical, color=colors[0], linewidth=LINEWIDTH, label='Analytical')

# Compute learned conditional variance
learned_cond_var = []
exact_cond_var = []
for mu_test in mu_test_vals:
    samples = model.sample(x=0.0, mu=mu_test, n_samples=cfg.N_SAMPLES_STAT).flatten()
    learned_cond_var.append(np.var(samples))
    exact_cond_var.append(cfg.exact_conditional_var(mu_test, cfg.DT))

learned_cond_var = np.array(learned_cond_var)
exact_cond_var = np.array(exact_cond_var)

# Learned: markers
ax.plot(mu_test_vals, learned_cond_var, 'o', color=colors[0], markersize=MARKERSIZE, label='Learned')
ax.set_xlabel(r'$\mu$')
ax.set_ylabel('Conditional Variance', rotation=90)
ax.set_title('One-Step Conditional Variance')
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(cfg.FIG_DIR, 'fig4_variance_analysis.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cfg.FIG_DIR}/fig4_variance_analysis.pdf")

# Print error statistics for Figure 4
print("\n" + "=" * 60)
print("Figure 4 Error Statistics (Variance Analysis)")
print("=" * 60)
stat_var_rel_errors = np.abs(learned_stat_var - exact_stat_var) / exact_stat_var * 100
cond_var_rel_errors = np.abs(learned_cond_var - exact_cond_var) / exact_cond_var * 100

print(f"Stationary Variance (after {n_steps_stationary} steps):")
print(f"  Max relative error: {np.max(stat_var_rel_errors):.2f}%")
print(f"  Mean relative error: {np.mean(stat_var_rel_errors):.2f}%")
print(f"  Exact range: [{np.min(exact_stat_var):.4f}, {np.max(exact_stat_var):.4f}]")
print(f"  Learned range: [{np.min(learned_stat_var):.4f}, {np.max(learned_stat_var):.4f}]")
print(f"One-Step Conditional Variance (from X_n=0):")
print(f"  Max relative error: {np.max(cond_var_rel_errors):.2f}%")
print(f"  Mean relative error: {np.mean(cond_var_rel_errors):.2f}%")
print(f"  Exact range: [{np.min(exact_cond_var):.6f}, {np.max(exact_cond_var):.6f}]")
print(f"  Learned range: [{np.min(learned_cond_var):.6f}, {np.max(learned_cond_var):.6f}]")

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
print(f"  - fig4_variance_analysis.pdf")
print(f"\nExact formulas for OU process dX = -mu*X*dt + sqrt(1+mu^2)*dW:")
print(f"  Conditional mean: E[X_{{n+1}}|X_n=x,mu] = x * exp(-mu*dt)")
print(f"  Conditional variance: Var[X_{{n+1}}|X_n=x,mu] = (1+mu^2)/(2*mu) * (1-exp(-2*mu*dt))")
print(f"  Stationary distribution: N(0, (1+mu^2)/(2*mu))")
print(f"  Note: Stationary variance is minimized at mu=1!")
