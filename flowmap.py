"""
Training-Free Conditional Diffusion Model for Parameter-Dependent Stochastic Flow Maps

This module implements the training-free conditional diffusion model framework for learning
stochastic flow maps of parameter-dependent stochastic differential equations (SDEs).

Reference:
    Yang, M., & He, S. (2026). A training-free conditional diffusion model for learning
    parameter-dependent stochastic flow maps. AIMS Mathematics.

Main Components:
    1. TrainingDataGenerator: Generate SDE trajectory data for different parameter values
    2. LabelGenerator: Generate labeled data (x, mu, z, y) using training-free diffusion
    3. FlowMapNet: Neural network for learning the parameter-dependent flow map
    4. FlowMapModel: Main class that orchestrates the entire workflow

Example Usage:
    >>> from flowmap import FlowMapModel
    >>>
    >>> # Define your SDE coefficients
    >>> def drift(x, mu):
    ...     return mu  # Simple Brownian motion with drift
    >>> def diffusion(x, mu):
    ...     return 1.0
    >>>
    >>> # Create and train the model
    >>> model = FlowMapModel(
    ...     drift_fn=drift,
    ...     diffusion_fn=diffusion,
    ...     x_dim=1,
    ...     mu_dim=1,
    ...     x_range=(-5, 5),
    ...     mu_range=(-1, 1)
    ... )
    >>> model.generate_training_data(n_mu=21, n_traj=5000)
    >>> model.generate_labels(n_labels=50000)
    >>> model.train_network(n_epochs=5000)
    >>>
    >>> # Generate samples
    >>> samples = model.sample(x=2.0, mu=0.5, n_samples=1000)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Callable, Optional, Tuple, Dict, Any, Union
from tqdm import tqdm
from scipy.spatial import cKDTree
import os


class FlowMapNet(nn.Module):
    """
    Neural network for learning the parameter-dependent stochastic flow map.

    The network learns G_theta(x, mu, z) such that:
        X_{n+1} = X_n + G_theta(X_n, mu, Z) / diff_scale
    where Z ~ N(0, I).

    Args:
        input_dim: Dimension of input (x_dim + mu_dim + z_dim)
        output_dim: Dimension of output (x_dim)
        hidden_size: Number of neurons in hidden layers
        n_hidden_layers: Number of hidden layers
        activation: Activation function ('tanh', 'relu', 'gelu')
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        n_hidden_layers: int = 3,
        activation: str = 'tanh'
    ):
        super(FlowMapNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size

        # Select activation function
        if activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = [nn.Linear(input_dim, hidden_size), act_fn()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), act_fn()])
        layers.append(nn.Linear(hidden_size, output_dim))

        self.net = nn.Sequential(*layers)

        # Store best weights for early stopping
        self.best_weights = None
        self.best_val_loss = float('inf')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def save_best(self):
        """Save current weights as best weights."""
        self.best_weights = {k: v.clone() for k, v in self.state_dict().items()}

    def load_best(self):
        """Load best weights."""
        if self.best_weights is not None:
            self.load_state_dict(self.best_weights)


class DiffusionSchedule:
    """
    Diffusion schedule functions for the forward/reverse diffusion process.

    Uses the variance-preserving (VP) schedule:
        alpha_t = 1 - t
        beta_t^2 = t

    With small regularization dt for numerical stability.
    """

    @staticmethod
    def alpha(t: float, dt: float) -> float:
        """Scaling function: alpha_t(0) = 1, alpha_t(1) = 0"""
        return 1 - t + dt

    @staticmethod
    def sigma2(t: float, dt: float) -> float:
        """Variance function: sigma2_t(0) = 0, sigma2_t(1) = 1"""
        return t + dt

    @staticmethod
    def f(t: float, dt: float) -> float:
        """Drift coefficient: f(t) = d(log alpha_t)/dt"""
        alpha_t = DiffusionSchedule.alpha(t, dt)
        return -1.0 / alpha_t

    @staticmethod
    def g2(t: float, dt: float) -> float:
        """Diffusion coefficient squared: g^2 = d(sigma_t^2)/dt - 2*f*sigma_t^2"""
        dsigma2_dt = 1.0
        f_t = DiffusionSchedule.f(t, dt)
        sigma2_t = DiffusionSchedule.sigma2(t, dt)
        return dsigma2_dt - 2 * f_t * sigma2_t


class TrainingDataGenerator:
    """
    Generate training data by simulating parameter-dependent SDEs.

    Simulates the SDE:
        dX_t = a(X_t, mu) dt + b(X_t, mu) dW_t

    using the Euler-Maruyama scheme with fine time steps, and records
    consecutive state pairs (X_n, X_{n+1}) at coarser time steps.

    Args:
        drift_fn: Drift coefficient function a(x, mu)
        diffusion_fn: Diffusion coefficient function b(x, mu)
        x_dim: State dimension
        mu_dim: Parameter dimension
        x_range: Tuple (x_min, x_max) for initial condition sampling
        mu_range: Tuple (mu_min, mu_max) for parameter domain
        dt: Time step for recording (Delta t)
        T: Total simulation time
        fine_dt: Fine time step for Euler-Maruyama integration
    """

    def __init__(
        self,
        drift_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        diffusion_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        x_dim: int = 1,
        mu_dim: int = 1,
        x_range: Tuple[float, float] = (-5.0, 5.0),
        mu_range: Tuple[float, float] = (-1.0, 1.0),
        dt: float = 0.1,
        T: float = 1.0,
        fine_dt: float = 0.001,
        init_sampler: Optional[Callable] = None
    ):
        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.x_dim = x_dim
        self.mu_dim = mu_dim
        self.x_range = x_range
        self.mu_range = mu_range
        self.dt = dt
        self.T = T
        self.fine_dt = fine_dt
        self.init_sampler = init_sampler

    def simulate(
        self,
        mu: float,
        n_traj: int,
        x_init: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate SDE trajectories for a fixed parameter value.

        Args:
            mu: Parameter value
            n_traj: Number of trajectories
            x_init: Initial conditions (n_traj, x_dim). If None, sample uniformly.

        Returns:
            x_sample: Starting positions (n_pairs, x_dim)
            y_sample: Ending positions (n_pairs, x_dim)
            mu_sample: Parameter values (n_pairs, mu_dim)
        """
        t_needed = int(self.dt / self.fine_dt)
        n_steps = int(np.floor(self.T / self.fine_dt))
        n_snap = int(np.floor(self.T / self.dt) + 1)

        # Initialize
        if x_init is None:
            if self.init_sampler is not None:
                x_current = self.init_sampler(mu, n_traj, self.x_dim)
            else:
                x_current = np.random.uniform(
                    self.x_range[0], self.x_range[1],
                    (n_traj, self.x_dim)
                )
        else:
            x_current = x_init.copy()

        # Storage for snapshots
        trajectories = np.zeros((n_traj, n_snap, self.x_dim))
        trajectories[:, 0, :] = x_current

        snap_idx = 1

        # Euler-Maruyama time stepping
        for step in range(n_steps):
            drift = self.drift_fn(x_current, mu)
            diff = self.diffusion_fn(x_current, mu)
            dW = np.random.normal(0, np.sqrt(self.fine_dt), (n_traj, self.x_dim))
            x_current = x_current + drift * self.fine_dt + diff * dW

            # Store snapshot
            if (step + 1) % t_needed == 0 and snap_idx < n_snap:
                trajectories[:, snap_idx, :] = x_current
                snap_idx += 1

        # Extract consecutive pairs
        x_list = []
        y_list = []

        for t_idx in range(n_snap - 1):
            x_list.append(trajectories[:, t_idx, :])
            y_list.append(trajectories[:, t_idx + 1, :])

        x_sample = np.vstack(x_list)
        y_sample = np.vstack(y_list)
        mu_sample = mu * np.ones((x_sample.shape[0], self.mu_dim))

        return x_sample, y_sample, mu_sample

    def generate_dataset(
        self,
        n_mu: int = 21,
        n_traj: int = 5000,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate full training dataset across parameter range.

        Args:
            n_mu: Number of parameter values to sample
            n_traj: Number of trajectories per parameter value
            seed: Random seed

        Returns:
            Dictionary with keys 'x_data', 'y_data', 'mu_data', 'mu_values'
        """
        if seed is not None:
            np.random.seed(seed)

        mu_values = np.linspace(self.mu_range[0], self.mu_range[1], n_mu)

        all_x = []
        all_y = []
        all_mu = []

        for mu in tqdm(mu_values, desc="Generating training data"):
            x_sample, y_sample, mu_sample = self.simulate(mu, n_traj)
            all_x.append(x_sample)
            all_y.append(y_sample)
            all_mu.append(mu_sample)

        return {
            'x_data': np.vstack(all_x),
            'y_data': np.vstack(all_y),
            'mu_data': np.vstack(all_mu),
            'mu_values': mu_values,
            'dt': self.dt
        }


class LabelGenerator:
    """
    Generate labeled training data using the training-free diffusion model.

    For each query (x, mu), this class:
    1. Finds nearest neighbors in the training data
    2. Uses closed-form score estimation via Monte Carlo
    3. Solves the reverse probability flow ODE to map z ~ N(0,1) to y ~ p(y|x,mu)

    Args:
        device: PyTorch device ('cuda' or 'cpu')
        n_neighbors: Number of nearest neighbors for score estimation
        ode_steps: Number of ODE discretization steps
        nu_x: Bandwidth for spatial (x) Gaussian kernel
        nu_mu: Bandwidth for parameter (mu) Gaussian kernel
        diff_scale: Scaling factor for displacements
    """

    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        n_neighbors: int = 2000,
        ode_steps: int = 10000,
        nu_x: float = 1.0,
        nu_mu: float = 0.5,
        diff_scale: float = 3.0
    ):
        self.device = torch.device(device)
        self.n_neighbors = n_neighbors
        self.ode_steps = ode_steps
        self.nu_x = nu_x
        self.nu_mu = nu_mu
        self.diff_scale = diff_scale
        self.schedule = DiffusionSchedule()

    def _find_neighbors(
        self,
        x_query: np.ndarray,
        mu_query: np.ndarray,
        x_data: np.ndarray,
        mu_data: np.ndarray,
        z_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find k nearest neighbors for each query point using KDTree."""
        n_query = x_query.shape[0]
        k = self.n_neighbors

        # Normalize for distance computation
        x_std = np.std(x_data) + 1e-8
        mu_std = np.std(mu_data) + 1e-8

        data_combined = np.hstack([x_data / x_std, mu_data / mu_std])
        query_combined = np.hstack([x_query / x_std, mu_query / mu_std])

        # Build KDTree (much faster than brute force)
        print("Building KDTree...")
        tree = cKDTree(data_combined)

        # Query in batches with progress bar (avoid workers=-1 on macOS)
        batch_size = 5000
        n_batches = (n_query + batch_size - 1) // batch_size
        all_indices = []

        for i in tqdm(range(n_batches), desc="KDTree query"):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_query)
            _, batch_indices = tree.query(query_combined[start:end], k=k)
            all_indices.append(batch_indices)

        indices = np.vstack(all_indices)

        # Gather neighbor data
        x_neighbors = x_data[indices]
        mu_neighbors = mu_data[indices]
        z_neighbors = z_data[indices]

        return x_neighbors, mu_neighbors, z_neighbors

    def _solve_reverse_ode(
        self,
        zt: torch.Tensor,
        x_neighbors: torch.Tensor,
        mu_neighbors: torch.Tensor,
        z_neighbors: torch.Tensor,
        x_query: torch.Tensor,
        mu_query: torch.Tensor
    ) -> torch.Tensor:
        """Solve the reverse probability flow ODE."""
        t_vec = torch.linspace(1.0, 0.0, self.ode_steps + 1)

        for j in range(self.ode_steps):
            t = t_vec[j + 1].item()
            dt = (t_vec[j] - t_vec[j + 1]).item()

            alpha_t = self.schedule.alpha(t, dt)
            sigma2_t = self.schedule.sigma2(t, dt)

            # Score from Gaussian transition
            score_gauss = -1.0 * (zt[:, None, :] - alpha_t * z_neighbors) / sigma2_t

            # Log weights
            log_w_gauss = -0.5 * torch.sum(
                (zt[:, None, :] - alpha_t * z_neighbors) ** 2 / sigma2_t,
                dim=2
            )
            log_w_x = -0.5 * torch.sum(
                (x_query[:, None, :] - x_neighbors) ** 2 / (self.nu_x ** 2),
                dim=2
            )
            log_w_mu = -0.5 * torch.sum(
                (mu_query[:, None, :] - mu_neighbors) ** 2 / (self.nu_mu ** 2),
                dim=2
            )

            log_w_total = log_w_gauss + log_w_x + log_w_mu

            # Softmax normalization
            w_temp = torch.exp(log_w_total - torch.max(log_w_total, dim=1, keepdim=True)[0])
            weights = w_temp / torch.sum(w_temp, dim=1, keepdim=True)

            # Weighted score
            score = torch.sum(score_gauss * weights[:, :, None], dim=1)

            # Euler step
            f_t = self.schedule.f(t, dt)
            g2_t = self.schedule.g2(t, dt)
            zt = zt - (f_t * zt - 0.5 * g2_t * score) * dt

        return zt

    def generate(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        mu_data: np.ndarray,
        n_labels: int = 50000,
        batch_size: int = 5000,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate labeled training data via the reverse ODE.

        Args:
            x_data: Training x values (N, x_dim)
            y_data: Training y values (N, x_dim)
            mu_data: Training mu values (N, mu_dim)
            n_labels: Number of labeled samples to generate
            batch_size: Batch size for ODE solving
            seed: Random seed

        Returns:
            Dictionary with 'x_train', 'mu_train', 'z_train', 'y_train'
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Compute scaled displacements
        z_data = (y_data - x_data) * self.diff_scale

        # Select query points
        n_data = x_data.shape[0]
        indices = np.random.permutation(n_data)[:n_labels]
        x_query = x_data[indices]
        mu_query = mu_data[indices]

        # Find neighbors
        x_neighbors, mu_neighbors, z_neighbors = self._find_neighbors(
            x_query, mu_query, x_data, mu_data, z_data
        )

        # Generate labels via reverse ODE
        x_dim = x_data.shape[1]
        z_latent = np.random.randn(n_labels, x_dim)
        y_generated = np.zeros((n_labels, x_dim))

        n_batches = (n_labels + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches), desc="Solving reverse ODE"):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_labels)

            zt = torch.tensor(z_latent[start:end], dtype=torch.float32, device=self.device)
            x_q = torch.tensor(x_query[start:end], dtype=torch.float32, device=self.device)
            mu_q = torch.tensor(mu_query[start:end], dtype=torch.float32, device=self.device)
            x_n = torch.tensor(x_neighbors[start:end], dtype=torch.float32, device=self.device)
            mu_n = torch.tensor(mu_neighbors[start:end], dtype=torch.float32, device=self.device)
            z_n = torch.tensor(z_neighbors[start:end], dtype=torch.float32, device=self.device)

            y_batch = self._solve_reverse_ode(zt, x_n, mu_n, z_n, x_q, mu_q)
            y_generated[start:end] = y_batch.cpu().numpy()

        return {
            'x_train': x_query,
            'mu_train': mu_query,
            'z_train': z_latent,
            'y_train': y_generated,
            'diff_scale': self.diff_scale
        }


class FlowMapModel:
    """
    Main class for the Training-Free Conditional Diffusion Model.

    This class orchestrates the full workflow:
    1. Generate training data from SDE simulations
    2. Generate labeled data using training-free diffusion
    3. Train a neural network for the flow map
    4. Generate samples from the learned model

    Args:
        drift_fn: Drift coefficient a(x, mu)
        diffusion_fn: Diffusion coefficient b(x, mu)
        x_dim: State dimension
        mu_dim: Parameter dimension
        x_range: Tuple (x_min, x_max)
        mu_range: Tuple (mu_min, mu_max)
        dt: Recording time step
        T: Total simulation time
        device: PyTorch device
        hidden_size: Neural network hidden layer size
        n_hidden_layers: Number of hidden layers
        diff_scale: Displacement scaling factor
        init_sampler: Optional custom initial condition sampler
    """

    def __init__(
        self,
        drift_fn: Callable,
        diffusion_fn: Callable,
        x_dim: int = 1,
        mu_dim: int = 1,
        x_range: Tuple[float, float] = (-5.0, 5.0),
        mu_range: Tuple[float, float] = (-1.0, 1.0),
        dt: float = 0.1,
        T: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        hidden_size: int = 128,
        n_hidden_layers: int = 3,
        diff_scale: float = 3.0,
        init_sampler: Optional[Callable] = None
    ):
        self.x_dim = x_dim
        self.mu_dim = mu_dim
        self.x_range = x_range
        self.mu_range = mu_range
        self.dt = dt
        self.T = T
        self.device = torch.device(device)
        self.diff_scale = diff_scale

        # Initialize components
        self.data_generator = TrainingDataGenerator(
            drift_fn=drift_fn,
            diffusion_fn=diffusion_fn,
            x_dim=x_dim,
            mu_dim=mu_dim,
            x_range=x_range,
            mu_range=mu_range,
            dt=dt,
            T=T,
            init_sampler=init_sampler
        )

        self.label_generator = LabelGenerator(
            device=device,
            diff_scale=diff_scale
        )

        # Neural network
        input_dim = x_dim + mu_dim + x_dim  # (x, mu, z)
        self.network = FlowMapNet(
            input_dim=input_dim,
            output_dim=x_dim,
            hidden_size=hidden_size,
            n_hidden_layers=n_hidden_layers
        ).to(self.device)

        # Data storage
        self.training_data = None
        self.labeled_data = None
        self.normalization = None
        self.is_trained = False

    def generate_training_data(
        self,
        n_mu: int = 21,
        n_traj: int = 5000,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate SDE trajectory training data.

        Args:
            n_mu: Number of parameter values
            n_traj: Trajectories per parameter
            seed: Random seed

        Returns:
            Training data dictionary
        """
        self.training_data = self.data_generator.generate_dataset(
            n_mu=n_mu,
            n_traj=n_traj,
            seed=seed
        )
        print(f"Generated {len(self.training_data['x_data'])} training pairs")
        return self.training_data

    def generate_labels(
        self,
        n_labels: int = 50000,
        n_neighbors: int = 2000,
        ode_steps: int = 10000,
        nu_x: float = 1.0,
        nu_mu: float = 0.5,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate labeled data using training-free diffusion.

        Args:
            n_labels: Number of labeled samples
            n_neighbors: Neighbors for score estimation
            ode_steps: ODE discretization steps
            nu_x: Spatial bandwidth
            nu_mu: Parameter bandwidth
            seed: Random seed

        Returns:
            Labeled data dictionary
        """
        if self.training_data is None:
            raise ValueError("Must generate training data first!")

        self.label_generator.n_neighbors = n_neighbors
        self.label_generator.ode_steps = ode_steps
        self.label_generator.nu_x = nu_x
        self.label_generator.nu_mu = nu_mu

        self.labeled_data = self.label_generator.generate(
            x_data=self.training_data['x_data'],
            y_data=self.training_data['y_data'],
            mu_data=self.training_data['mu_data'],
            n_labels=n_labels,
            seed=seed
        )
        print(f"Generated {n_labels} labeled samples")
        return self.labeled_data

    def train_network(
        self,
        n_epochs: int = 5000,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        patience: int = 50,
        train_ratio: float = 0.9,
        weight_decay: float = 1e-6
    ) -> Dict[str, list]:
        """
        Train the neural network.

        Args:
            n_epochs: Maximum training epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            patience: Early stopping patience
            train_ratio: Train/validation split ratio
            weight_decay: L2 regularization

        Returns:
            Training history with 'train_loss' and 'val_loss'
        """
        if self.labeled_data is None:
            raise ValueError("Must generate labeled data first!")

        # Prepare data
        x_train = self.labeled_data['x_train']
        mu_train = self.labeled_data['mu_train']
        z_train = self.labeled_data['z_train']
        y_train = self.labeled_data['y_train']

        # Combine inputs: (x, mu, z)
        X = np.hstack([x_train, mu_train, z_train])
        Y = y_train

        # Filter invalid values
        valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        X, Y = X[valid], Y[valid]

        # Shuffle
        indices = np.random.permutation(len(X))
        X, Y = X[indices], Y[indices]

        # Normalize
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = np.where(X.std(axis=0, keepdims=True) < 1e-8, 1.0, X.std(axis=0, keepdims=True))
        Y_mean = Y.mean(axis=0, keepdims=True)
        Y_std = np.where(Y.std(axis=0, keepdims=True) < 1e-8, 1.0, Y.std(axis=0, keepdims=True))

        X_norm = (X - X_mean) / X_std
        Y_norm = (Y - Y_mean) / Y_std

        self.normalization = {
            'X_mean': torch.tensor(X_mean, dtype=torch.float32, device=self.device),
            'X_std': torch.tensor(X_std, dtype=torch.float32, device=self.device),
            'Y_mean': torch.tensor(Y_mean, dtype=torch.float32, device=self.device),
            'Y_std': torch.tensor(Y_std, dtype=torch.float32, device=self.device)
        }

        # Train/val split
        n_train = int(len(X_norm) * train_ratio)
        X_tr = torch.tensor(X_norm[:n_train], dtype=torch.float32, device=self.device)
        Y_tr = torch.tensor(Y_norm[:n_train], dtype=torch.float32, device=self.device)
        X_val = torch.tensor(X_norm[n_train:], dtype=torch.float32, device=self.device)
        Y_val = torch.tensor(Y_norm[n_train:], dtype=torch.float32, device=self.device)

        # Training setup
        optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        criterion = nn.MSELoss()

        train_loader = DataLoader(
            TensorDataset(X_tr, Y_tr),
            batch_size=batch_size,
            shuffle=True
        )

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        epochs_no_improve = 0

        print(f"\nTraining neural network...")
        print(f"  Training samples: {n_train}")
        print(f"  Validation samples: {len(X_norm) - n_train}")

        for epoch in range(n_epochs):
            # Train
            self.network.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self.network(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)

            # Validate
            self.network.eval()
            with torch.no_grad():
                val_pred = self.network(X_val)
                val_loss = criterion(val_pred, Y_val).item()
            history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.network.save_best()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 100 == 0:
                print(f"  Epoch {epoch:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        self.network.load_best()
        self.is_trained = True
        print(f"  Best validation loss: {best_val_loss:.6f}")

        return history

    def sample(
        self,
        x: Union[float, np.ndarray],
        mu: Union[float, np.ndarray],
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        Generate samples from the learned conditional distribution p(y|x,mu).

        Args:
            x: Initial state (scalar or array)
            mu: Parameter value (scalar or array)
            n_samples: Number of samples

        Returns:
            Samples of shape (n_samples, x_dim)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")

        x = np.atleast_1d(x).astype(np.float32)
        mu = np.atleast_1d(mu).astype(np.float32)

        x_arr = np.tile(x, (n_samples, 1))
        mu_arr = np.tile(mu, (n_samples, 1))
        z_arr = np.random.randn(n_samples, self.x_dim).astype(np.float32)

        X = np.hstack([x_arr, mu_arr, z_arr])
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_norm = (X_tensor - self.normalization['X_mean']) / self.normalization['X_std']

        self.network.eval()
        with torch.no_grad():
            Y_norm = self.network(X_norm)
            Y = Y_norm * self.normalization['Y_std'] + self.normalization['Y_mean']

        displacement_scaled = Y.cpu().numpy()
        y_samples = displacement_scaled / self.diff_scale + x_arr

        return y_samples

    def generate_trajectory(
        self,
        x0: Union[float, np.ndarray],
        mu: float,
        n_steps: int,
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        Generate multi-step trajectories.

        Args:
            x0: Initial state(s)
            mu: Parameter value
            n_steps: Number of time steps
            n_samples: Number of trajectories

        Returns:
            Final state samples of shape (n_samples, x_dim)
        """
        x0 = np.atleast_1d(x0).astype(np.float32)

        if len(x0) == 1:
            x_current = np.tile(x0, (n_samples, 1))
        else:
            x_current = x0.reshape(-1, self.x_dim)
            n_samples = x_current.shape[0]

        for _ in range(n_steps):
            # Generate next step for all samples
            mu_arr = mu * np.ones((n_samples, self.mu_dim), dtype=np.float32)
            z_arr = np.random.randn(n_samples, self.x_dim).astype(np.float32)

            X = np.hstack([x_current, mu_arr, z_arr])
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            X_norm = (X_tensor - self.normalization['X_mean']) / self.normalization['X_std']

            self.network.eval()
            with torch.no_grad():
                Y_norm = self.network(X_norm)
                Y = Y_norm * self.normalization['Y_std'] + self.normalization['Y_mean']

            displacement_scaled = Y.cpu().numpy()
            x_current = displacement_scaled / self.diff_scale + x_current

        return x_current

    def save(self, path: str):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        save_dict = {
            'network_state': self.network.state_dict(),
            'normalization': {k: v.cpu() for k, v in self.normalization.items()} if self.normalization else None,
            'config': {
                'x_dim': self.x_dim,
                'mu_dim': self.mu_dim,
                'x_range': self.x_range,
                'mu_range': self.mu_range,
                'dt': self.dt,
                'T': self.T,
                'diff_scale': self.diff_scale,
                'hidden_size': self.network.hidden_size,
                'is_trained': self.is_trained
            }
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load the model from disk."""
        save_dict = torch.load(path, map_location=self.device)

        self.network.load_state_dict(save_dict['network_state'])
        if save_dict['normalization'] is not None:
            self.normalization = {k: v.to(self.device) for k, v in save_dict['normalization'].items()}

        config = save_dict['config']
        self.x_dim = config['x_dim']
        self.mu_dim = config['mu_dim']
        self.x_range = config['x_range']
        self.mu_range = config['mu_range']
        self.dt = config['dt']
        self.T = config['T']
        self.diff_scale = config['diff_scale']
        self.is_trained = config['is_trained']

        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Training-Free Conditional Diffusion Model")
    print("=" * 60)

    # Example: Brownian motion with drift
    # SDE: dX_t = mu * dt + dW_t
    print("\nExample: Brownian Motion with Parameter-Dependent Drift")
    print("SDE: dX_t = mu * dt + dW_t, mu in [-1, 1]")

    def drift(x, mu):
        return mu * np.ones_like(x)

    def diffusion(x, mu):
        return np.ones_like(x)

    model = FlowMapModel(
        drift_fn=drift,
        diffusion_fn=diffusion,
        x_dim=1,
        mu_dim=1,
        x_range=(-5.0, 5.0),
        mu_range=(-1.0, 1.0)
    )

    # Generate training data (small for demo)
    model.generate_training_data(n_mu=11, n_traj=1000, seed=42)

    # Generate labeled data (small for demo)
    model.generate_labels(n_labels=5000, ode_steps=1000, seed=42)

    # Train network (fewer epochs for demo)
    history = model.train_network(n_epochs=500, patience=20)

    # Generate samples
    x_test = 2.0
    mu_test = 0.5
    samples = model.sample(x=x_test, mu=mu_test, n_samples=1000)

    print(f"\nSamples from p(X_{{n+1}} | X_n = {x_test}, mu = {mu_test}):")
    print(f"  Mean: {np.mean(samples):.4f} (Exact: {x_test + mu_test * model.dt:.4f})")
    print(f"  Std:  {np.std(samples):.4f} (Exact: {np.sqrt(model.dt):.4f})")

    print("\n=== Demo Complete ===")
