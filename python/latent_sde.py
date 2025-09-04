"""
Latent Neural SDE
=================

Latent SDE model for time series that models observations as noisy
projections of a continuous latent stochastic process.

Architecture:
    Encoder:   observations -> initial latent distribution q(z_0)
    Prior SDE: dZ = f_theta(Z,t) dt + g_phi(Z,t) dW
    Posterior:  dZ = [f_theta(Z,t) + g_phi(Z,t)*u_omega(Z,t)] dt + g_phi(Z,t) dW
    Decoder:   Z(t_i) -> Y_hat(t_i)

Training objective:
    ELBO = E_q[sum_i log p(Y_i | Z(t_i))] - KL(posterior || prior)

The KL divergence has a simple form via Girsanov's theorem:
    KL = 0.5 * E_q[integral_0^T ||u_omega(Z(t), t)||^2 dt]
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sde import DiffusionNetwork, DriftNetwork


class Encoder(nn.Module):
    """
    GRU-based encoder that maps a sequence of observations to an
    initial latent state distribution q(z_0 | Y) = N(mu, sigma^2).
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Map GRU hidden state to mean and log-variance
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observations to initial latent distribution.

        Args:
            observations: (batch, seq_len, obs_dim)

        Returns:
            mean: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        # Process sequence with GRU
        _, hidden = self.gru(observations)
        # Take the last layer's hidden state
        h = hidden[-1]  # (batch, hidden_dim)

        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)

        return mean, logvar


class PosteriorDriftCorrection(nn.Module):
    """
    Posterior drift correction network u_omega(Z, t).

    This network learns the difference between the posterior and prior
    drift, conditioned on the observed data (implicitly through training).
    The KL divergence is simply 0.5 * integral ||u_omega||^2 dt.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Initialize with small weights so posterior starts near prior
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute posterior drift correction.

        Args:
            z: Latent state (batch, latent_dim)
            t: Time (scalar or (batch, 1))

        Returns:
            Correction vector (batch, latent_dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(z.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        inp = torch.cat([z, t], dim=-1)
        return self.net(inp)


class Decoder(nn.Module):
    """
    Decoder network: maps latent state to observation space.
    p(Y | Z) = N(h_psi(Z), sigma_obs^2)
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Learnable observation noise
        self.log_sigma_obs = nn.Parameter(torch.tensor(-2.0))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent state to observation distribution parameters.

        Args:
            z: Latent state (batch, latent_dim) or (time, batch, latent_dim)

        Returns:
            mean: Predicted observation mean
            sigma: Observation noise std
        """
        mean = self.net(z)
        sigma = self.log_sigma_obs.exp()
        return mean, sigma


class LatentSDE(nn.Module):
    """
    Latent Neural SDE for time series modeling.

    This model:
    1. Encodes observations into an initial latent state
    2. Evolves the latent state via a Neural SDE (posterior during training)
    3. Decodes latent states back to observation space
    4. Regularizes via KL divergence between posterior and prior SDEs

    Compatible with torchsde for efficient training via stochastic adjoint.

    Args:
        obs_dim: Dimension of observations
        latent_dim: Dimension of latent state
        hidden_dim: Hidden layer size
        encoder_layers: Number of GRU layers in encoder
    """

    # Required by torchsde
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        obs_dim: int = 1,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        encoder_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        # Encoder: observations -> q(z_0)
        self.encoder = Encoder(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
        )

        # Prior SDE components
        self.drift_net = DriftNetwork(
            state_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
        )

        self.diffusion_net = DiffusionNetwork(
            state_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            diffusion_type="diagonal",
        )

        # Posterior drift correction u_omega
        self.posterior_correction = PosteriorDriftCorrection(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

        # Decoder: latent state -> observations
        self.decoder = Decoder(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
        )

        # Track KL divergence during forward pass
        self._kl_accumulator = 0.0
        self._kl_steps = 0
        self._use_posterior = True

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Drift function for torchsde interface.

        During training (posterior): f_theta + g_phi * u_omega
        During generation (prior):   f_theta

        Args:
            t: Current time
            y: Current latent state (batch, latent_dim)

        Returns:
            Drift vector (batch, latent_dim)
        """
        prior_drift = self.drift_net(y, t)

        if self._use_posterior:
            correction = self.posterior_correction(y, t)
            diffusion = self.diffusion_net(y, t)
            posterior_drift = prior_drift + diffusion * correction

            # Accumulate KL divergence: 0.5 * ||u_omega||^2 * dt
            self._kl_accumulator += 0.5 * (correction ** 2).sum(dim=-1).mean()
            self._kl_steps += 1

            return posterior_drift
        else:
            return prior_drift

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Diffusion function for torchsde interface.

        Same for both prior and posterior SDEs.

        Args:
            t: Current time
            y: Current latent state (batch, latent_dim)

        Returns:
            Diffusion coefficient (batch, latent_dim)
        """
        return self.diffusion_net(y, t)

    def reparameterize(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample z_0 from q(z_0) using reparameterization trick."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        observations: torch.Tensor,
        ts: torch.Tensor,
        dt: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: encode, solve SDE, decode.

        Args:
            observations: (batch, seq_len, obs_dim) observed time series
            ts: (seq_len,) time points for observations
            dt: SDE solver step size

        Returns:
            Dictionary with:
                - predictions: decoded observations (seq_len, batch, obs_dim)
                - z_mean: encoder mean
                - z_logvar: encoder log-variance
                - kl_divergence: KL(posterior || prior)
                - latent_paths: latent trajectories (seq_len, batch, latent_dim)
        """
        batch_size = observations.shape[0]

        # Encode observations to initial latent distribution
        z_mean, z_logvar = self.encoder(observations)

        # Sample initial latent state
        z0 = self.reparameterize(z_mean, z_logvar)

        # Reset KL accumulator
        self._kl_accumulator = 0.0
        self._kl_steps = 0
        self._use_posterior = True

        # Solve posterior SDE
        try:
            import torchsde

            latent_paths = torchsde.sdeint(
                self, z0, ts, dt=dt, method="euler"
            )
        except ImportError:
            latent_paths = self._manual_solve(z0, ts, dt)

        # Compute KL divergence (average over solver steps, scaled by T)
        T = (ts[-1] - ts[0]).item()
        if self._kl_steps > 0:
            kl_divergence = self._kl_accumulator / self._kl_steps * T
        else:
            kl_divergence = torch.tensor(0.0, device=z0.device)

        # Add KL from initial state: KL(q(z_0) || p(z_0))
        # Assuming p(z_0) = N(0, I)
        kl_initial = -0.5 * (
            1 + z_logvar - z_mean.pow(2) - z_logvar.exp()
        ).sum(dim=-1).mean()

        total_kl = kl_divergence + kl_initial

        # Decode latent paths to observation space
        predictions, obs_sigma = self.decoder(latent_paths)

        return {
            "predictions": predictions,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "kl_divergence": total_kl,
            "latent_paths": latent_paths,
            "obs_sigma": obs_sigma,
        }

    def _manual_solve(
        self, z0: torch.Tensor, ts: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """Manual Euler-Maruyama solver for when torchsde is not available."""
        trajectory = [z0]
        z = z0
        t_current = ts[0]

        for i in range(1, len(ts)):
            t_target = ts[i]
            while t_current < t_target - 1e-8:
                t_next = min(t_current + dt, t_target)
                actual_dt = (t_next - t_current).item()

                drift = self.f(t_current, z)
                diffusion = self.g(t_current, z)
                dW = torch.randn_like(z) * math.sqrt(actual_dt)
                z = z + drift * actual_dt + diffusion * dW

                t_current = t_next

            trajectory.append(z.clone())

        return torch.stack(trajectory)

    @torch.no_grad()
    def generate(
        self,
        z0: Optional[torch.Tensor] = None,
        ts: Optional[torch.Tensor] = None,
        num_paths: int = 100,
        dt: float = 0.01,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """
        Generate sample paths from the prior SDE.

        Args:
            z0: Initial latent state (if None, sample from N(0,I))
            ts: Time points (if None, use linspace(0,1,num_steps))
            num_paths: Number of paths to generate
            dt: Solver step size
            num_steps: Number of time steps if ts is None

        Returns:
            Generated observations (num_paths, num_steps, obs_dim)
        """
        self._use_posterior = False

        if z0 is None:
            z0 = torch.randn(num_paths, self.latent_dim)
        elif z0.shape[0] != num_paths:
            z0 = z0.expand(num_paths, -1)

        device = next(self.parameters()).device
        z0 = z0.to(device)

        if ts is None:
            ts = torch.linspace(0, 1, num_steps, device=device)

        try:
            import torchsde

            latent_paths = torchsde.sdeint(
                self, z0, ts, dt=dt, method="euler"
            )
        except ImportError:
            latent_paths = self._manual_solve(z0, ts, dt)

        # Decode to observation space
        predictions, _ = self.decoder(latent_paths)

        # (num_steps, num_paths, obs_dim) -> (num_paths, num_steps, obs_dim)
        return predictions.permute(1, 0, 2)

    def compute_elbo(
        self,
        observations: torch.Tensor,
        ts: torch.Tensor,
        dt: float = 0.01,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Evidence Lower Bound (ELBO).

        ELBO = E_q[sum_i log p(Y_i | Z(t_i))] - beta * KL(q || p)

        Args:
            observations: (batch, seq_len, obs_dim)
            ts: (seq_len,) time points
            dt: Solver step size
            beta: KL annealing weight (0 -> 1 during training)

        Returns:
            Dictionary with elbo, reconstruction_loss, kl_divergence
        """
        result = self.forward(observations, ts, dt=dt)

        # Reconstruction loss: -log p(Y | Z)
        # Assuming Gaussian observation model
        predictions = result["predictions"]  # (seq_len, batch, obs_dim)
        obs_sigma = result["obs_sigma"]

        # Transpose observations to match: (batch, seq_len, obs_dim) -> (seq_len, batch, obs_dim)
        targets = observations.permute(1, 0, 2)

        # Gaussian log-likelihood
        recon_loss = 0.5 * (
            ((predictions - targets) / obs_sigma) ** 2
            + 2 * torch.log(obs_sigma)
            + math.log(2 * math.pi)
        ).sum(dim=(0, 2)).mean()

        kl = result["kl_divergence"]

        elbo = -recon_loss - beta * kl

        return {
            "elbo": elbo,
            "reconstruction_loss": recon_loss,
            "kl_divergence": kl,
            "predictions": predictions,
            "latent_paths": result["latent_paths"],
        }
