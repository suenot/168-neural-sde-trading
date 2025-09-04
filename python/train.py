"""
Training Loop
=============

Training pipeline for Neural SDE models using the stochastic adjoint method.
Supports both the basic Neural SDE and the Latent SDE for time series.

Key features:
    - KL annealing (beta warmup)
    - Gradient clipping
    - Learning rate scheduling
    - Checkpoint saving
    - Validation monitoring
    - Multiple SDE sample averaging
"""

import argparse
import json
import math
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import (
    create_dataloaders,
    generate_synthetic_data,
    load_crypto_data,
    load_stock_data,
    prepare_sde_data,
)
from latent_sde import LatentSDE
from neural_sde import FinancialNeuralSDE, NeuralSDE


def get_kl_weight(epoch: int, warmup_epochs: int, max_beta: float = 1.0) -> float:
    """
    Compute KL annealing weight.

    Linear warmup from 0 to max_beta over warmup_epochs.
    This prevents posterior collapse early in training.

    Args:
        epoch: Current epoch
        warmup_epochs: Number of warmup epochs
        max_beta: Maximum KL weight

    Returns:
        Current KL weight (beta)
    """
    if warmup_epochs <= 0:
        return max_beta
    return min(max_beta, max_beta * epoch / warmup_epochs)


def train_latent_sde(
    model: LatentSDE,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    ts: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    lr_min: float = 1e-5,
    kl_warmup_epochs: int = 30,
    max_beta: float = 1.0,
    grad_clip: float = 10.0,
    num_sde_samples: int = 4,
    dt: float = 0.01,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """
    Train a Latent Neural SDE model.

    Args:
        model: LatentSDE model
        train_loader: Training data loader
        val_loader: Validation data loader
        ts: Time points tensor
        epochs: Number of training epochs
        lr: Initial learning rate
        lr_min: Minimum learning rate
        kl_warmup_epochs: Number of KL annealing warmup epochs
        max_beta: Maximum KL weight
        grad_clip: Maximum gradient norm
        num_sde_samples: Number of SDE samples to average per data point
        dt: SDE solver step size
        checkpoint_dir: Directory for saving checkpoints
        device: Training device ('cpu' or 'cuda')
        verbose: Print training progress

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    ts = ts.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr_min
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
        "val_loss": [],
        "val_recon": [],
        "val_kl": [],
        "lr": [],
        "beta": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        # KL annealing weight
        beta = get_kl_weight(epoch, kl_warmup_epochs, max_beta)

        # ─── Training ───
        model.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        train_batches = 0

        for batch_obs, batch_targets in train_loader:
            batch_obs = batch_obs.to(device)

            optimizer.zero_grad()

            # Average over multiple SDE samples for variance reduction
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0

            for _ in range(num_sde_samples):
                result = model.compute_elbo(batch_obs, ts, dt=dt, beta=beta)
                total_loss += -result["elbo"]
                total_recon += result["reconstruction_loss"]
                total_kl += result["kl_divergence"]

            loss = total_loss / num_sde_samples
            recon = total_recon / num_sde_samples
            kl = total_kl / num_sde_samples

            loss.backward()

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_loss_sum += loss.item()
            train_recon_sum += recon.item()
            train_kl_sum += kl.item()
            train_batches += 1

        scheduler.step()

        # ─── Validation ───
        model.eval()
        val_loss_sum = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_obs, batch_targets in val_loader:
                batch_obs = batch_obs.to(device)

                result = model.compute_elbo(batch_obs, ts, dt=dt, beta=beta)
                val_loss = -result["elbo"]

                val_loss_sum += val_loss.item()
                val_recon_sum += result["reconstruction_loss"].item()
                val_kl_sum += result["kl_divergence"].item()
                val_batches += 1

        # Compute averages
        avg_train_loss = train_loss_sum / max(train_batches, 1)
        avg_train_recon = train_recon_sum / max(train_batches, 1)
        avg_train_kl = train_kl_sum / max(train_batches, 1)
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_recon = val_recon_sum / max(val_batches, 1)
        avg_val_kl = val_kl_sum / max(val_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["train_recon"].append(avg_train_recon)
        history["train_kl"].append(avg_train_kl)
        history["val_loss"].append(avg_val_loss)
        history["val_recon"].append(avg_val_recon)
        history["val_kl"].append(avg_val_kl)
        history["lr"].append(current_lr)
        history["beta"].append(beta)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                },
                os.path.join(checkpoint_dir, "best_model.pt"),
            )

        epoch_time = time.time() - epoch_start

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train: {avg_train_loss:.4f} (R:{avg_train_recon:.4f} K:{avg_train_kl:.4f}) | "
                f"Val: {avg_val_loss:.4f} | "
                f"beta={beta:.3f} lr={current_lr:.2e} | "
                f"{epoch_time:.1f}s"
            )

    # Save final model
    torch.save(
        {
            "epoch": epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        os.path.join(checkpoint_dir, "final_model.pt"),
    )

    # Save training history
    with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    return history


def train_basic_neural_sde(
    model: NeuralSDE,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    ts: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    grad_clip: float = 10.0,
    dt: float = 0.01,
    device: str = "cpu",
) -> Dict:
    """
    Train a basic Neural SDE by matching path statistics.

    This simpler training procedure fits the Neural SDE to reproduce
    the observed path distributions (mean and variance at each time point).

    Args:
        model: NeuralSDE model
        train_data: Training trajectories (batch, time, features)
        val_data: Validation trajectories
        ts: Time points
        epochs: Training epochs
        lr: Learning rate
        grad_clip: Gradient clipping norm
        dt: SDE solver step size
        device: Training device

    Returns:
        Training history
    """
    model = model.to(device)
    ts = ts.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()

        # Use the initial state from training data
        y0 = train_data[:, 0, :]  # (batch, features)

        # Solve SDE forward
        pred_paths = model.forward(y0, ts, dt=dt)
        # pred_paths: (num_times, batch, features)

        # Match path statistics: mean squared error at each time point
        target_paths = train_data.permute(1, 0, 2)  # (time, batch, features)
        loss = ((pred_paths - target_paths) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            y0_val = val_data[:, 0, :]
            val_pred = model.forward(y0_val, ts, dt=dt)
            val_target = val_data.permute(1, 0, 2)
            val_loss = ((val_pred - val_target) ** 2).mean()

        scheduler.step(val_loss)

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train: {loss.item():.6f} | Val: {val_loss.item():.6f}"
            )

    return history


def main():
    """Main training entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description="Train Neural SDE model")

    parser.add_argument(
        "--model",
        type=str,
        default="latent",
        choices=["basic", "latent", "financial"],
        help="Model type to train",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="synthetic",
        choices=["synthetic", "bybit", "stock"],
        help="Data source",
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--window-size", type=int, default=60, help="Sequence window size")
    parser.add_argument("--dt", type=float, default=0.02, help="SDE solver step size")
    parser.add_argument(
        "--kl-warmup", type=int, default=30, help="KL annealing warmup epochs"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )

    args = parser.parse_args()

    # Load data
    print("=" * 60)
    print(f"Neural SDE Training")
    print(f"Model: {args.model} | Data: {args.data} | Symbol: {args.symbol}")
    print("=" * 60)

    if args.data == "bybit":
        data_dict = load_crypto_data(symbol=args.symbol, days=90)
    elif args.data == "stock":
        data_dict = load_stock_data(symbol=args.symbol)
    else:
        df = generate_synthetic_data(num_points=2000, regime_switching=True)
        data_dict = prepare_sde_data(
            df, window_size=args.window_size, prediction_horizon=24
        )

    # Create data loaders
    loaders = create_dataloaders(data_dict, batch_size=args.batch_size)

    obs_dim = data_dict["observations"].shape[-1]
    ts = loaders["ts"]

    print(f"\nFeatures ({obs_dim}): {data_dict['feature_columns']}")

    # Create model
    if args.model == "latent":
        model = LatentSDE(
            obs_dim=obs_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
        )
        print(f"\nLatent SDE: obs_dim={obs_dim}, latent_dim={args.latent_dim}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        history = train_latent_sde(
            model=model,
            train_loader=loaders["train_loader"],
            val_loader=loaders["val_loader"],
            ts=ts,
            epochs=args.epochs,
            lr=args.lr,
            kl_warmup_epochs=args.kl_warmup,
            dt=args.dt,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
        )

    elif args.model == "financial":
        model = FinancialNeuralSDE(
            num_features=obs_dim,
            hidden_dim=args.hidden_dim,
        )
        print(f"\nFinancial Neural SDE: features={obs_dim}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        history = train_basic_neural_sde(
            model=model,
            train_data=data_dict["observations"][:200],
            val_data=data_dict["observations"][200:250],
            ts=ts,
            epochs=args.epochs,
            lr=args.lr,
            dt=args.dt,
            device=args.device,
        )

    else:
        model = NeuralSDE(
            state_dim=obs_dim,
            hidden_dim=args.hidden_dim,
        )
        print(f"\nBasic Neural SDE: state_dim={obs_dim}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        history = train_basic_neural_sde(
            model=model,
            train_data=data_dict["observations"][:200],
            val_data=data_dict["observations"][200:250],
            ts=ts,
            epochs=args.epochs,
            lr=args.lr,
            dt=args.dt,
            device=args.device,
        )

    print("\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")

    if "val_loss" in history and history["val_loss"]:
        print(f"Final val loss: {history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
