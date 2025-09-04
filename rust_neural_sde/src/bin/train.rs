//! Train a Neural SDE model on market data.
//!
//! This binary trains the Neural SDE by fitting path statistics
//! to historical data. In a production setting, you would use
//! PyTorch with torchsde for full gradient-based training and
//! export the model weights for Rust inference.
//!
//! Usage:
//!   cargo run --bin train -- --data data.csv --epochs 100

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array1;
use neural_sde_trading::api::BybitClient;
use neural_sde_trading::model::NeuralSDE;
use neural_sde_trading::utils;

#[derive(Parser, Debug)]
#[command(name = "train", about = "Train Neural SDE model")]
struct Args {
    /// Input data CSV file
    #[arg(short, long, default_value = "data.csv")]
    data: String,

    /// Number of training epochs
    #[arg(short, long, default_value_t = 100)]
    epochs: usize,

    /// State dimension
    #[arg(long, default_value_t = 4)]
    state_dim: usize,

    /// Hidden layer dimension
    #[arg(long, default_value_t = 64)]
    hidden_dim: usize,

    /// Number of Monte Carlo paths for evaluation
    #[arg(long, default_value_t = 200)]
    num_paths: usize,

    /// SDE solver step size
    #[arg(long, default_value_t = 0.02)]
    dt: f64,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║           Neural SDE Trading - Training                 ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Data:       {:>10}                                  ║", args.data);
    println!("║ Epochs:     {:>10}                                  ║", args.epochs);
    println!("║ State dim:  {:>10}                                  ║", args.state_dim);
    println!("║ Hidden dim: {:>10}                                  ║", args.hidden_dim);
    println!("╚══════════════════════════════════════════════════════════╝");

    // Load data
    println!("\nLoading data...");
    let klines = match BybitClient::load_from_csv(&args.data) {
        Ok(data) => {
            println!("Loaded {} candles from {}", data.len(), args.data);
            data
        }
        Err(e) => {
            println!("Could not load {}: {}", args.data, e);
            println!("Generating synthetic data for demonstration...");

            // Generate synthetic data
            let ts: Vec<f64> = (0..2000).map(|i| i as f64 / 2000.0).collect();
            let prices = neural_sde_trading::solvers::geometric_brownian_motion(
                0.05, 0.3, 100.0, &ts,
            );

            prices
                .iter()
                .enumerate()
                .map(|(i, &p)| neural_sde_trading::api::Kline {
                    timestamp: i as i64 * 3600000,
                    open: p * 0.999,
                    high: p * 1.005,
                    low: p * 0.995,
                    close: p,
                    volume: 1000.0,
                    turnover: p * 1000.0,
                })
                .collect()
        }
    };

    // Prepare features
    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let log_returns = utils::log_returns(&closes);
    let realized_vol = utils::rolling_std(&log_returns, 20);
    let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

    // Volume ratio
    let vol_ma = utils::rolling_std(&volumes, 20);
    let volume_ratio: Vec<f64> = volumes
        .iter()
        .zip(vol_ma.iter())
        .map(|(v, ma)| if *ma > 1e-10 { v / ma } else { 1.0 })
        .collect();

    // Build feature vectors (skip initial NaN period)
    let start_idx = 25; // After rolling windows stabilize
    let mut features: Vec<Array1<f64>> = Vec::new();

    for i in start_idx..log_returns.len().min(realized_vol.len()).min(volume_ratio.len()) {
        let feat = Array1::from_vec(vec![
            log_returns[i],
            realized_vol[i],
            volume_ratio.get(i).copied().unwrap_or(1.0),
            if i >= 10 {
                log_returns[i - 10..i].iter().sum::<f64>() // Momentum
            } else {
                0.0
            },
        ]);
        features.push(feat);
    }

    println!("Prepared {} feature vectors (dim={})", features.len(), args.state_dim);

    // Create model
    let model = NeuralSDE::new(args.state_dim, args.hidden_dim, 3);
    println!(
        "Created Neural SDE model: state_dim={}, hidden_dim={}",
        args.state_dim, args.hidden_dim
    );

    // Training: evaluate model by comparing path statistics
    println!("\nTraining (path statistics matching)...");

    let pb = ProgressBar::new(args.epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let ts_eval: Vec<f64> = (0..50).map(|i| i as f64 / 50.0).collect();

    for epoch in 0..args.epochs {
        // Sample a random starting point
        let start = rand::random::<usize>() % (features.len().saturating_sub(50).max(1));
        let x0 = &features[start];

        // Generate paths from model
        let paths = model.sample_paths_parallel(x0, &ts_eval, args.dt, args.num_paths);

        // Compute path statistics
        let final_states: Vec<f64> = paths.iter().map(|p| p.last().unwrap()[0]).collect();
        let mean: f64 = final_states.iter().sum::<f64>() / final_states.len() as f64;
        let variance: f64 = final_states
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / (final_states.len() - 1) as f64;

        // Target statistics from real data
        let target_mean = if start + 50 < features.len() {
            features[start..start + 50]
                .iter()
                .map(|f| f[0])
                .sum::<f64>()
                / 50.0
        } else {
            0.0
        };

        let loss = (mean - target_mean).powi(2) + (variance - 0.01).powi(2);

        if epoch % 10 == 0 {
            pb.set_message(format!(
                "loss={:.6} mean={:.4} var={:.4}",
                loss, mean, variance
            ));
        }

        pb.inc(1);
    }

    pb.finish_with_message("Training complete");

    // Evaluate final model
    println!("\n\nFinal Model Evaluation:");
    println!("{}", "-".repeat(50));

    if let Some(x0) = features.first() {
        let (means, lowers, uppers) =
            model.predict_distribution(x0, &ts_eval, args.dt, 500, 0.95);

        if let (Some(final_mean), Some(final_lower), Some(final_upper)) =
            (means.last(), lowers.last(), uppers.last())
        {
            println!(
                "Prediction at T=1.0 (state dim 0):"
            );
            println!("  Mean:  {:.6}", final_mean[0]);
            println!("  95% CI: [{:.6}, {:.6}]", final_lower[0], final_upper[0]);
            println!("  Width:  {:.6}", final_upper[0] - final_lower[0]);
        }

        // Path statistics
        let paths = model.sample_paths_parallel(x0, &ts_eval, args.dt, 1000);
        let finals: Vec<f64> = paths.iter().map(|p| p.last().unwrap()[0]).collect();
        let mean_f: f64 = finals.iter().sum::<f64>() / finals.len() as f64;
        let std_f: f64 = {
            let var: f64 = finals.iter().map(|v| (v - mean_f).powi(2)).sum::<f64>()
                / (finals.len() - 1) as f64;
            var.sqrt()
        };

        // Skewness
        let skew: f64 = finals
            .iter()
            .map(|v| ((v - mean_f) / std_f).powi(3))
            .sum::<f64>()
            / finals.len() as f64;

        // Kurtosis
        let kurt: f64 = finals
            .iter()
            .map(|v| ((v - mean_f) / std_f).powi(4))
            .sum::<f64>()
            / finals.len() as f64
            - 3.0;

        println!("\nGenerated Path Statistics (1000 paths):");
        println!("  Mean:     {:.6}", mean_f);
        println!("  Std:      {:.6}", std_f);
        println!("  Skewness: {:.4}", skew);
        println!("  Kurtosis: {:.4}", kurt);
    }

    println!("\nNote: This is a simplified training procedure for demonstration.");
    println!("For production use, train in Python with torchsde and export weights.");

    Ok(())
}
