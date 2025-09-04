//! Generate sample paths from a Neural SDE model.
//!
//! Generates Monte Carlo paths and computes path statistics.
//! Useful for:
//! - Visualizing the learned dynamics
//! - Monte Carlo pricing
//! - Uncertainty quantification
//!
//! Usage:
//!   cargo run --bin generate_paths -- --num-paths 1000 --steps 100

use anyhow::Result;
use clap::Parser;
use ndarray::Array1;
use neural_sde_trading::model::NeuralSDE;
use neural_sde_trading::solvers;

#[derive(Parser, Debug)]
#[command(name = "generate_paths", about = "Generate sample paths from Neural SDE")]
struct Args {
    /// Number of sample paths to generate
    #[arg(short, long, default_value_t = 500)]
    num_paths: usize,

    /// Number of time steps
    #[arg(short, long, default_value_t = 100)]
    steps: usize,

    /// State dimension
    #[arg(long, default_value_t = 4)]
    state_dim: usize,

    /// Hidden dimension
    #[arg(long, default_value_t = 64)]
    hidden_dim: usize,

    /// SDE solver step size
    #[arg(long, default_value_t = 0.01)]
    dt: f64,

    /// SDE solver method: euler, milstein, srk
    #[arg(long, default_value = "euler")]
    method: String,

    /// Output CSV file for paths
    #[arg(short, long, default_value = "paths.csv")]
    output: String,

    /// Also generate GBM paths for comparison
    #[arg(long)]
    compare_gbm: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║        Neural SDE Trading - Path Generator              ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Paths:     {:>10}                                   ║", args.num_paths);
    println!("║ Steps:     {:>10}                                   ║", args.steps);
    println!("║ State dim: {:>10}                                   ║", args.state_dim);
    println!("║ Method:    {:>10}                                   ║", args.method);
    println!("║ dt:        {:>10.4}                                 ║", args.dt);
    println!("╚══════════════════════════════════════════════════════════╝");

    // Create model
    let model = NeuralSDE::new(args.state_dim, args.hidden_dim, 3);

    // Time grid
    let ts: Vec<f64> = (0..=args.steps).map(|i| i as f64 / args.steps as f64).collect();

    // Initial state
    let x0 = Array1::zeros(args.state_dim);

    // Generate paths
    println!("\nGenerating {} paths with {} method...", args.num_paths, args.method);

    let start_time = std::time::Instant::now();

    let paths = model.sample_paths_parallel(&x0, &ts, args.dt, args.num_paths);

    let elapsed = start_time.elapsed();
    println!(
        "Generated {} paths in {:.2}s ({:.0} paths/sec)",
        args.num_paths,
        elapsed.as_secs_f64(),
        args.num_paths as f64 / elapsed.as_secs_f64()
    );

    // Compute statistics at each time point
    println!("\n{:>6} {:>12} {:>12} {:>12} {:>12}", "Time", "Mean", "Std", "5%ile", "95%ile");
    println!("{}", "-".repeat(60));

    for (t_idx, &t) in ts.iter().enumerate().step_by(ts.len() / 10 + 1) {
        let values: Vec<f64> = paths.iter().map(|p| p[t_idx][0]).collect();
        let n = values.len() as f64;

        let mean: f64 = values.iter().sum::<f64>() / n;
        let std: f64 = {
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            var.sqrt()
        };

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p5 = sorted[(0.05 * n) as usize];
        let p95 = sorted[(0.95 * n) as usize];

        println!(
            "{:>6.3} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            t, mean, std, p5, p95
        );
    }

    // Final time statistics
    println!("\n\nDetailed Statistics at T=1.0:");
    println!("{}", "=".repeat(50));

    let final_values: Vec<f64> = paths.iter().map(|p| p.last().unwrap()[0]).collect();
    let n = final_values.len() as f64;
    let mean: f64 = final_values.iter().sum::<f64>() / n;
    let std: f64 = {
        let var = final_values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        var.sqrt()
    };

    let skewness: f64 = if std > 1e-10 {
        final_values
            .iter()
            .map(|v| ((v - mean) / std).powi(3))
            .sum::<f64>()
            / n
    } else {
        0.0
    };

    let kurtosis: f64 = if std > 1e-10 {
        final_values
            .iter()
            .map(|v| ((v - mean) / std).powi(4))
            .sum::<f64>()
            / n
            - 3.0
    } else {
        0.0
    };

    println!("Mean:       {:.6}", mean);
    println!("Std Dev:    {:.6}", std);
    println!("Skewness:   {:.4}", skewness);
    println!("Kurtosis:   {:.4} (excess)", kurtosis);
    println!(
        "Min:        {:.6}",
        final_values.iter().cloned().fold(f64::INFINITY, f64::min)
    );
    println!(
        "Max:        {:.6}",
        final_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Compare with GBM if requested
    if args.compare_gbm {
        println!("\n\nComparison with Geometric Brownian Motion:");
        println!("{}", "=".repeat(50));

        let gbm_paths: Vec<Vec<f64>> = (0..args.num_paths)
            .map(|_| solvers::geometric_brownian_motion(0.05, 0.2, 1.0, &ts))
            .collect();

        let gbm_finals: Vec<f64> = gbm_paths.iter().map(|p| *p.last().unwrap()).collect();
        let gbm_mean = gbm_finals.iter().sum::<f64>() / gbm_finals.len() as f64;
        let gbm_std = {
            let var = gbm_finals
                .iter()
                .map(|v| (v - gbm_mean).powi(2))
                .sum::<f64>()
                / (gbm_finals.len() - 1) as f64;
            var.sqrt()
        };

        println!(
            "{:>20} {:>12} {:>12}",
            "Metric", "Neural SDE", "GBM"
        );
        println!("{}", "-".repeat(50));
        println!(
            "{:>20} {:>12.6} {:>12.6}",
            "Mean", mean, gbm_mean
        );
        println!(
            "{:>20} {:>12.6} {:>12.6}",
            "Std Dev", std, gbm_std
        );
        println!(
            "{:>20} {:>12.4} {:>12.4}",
            "Skewness", skewness, 0.0
        );
        println!(
            "{:>20} {:>12.4} {:>12.4}",
            "Kurtosis", kurtosis, 0.0
        );
    }

    // Save paths to CSV
    println!("\nSaving paths to {}...", args.output);
    let mut writer = csv::Writer::from_path(&args.output)?;

    // Header: time, path_0, path_1, ...
    let mut header = vec!["time".to_string()];
    for i in 0..args.num_paths.min(100) {
        // Save up to 100 paths
        header.push(format!("path_{}", i));
    }
    writer.write_record(&header)?;

    // Data rows
    for (t_idx, &t) in ts.iter().enumerate() {
        let mut row = vec![format!("{:.6}", t)];
        for path_idx in 0..args.num_paths.min(100) {
            row.push(format!("{:.6}", paths[path_idx][t_idx][0]));
        }
        writer.write_record(&row)?;
    }

    writer.flush()?;
    println!("Saved {} paths to {}", args.num_paths.min(100), args.output);

    Ok(())
}
