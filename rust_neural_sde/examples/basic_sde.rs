//! Basic Neural SDE Example
//!
//! Demonstrates the core Neural SDE functionality:
//! 1. Creating a Neural SDE model
//! 2. Generating sample paths
//! 3. Computing path statistics
//! 4. Comparing solvers (Euler-Maruyama, Milstein, SRK)
//!
//! Run with: cargo run --example basic_sde

use ndarray::Array1;
use neural_sde_trading::model::NeuralSDE;
use neural_sde_trading::solvers;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║        Neural SDE Trading - Basic Example               ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    // ─── 1. Create a Neural SDE model ───
    println!("\n1. Creating Neural SDE model...");

    let state_dim = 2;
    let hidden_dim = 32;
    let num_layers = 3;

    let model = NeuralSDE::new(state_dim, hidden_dim, num_layers);
    println!("   State dim: {}, Hidden dim: {}, Layers: {}", state_dim, hidden_dim, num_layers);

    // ─── 2. Generate sample paths ───
    println!("\n2. Generating sample paths...");

    let x0 = Array1::from_vec(vec![0.0, 0.0]);
    let ts: Vec<f64> = (0..=50).map(|i| i as f64 / 50.0).collect();
    let dt = 0.01;
    let num_paths = 100;

    let start = std::time::Instant::now();
    let paths = model.sample_paths_parallel(&x0, &ts, dt, num_paths);
    let elapsed = start.elapsed();

    println!("   Generated {} paths in {:.2}ms", num_paths, elapsed.as_millis());

    // ─── 3. Path statistics ───
    println!("\n3. Path statistics at final time T=1.0:");

    for dim in 0..state_dim {
        let finals: Vec<f64> = paths.iter().map(|p| p.last().unwrap()[dim]).collect();
        let mean = finals.iter().sum::<f64>() / finals.len() as f64;
        let std = {
            let var = finals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (finals.len() - 1) as f64;
            var.sqrt()
        };

        println!("   Dim {}: mean = {:.4}, std = {:.4}", dim, mean, std);
    }

    // ─── 4. Compare drift and diffusion at different states ───
    println!("\n4. Drift and diffusion at various states (t=0.5):");
    println!("   {:>15} {:>12} {:>12} {:>12} {:>12}", "State", "Drift[0]", "Drift[1]", "Diff[0]", "Diff[1]");
    println!("   {}", "-".repeat(65));

    for x_val in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        let state = Array1::from_vec(vec![x_val, 0.0]);
        let drift = model.f(&state, 0.5);
        let diff = model.g(&state, 0.5);

        println!(
            "   [{:>5.1}, 0.0] {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            x_val, drift[0], drift[1], diff[0], diff[1]
        );
    }

    // ─── 5. Compare SDE solvers ───
    println!("\n5. Comparing SDE solvers (100 paths each):");

    let solver_names = ["Euler-Maruyama", "Milstein", "SRK"];
    let solver_methods = [
        solvers::SolverMethod::EulerMaruyama,
        solvers::SolverMethod::Milstein,
        solvers::SolverMethod::StochasticRungeKutta,
    ];

    for (name, method) in solver_names.iter().zip(solver_methods.iter()) {
        let start = std::time::Instant::now();

        let mut final_means = Array1::zeros(state_dim);
        let mut final_vars = Array1::zeros(state_dim);

        for _ in 0..100 {
            let path = solvers::solve(
                |x, t| model.f(x, t),
                |x, t| model.g(x, t),
                &x0,
                &ts,
                dt,
                *method,
            );
            let final_state = path.last().unwrap();
            final_means = final_means + final_state;
        }
        final_means /= 100.0;

        let elapsed = start.elapsed();

        println!(
            "   {:>20}: mean=[{:.4}, {:.4}], time={:.1}ms",
            name, final_means[0], final_means[1], elapsed.as_millis()
        );
    }

    // ─── 6. GBM comparison ───
    println!("\n6. GBM baseline comparison:");

    let gbm_paths: Vec<Vec<f64>> = (0..num_paths)
        .map(|_| solvers::geometric_brownian_motion(0.05, 0.2, 100.0, &ts))
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

    println!("   GBM(mu=0.05, sigma=0.2, S0=100):");
    println!("   Mean final price: {:.2}", gbm_mean);
    println!("   Std final price:  {:.2}", gbm_std);

    // ─── 7. Prediction distribution ───
    println!("\n7. Prediction with uncertainty (Neural SDE):");

    let (means, lowers, uppers) = model.predict_distribution(&x0, &ts, dt, 500, 0.95);

    // Show at a few time points
    println!("   {:>6} {:>12} {:>12} {:>12} {:>12}", "Time", "Mean[0]", "Lower", "Upper", "Width");
    println!("   {}", "-".repeat(60));

    for &t_idx in &[0, 10, 25, 40, 50] {
        if t_idx < means.len() {
            println!(
                "   {:>6.2} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
                ts[t_idx],
                means[t_idx][0],
                lowers[t_idx][0],
                uppers[t_idx][0],
                uppers[t_idx][0] - lowers[t_idx][0],
            );
        }
    }

    println!("\nExample complete!");
}
