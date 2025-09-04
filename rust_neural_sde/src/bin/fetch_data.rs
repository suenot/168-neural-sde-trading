//! Fetch market data from Bybit for Neural SDE training.
//!
//! Usage:
//!   cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --days 90

use anyhow::Result;
use clap::Parser;
use neural_sde_trading::api::BybitClient;

#[derive(Parser, Debug)]
#[command(name = "fetch_data", about = "Fetch market data from Bybit")]
struct Args {
    /// Trading pair symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of days of historical data
    #[arg(short, long, default_value_t = 90)]
    days: u32,

    /// Output CSV file path
    #[arg(short, long, default_value = "data.csv")]
    output: String,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║           Neural SDE Trading - Data Fetcher             ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Symbol:   {:>10}                                    ║", args.symbol);
    println!("║ Interval: {:>10}                                    ║", args.interval);
    println!("║ Days:     {:>10}                                    ║", args.days);
    println!("║ Output:   {:>10}                                    ║", args.output);
    println!("╚══════════════════════════════════════════════════════════╝");

    let client = BybitClient::new();

    println!("\nFetching data from Bybit...");

    match client.fetch_extended(&args.symbol, &args.interval, args.days) {
        Ok(klines) => {
            println!("Fetched {} candles", klines.len());

            if let (Some(first), Some(last)) = (klines.first(), klines.last()) {
                let first_dt = chrono::DateTime::from_timestamp_millis(first.timestamp)
                    .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
                    .unwrap_or_else(|| "N/A".to_string());
                let last_dt = chrono::DateTime::from_timestamp_millis(last.timestamp)
                    .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
                    .unwrap_or_else(|| "N/A".to_string());

                println!("Period: {} to {}", first_dt, last_dt);

                // Price statistics
                let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
                let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let avg_price = closes.iter().sum::<f64>() / closes.len() as f64;

                println!("\nPrice Statistics:");
                println!("  Min:  {:.2}", min_price);
                println!("  Max:  {:.2}", max_price);
                println!("  Avg:  {:.2}", avg_price);

                // Volatility
                let returns = neural_sde_trading::utils::log_returns(&closes);
                if !returns.is_empty() {
                    let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
                    let var: f64 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
                        / (returns.len() - 1) as f64;
                    let hourly_vol = var.sqrt();

                    println!("  Hourly Vol: {:.6}", hourly_vol);
                    println!(
                        "  Annualized Vol: {:.2}%",
                        hourly_vol * (252.0 * 24.0_f64).sqrt() * 100.0
                    );
                }
            }

            // Save to CSV
            println!("\nSaving to {}...", args.output);
            client.save_to_csv(&klines, &args.output)?;
            println!("Data saved successfully!");
        }
        Err(e) => {
            eprintln!("Failed to fetch data: {}", e);
            eprintln!("\nGenerating synthetic data instead...");

            // Generate synthetic data
            let ts: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
            let prices = neural_sde_trading::solvers::geometric_brownian_motion(
                0.1,  // drift
                0.3,  // volatility
                100.0, // initial price
                &ts,
            );

            println!("Generated {} synthetic price points", prices.len());
            println!(
                "Price range: {:.2} - {:.2}",
                prices.iter().cloned().fold(f64::INFINITY, f64::min),
                prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            );

            // Save synthetic data
            let mut writer = csv::Writer::from_path(&args.output)?;
            writer.write_record(&[
                "timestamp", "open", "high", "low", "close", "volume", "turnover",
            ])?;

            for (i, price) in prices.iter().enumerate() {
                let noise = 0.01 * price;
                writer.write_record(&[
                    (i as i64 * 3600000).to_string(), // Hourly timestamps
                    format!("{:.4}", price - noise * 0.5),
                    format!("{:.4}", price + noise),
                    format!("{:.4}", price - noise),
                    format!("{:.4}", price),
                    "1000.0".to_string(),
                    format!("{:.4}", price * 1000.0),
                ])?;
            }
            writer.flush()?;
            println!("Synthetic data saved to {}", args.output);
        }
    }

    Ok(())
}
