//! Measures what SDD assumption 6 guesses: how much memory a model actually
//! costs at a given context window, and whether that cost is linear in the
//! window. The memory budget in SDD §8 depends on the answer.

use anyhow::{Context, Result};
use minions_core::memory::{MemoryProbe, VmStatProbe};
use minions_core::ollama::{Ollama, Options};
use std::time::{Duration, Instant};

const WINDOWS: [u32; 4] = [4096, 8192, 16384, 32768];
const DEFAULT_MODELS: [&str; 2] = ["qwen2.5-coder:14b", "qwen2.5:14b"];

fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

async fn settle(o: &Ollama) -> Result<()> {
    for m in DEFAULT_MODELS {
        let _ = o.unload(m).await;
    }
    let _ = o.unload("qwen2.5:7b").await;
    for _ in 0..40 {
        if o.loaded().await?.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    tokio::time::sleep(Duration::from_millis(800)).await;
    Ok(())
}

#[derive(Debug)]
struct Point {
    window: u32,
    reported: u64,
    system_delta: i64,
    load_secs: f64,
    tok_per_s: f64,
}

/// Least squares over (window, reported size) -> (bytes per token, weights).
fn fit(points: &[Point]) -> (f64, f64) {
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.window as f64).sum();
    let sy: f64 = points.iter().map(|p| p.reported as f64).sum();
    let sxx: f64 = points.iter().map(|p| (p.window as f64).powi(2)).sum();
    let sxy: f64 = points.iter().map(|p| p.window as f64 * p.reported as f64).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < f64::EPSILON {
        return (0.0, sy / n);
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    (slope, intercept)
}

#[tokio::main]
async fn main() -> Result<()> {
    let probe = VmStatProbe;
    let o = Ollama::local()?;

    if !o.reachable().await {
        anyhow::bail!("ollama is not reachable at 127.0.0.1:11434 — start it first");
    }

    let installed = o.list().await.context("listing models")?;
    println!("installed models: {}", installed.len());
    for m in &installed {
        println!("  {:<26} {:>7.2} GiB", m.name, gib(m.size));
    }

    let base = probe.snapshot()?;
    println!("\nbaseline available: {:.2} GiB  (page size {})", base.available_gib(), base.page_size);

    let args: Vec<String> = std::env::args().skip(1).collect();
    let models: Vec<String> = if args.is_empty() {
        DEFAULT_MODELS.iter().map(|s| s.to_string()).collect()
    } else {
        args
    };

    for model in models.iter().map(|s| s.as_str()) {
        if !installed.iter().any(|m| m.name == model) {
            println!("\n{model}: not installed, skipping");
            continue;
        }
        println!("\n=== {model} ===");
        println!("{:>8}  {:>12}  {:>14}  {:>10}  {:>10}", "window", "reported", "system delta", "load s", "tok/s");

        let mut points: Vec<Point> = Vec::new();

        for window in WINDOWS {
            settle(&o).await?;
            let before = probe.snapshot()?;

            let opts = Options { num_ctx: window, num_predict: 24, temperature: 0.0, repeat_penalty: 1.1, seed: Some(7) };
            let t0 = Instant::now();
            let mut first_token: Option<Instant> = None;
            let completion = o
                .generate(model, "Reply with the single word: ready.", &opts, "120s", |_t| {
                    if first_token.is_none() {
                        first_token = Some(Instant::now());
                    }
                })
                .await
                .with_context(|| format!("generating at window {window}"))?;

            let load_secs = first_token.map(|f| (f - t0).as_secs_f64()).unwrap_or_default();
            let gen_secs = first_token.map(|f| f.elapsed().as_secs_f64()).unwrap_or(1.0).max(0.001);
            let tok_per_s = completion.eval_tokens as f64 / gen_secs;

            tokio::time::sleep(Duration::from_millis(600)).await;
            let after = probe.snapshot()?;
            let loaded = o.loaded().await?;
            let reported = loaded.iter().find(|m| m.name == model).map(|m| m.size).unwrap_or(0);
            let system_delta = before.available_bytes() as i64 - after.available_bytes() as i64;

            println!(
                "{:>8}  {:>9.2} GiB  {:>11.2} GiB  {:>10.1}  {:>10.1}",
                window, gib(reported), system_delta as f64 / 1024.0 / 1024.0 / 1024.0, load_secs, tok_per_s
            );

            points.push(Point { window, reported, system_delta, load_secs, tok_per_s });
        }

        let usable: Vec<&Point> = points.iter().filter(|p| p.reported > 0).collect();
        if usable.len() >= 2 {
            let owned: Vec<Point> = usable
                .iter()
                .map(|p| Point { window: p.window, reported: p.reported, system_delta: p.system_delta, load_secs: p.load_secs, tok_per_s: p.tok_per_s })
                .collect();
            let (per_token, weights) = fit(&owned);
            println!("\n  fit: weights {:.2} GiB + {:.1} KiB per context token", gib(weights as u64), per_token / 1024.0);
            println!("  kv at 16384: {:.2} GiB", per_token * 16384.0 / 1024.0 / 1024.0 / 1024.0);
            println!("  two such models at 16384: {:.2} GiB", 2.0 * (weights + per_token * 16384.0) / 1024.0 / 1024.0 / 1024.0);
        } else {
            println!("\n  /api/ps reported no size — fit skipped");
        }
    }

    settle(&o).await?;
    let end = probe.snapshot()?;
    println!("\nfinal available: {:.2} GiB", end.available_gib());
    Ok(())
}
