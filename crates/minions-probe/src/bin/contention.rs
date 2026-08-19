//! Do two resident models actually work in parallel, or do they contend?
//!
//! Memory says a 14B pipeline model and the 7B chat companion fit together.
//! Memory is not the only shared resource: both run on the same neural engine.
//! This measures whether "chat alongside a running pipeline" is a real feature
//! or a tax on the work the user actually cares about.

use anyhow::Result;
use minions_core::ollama::{Ollama, Options};
use std::time::Instant;

const BIG: &str = "qwen2.5-coder:14b";
const SMALL: &str = "qwen2.5:7b";
const PROMPT_BIG: &str = "Write a short Swift function that reverses a string. Explain each line.";
const PROMPT_SMALL: &str = "List five colours, one per line, with a one-word description each.";

async fn timed(o: &Ollama, model: &str, prompt: &str, predict: i32) -> Result<(u32, f64)> {
    let opts = Options { num_ctx: 8192, num_predict: predict, temperature: 0.0, repeat_penalty: 1.1, seed: Some(11) };
    let t0 = Instant::now();
    let c = o.generate(model, prompt, &opts, "300s", |_| {}).await?;
    Ok((c.eval_tokens, c.eval_tokens as f64 / t0.elapsed().as_secs_f64().max(0.001)))
}

#[tokio::main]
async fn main() -> Result<()> {
    let o = Ollama::local()?;
    if !o.reachable().await {
        anyhow::bail!("ollama unreachable");
    }

    println!("warming both models...");
    let _ = timed(&o, BIG, "hi", 8).await?;
    let _ = timed(&o, SMALL, "hi", 8).await?;

    println!("\n--- solo ---");
    let (bt, brate) = timed(&o, BIG, PROMPT_BIG, 200).await?;
    println!("{BIG:<20} {bt:>4} tok  {brate:>6.1} tok/s");
    let (st, srate) = timed(&o, SMALL, PROMPT_SMALL, 200).await?;
    println!("{SMALL:<20} {st:>4} tok  {srate:>6.1} tok/s");

    println!("\n--- concurrent ---");
    let o2 = Ollama::local()?;
    let big = tokio::spawn(async move { timed(&o2, BIG, PROMPT_BIG, 200).await });
    let o3 = Ollama::local()?;
    let small = tokio::spawn(async move { timed(&o3, SMALL, PROMPT_SMALL, 200).await });

    let (b, s) = (big.await??, small.await??);
    println!("{BIG:<20} {:>4} tok  {:>6.1} tok/s   ({:+.0}%)", b.0, b.1, (b.1 / brate - 1.0) * 100.0);
    println!("{SMALL:<20} {:>4} tok  {:>6.1} tok/s   ({:+.0}%)", s.0, s.1, (s.1 / srate - 1.0) * 100.0);

    println!("\nresident after:");
    for m in o.loaded().await? {
        println!("  {:<22} {:>6.2} GiB", m.name, m.size as f64 / 1024.0 / 1024.0 / 1024.0);
    }
    Ok(())
}
