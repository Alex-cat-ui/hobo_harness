//! Measures every installed model and stores the profiles.

use anyhow::Result;
use minions_core::ollama::Ollama;
use minions_core::probe::measure;
use minions_core::profile::ProfileStore;
use minions_core::tokens::CharRatioTokenizer;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    let o = Ollama::local()?;
    if !o.reachable().await {
        anyhow::bail!("ollama unreachable — run tools/ollama.sh up");
    }

    let path = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join("Library/Application Support/MyITMinions/profiles.json");
    let mut store = ProfileStore::load(&path);
    let tok = CharRatioTokenizer::default();

    let wanted: Vec<String> = std::env::args().skip(1).collect();
    let installed = o.list().await?;
    let models: Vec<String> = installed
        .iter()
        .map(|m| m.name.clone())
        .filter(|n| !n.contains("embed"))
        .filter(|n| wanted.is_empty() || wanted.iter().any(|w| n.contains(w.as_str())))
        .collect();

    let now = String::from_utf8_lossy(
        &std::process::Command::new("date").args(["-u", "+%Y-%m-%dT%H:%M:%SZ"]).output()?.stdout,
    )
    .trim()
    .to_string();

    for model in &models {
        println!("\n=== {model} ===");
        let profile = measure(&o, model, &tok, &now, |s| println!("  {s}")).await?;
        println!(
            "  → channel {:?}, edits {:?}, parallel {}, format {}, max {} chars",
            profile.channel, profile.edit_style, profile.parallel_calls, profile.holds_format, profile.max_output_chars
        );
        store.put(profile);
    }

    store.save(&path)?;
    println!("\nprofiles written to {}", path.display());
    Ok(())
}
