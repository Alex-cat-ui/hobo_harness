//! Undoes a run: restores exactly the paths its journal names, from the
//! checkpoint it took. Files the run never touched are not considered.

use anyhow::{Context, Result};
use minions_core::checkpoint::{rollback, Checkpoint, Restored};
use minions_core::journal::Journal;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let run_dir = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: mrollback <run-dir> [project-root]");
        std::process::exit(2)
    }));
    let project = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("../../..").canonicalize().unwrap_or_else(|_| PathBuf::from(".")));

    let state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("run.json")).context("reading run.json")?)?;
    let commit = state["checkpoint"]
        .as_str()
        .context("this run recorded no checkpoint, so there is nothing to roll back to")?
        .to_string();
    let run_id = state["run_id"].as_str().unwrap_or("unknown").to_string();

    let entries = Journal::load(&run_dir.join("journal.jsonl")).context("reading the journal")?;
    let mutated = Journal::mutated_paths(&entries);

    println!("project  : {}", project.display());
    println!("run      : {run_id}");
    println!("checkpoint: {}", &commit[..12.min(commit.len())]);
    println!("paths the run touched: {}", mutated.len());
    for (p, existed) in &mutated {
        println!("  {} {}", if *existed { "restore" } else { "remove " }, p.display());
    }
    if mutated.is_empty() {
        println!("\nnothing to undo");
        return Ok(());
    }

    let cp = Checkpoint { commit, reference: format!("refs/minions/checkpoints/{run_id}") };
    let done = rollback(&project, &cp, &mutated)?;

    println!("\nresult:");
    for d in &done {
        match d {
            Restored::Content(p) => println!("  restored  {}", p.display()),
            Restored::Removed(p) => println!("  removed   {}", p.display()),
            Restored::Unrecoverable(p) => println!("  COULD NOT {} (absent from the checkpoint)", p.display()),
        }
    }
    Ok(())
}
