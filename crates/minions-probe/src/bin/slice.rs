//! Vertical slice: task in, parsed document on disk, tokens streamed live.
//!
//! This exercises the riskiest claim in the whole design — that a local 14B
//! model will hold a document format reliably enough for agents to hand work
//! to each other through files.

use anyhow::Result;
use minions_core::document::Artifact;
use minions_core::node::{run_agent_node, Role};
use minions_core::ollama::Ollama;
use minions_core::tokens::CharRatioTokenizer;
use std::io::Write;
use std::path::PathBuf;

const ANALYST: &str = "You are the analyst in a development pipeline. Your only job is to turn a vague \
task into verifiable requirements. You do NOT design the solution and you do NOT write code.

The document body must contain, in this order:
- Statement: what is required, one paragraph.
- Requirements: a numbered list of verifiable claims. For each it must be unambiguous whether it is met.
- Out of scope: what must NOT be done. At least three items.
- Areas touched: files and modules the change will reach.
- Unknowns: questions the statement does not answer. If none, write \"none\". Do NOT invent answers.

Rules:
- Do not propose an implementation.
- Do not widen the task. If tempted to add something while you are here, put it under Out of scope.
- If the statement is so unclear that requirements would be fabricated, say so under Unknowns.";

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let task = args
        .first()
        .cloned()
        .unwrap_or_else(|| "Make tunnel profile switching work without restarting the application.".to_string());

    let ollama = Ollama::local()?;
    if !ollama.reachable().await {
        anyhow::bail!("ollama unreachable at 127.0.0.1:11434");
    }

    let run_id = format!("{}_slice", chrono_stamp());
    let run_dir: PathBuf = PathBuf::from(".minions/runs").join(&run_id);

    let role = Role {
        name: "analyst",
        model: "qwen2.5:14b".to_string(),
        window: 8192,
        temperature: 0.5,
        artifact: Artifact::Requirements,
        system: ANALYST,
    };

    println!("run {run_id}");
    println!("task: {task}\n");
    println!("--- live stream ---");

    let tok = CharRatioTokenizer::default();
    let mut printed = 0usize;
    let outcome = run_agent_node(
        &ollama,
        &role,
        &run_id,
        &task,
        &["00_task.md".to_string()],
        &run_dir,
        "01_requirements",
        &tok,
        |t| {
            print!("{t}");
            let _ = std::io::stdout().flush();
            printed += t.len();
        },
        |e| println!("\n[{e}]"),
    )
    .await?;

    println!("\n--- result ---");
    println!("attempts used : {}", outcome.attempts_used);
    println!("written to    : {}", outcome.path.display());
    println!("streamed bytes: {printed}");
    println!("digest        : {}", outcome.document.header.digest.replace('\n', " "));
    println!("results       : {:?}", outcome.document.header.results);
    println!("body lines    : {}", outcome.document.body.lines().count());
    Ok(())
}

fn chrono_stamp() -> String {
    let out = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H-%M-%S"])
        .output()
        .expect("date");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}
