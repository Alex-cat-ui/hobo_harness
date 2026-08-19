//! Does native tool calling actually work on these models, on this Ollama?
//! Asked of the machine rather than of the documentation.

use anyhow::Result;
use minions_core::chat::{tool_schemas, to_tool_call, Message};
use minions_core::ollama::{Ollama, Options};

#[tokio::main]
async fn main() -> Result<()> {
    let o = Ollama::local()?;
    if !o.reachable().await {
        anyhow::bail!("ollama unreachable");
    }

    for model in ["qwen2.5-coder:14b", "qwen2.5:14b", "qwen2.5:7b"] {
        println!("\n=== {model} ===");
        let messages = vec![
            Message::system(
                "You change files in a project. Use the tools. Read a file before changing it.",
            ),
            Message::user("Read the file src/durations.py so you can see what is in it."),
        ];
        let opts = Options { num_ctx: 8192, num_predict: 300, temperature: 0.1, repeat_penalty: 1.1, seed: Some(3) };

        let reply = o.chat(model, &messages, tool_schemas(), &opts, "120s", |_| {}).await?;

        println!("  text     : {:?}", reply.text.chars().take(90).collect::<String>());
        println!("  calls    : {}", reply.tool_calls.len());
        for c in &reply.tool_calls {
            print!("    {} {}", c.name, c.arguments);
            match to_tool_call(c) {
                Ok(mapped) => println!("  -> {mapped:?}"),
                Err(e) => println!("  -> UNMAPPABLE: {e}"),
            }
        }
        if reply.tool_calls.is_empty() {
            println!("    (no native call — the text fallback would be needed here)");
        }
    }
    Ok(())
}
