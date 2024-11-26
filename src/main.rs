mod llmgrep;
mod llmsort;
use anyhow::Result;
use clap::Parser;
use llmgrep::LlmGrep;
use std::path::PathBuf;

/// Semantic code search using local LLMs
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Search query - what to look for semantically
    query: String,

    /// Directory to search in
    #[arg(default_value = ".")]
    directory: PathBuf,

    /// LLM model to use (default: dolphin-mistral:latest)
    #[arg(long, default_value = "dolphin-mistral:latest")]
    model: String,

    /// Paths to ignore during search (comma separated)
    #[arg(
        long,
        value_delimiter = ',',
        default_value = ".git,.gitignore,.vscode,.idea,.vscode-test,target,dist,.gradle,dep,node_modules,package-lock.json,Cargo.lock"
    )]
    ignore_paths: Vec<String>,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if args.verbose {
        println!("Initializing LLM Grep with local Ollama model...");
    }
    let llm_grep = LlmGrep::new(&args.model, args.verbose).await?;

    let ignore_paths: Vec<&str> = args.ignore_paths.iter().map(|s| s.as_str()).collect();

    if args.verbose {
        println!("Searching for: {}", args.query);
    }
    llm_grep
        .search_directory(&args.directory, &ignore_paths, &args.query)
        .await?;

    Ok(())
}
