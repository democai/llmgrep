# LLM Grep

LLM Grep is a semantic search tool that uses Ollama's Large Language Models to find files containing information related to your search query. Unlike traditional grep that matches exact text patterns, LLM Grep tries to understand the meaning of the query and returns files that are semantically related.

## Features

- Two-phase semantic search:
  - Phase 1: Smart filename analysis and scoring
  - Phase 2: In-depth content analysis of promising files
- Recursive directory traversal
- Efficient file handling:
  - Skips files larger than 1MB
  - Binary file detection using null byte heuristics
  - Processes large text files in 2000-character chunks
- Natural language queries instead of regex patterns
- Detailed relevance explanations for each match
- Built with Rust and Ollama for performance and reliability

## Prerequisites

- [Rust](https://rustup.rs/) (latest stable version)
- [Ollama](https://ollama.ai/) installed and running locally
- The [Dolphin Mistral](hhttps://ollama.com/library/dolphin-mistral) model pulled in Ollama (`ollama pull dolphin-mistral:latest`)

## Installation

1. Clone the repository

```sh
git clone https://github.com/democai/llmgrep.git
```

2. Navigate to the project directory:

```sh
cd llmgrep
```

3. Build the project:

```sh
cargo build --release
```

4. Run the binary:

```sh
cargo run --release -- "<search query>"
```

## Example

```sh
cargo run --release -- "find me the file with the string 'hello'"
```

## Usage

Usage: `llmgrep` [OPTIONS] `<QUERY>` [DIRECTORY]  
Arguments:  
  `<QUERY>`      Search query - what to look for semantically  
  `[DIRECTORY]`  Directory to search in (default: `.`)  

Options:  
  `--model <MODEL>`                LLM model to use (default: `dolphin-mistral:latest`)  
  `--ignore-paths <IGNORE_PATHS>`  Paths to ignore during search (comma separated) (default: `.git,.gitignore,.vscode,.idea,.vscode-test,target,dist,.gradle,dep,node_modules,package-lock.json,Cargo.lock`)  
  `-v, --verbose`                  Enable verbose output  
  `-h, --help`                     Print help  
  `-V, --version`                  Print version  
