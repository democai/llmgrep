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
- The [Dolphin Mistral](https://ollama.com/models/dolphin-mistral) model pulled in Ollama (`ollama pull dolphin-mistral:latest`)

## Installation

1. Clone the repository

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
cargo run --release -- <directory> "<search query>"
```

## Example

```sh
cargo run --release -- . "find me the file with the string 'hello'"
```
