use crate::llmsort::LlmSort;
use anyhow::Result;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::parameters::FormatType;
use ollama_rs::Ollama;
use serde::Deserialize;
use std::fs;
use std::path::Path;

const CHUNK_SIZE: usize = 2000; // Characters per chunk

#[derive(Debug, Deserialize)]
struct AnalysisResponse {
    has_match: bool,
    analysis: Option<String>,
}

pub struct LlmGrep {
    ollama: Ollama,
    model: String,
    sorter: LlmSort,
}

impl LlmGrep {
    pub async fn new(model: &str) -> Result<Self> {
        let ollama = Ollama::default();
        let sorter = LlmSort::new(model).await?;
        Ok(LlmGrep {
            ollama,
            model: model.to_string(),
            sorter,
        })
    }

    async fn analyze_content(
        &self,
        path: &Path,
        content: &str,
        query: &str,
    ) -> Result<Option<String>> {
        let system_prompt = "You are a highly accurate semantic search function. Your task is to analyze text and determine if it contains information semantically related to a search query.

Instructions:
1. Carefully analyze the semantic meaning and context, not just exact keyword matches
2. Consider related concepts, implications, and domain-specific terminology
3. If relevant information is found, explain the relationship briefly and precisely
4. If no relevant information exists, set has_match to false

Respond with a JSON object matching this structure:
{
    \"has_match\": boolean,
    \"analysis\": string | null
}

Example input: Text about 'database indexing' with query 'performance optimization'
Example response: {\"has_match\": true, \"analysis\": \"Discusses B-tree indexes which improve query performance by reducing disk I/O\"}

Remember: Be concise, objective, and focus on semantic relevance rather than surface-level matches.";

        let prompt = format!(
            "
            Filename: {}
            Text: 
            {}\n
            Does the user query '{}' relate to the above text? \
            Respond with a JSON object containing has_match and analysis fields.",
            path.display(),
            content,
            query
        );
        let mut request = GenerationRequest::new(self.model.clone(), prompt);
        request.system = Some(system_prompt.to_string());
        request.format = Some(FormatType::Json);

        let response = self.ollama.generate(request).await?;

        let analysis: AnalysisResponse = serde_json::from_str(&response.response)?;

        if !analysis.has_match {
            Ok(None)
        } else {
            Ok(analysis.analysis)
        }
    }

    pub async fn search_directory(
        &self,
        dir: &Path,
        ignore_paths: &[&str],
        query: &str,
    ) -> Result<()> {
        println!("First pass: Recursively collecting and scoring all files...");

        let mut try_count = 0;
        let mut candidates = Vec::new();
        // Pass ignore_paths to collect_and_sort_candidates
        while try_count < 3 {
            candidates = self
                .sorter
                .collect_and_sort_candidates(dir, ignore_paths, query)
                .await?;

            if candidates.is_empty() {
                println!("No candidates found. Exiting...");
                return Ok(());
            }

            if candidates.iter().any(|(_, score)| *score > 0.0) {
                break;
            }
            try_count += 1;
        }
        println!(
            "Sorted candidates: \n{}",
            candidates
                .iter()
                .map(|(path, score)| format!("{} (score: {:.2})", path.display(), score))
                .collect::<Vec<String>>()
                .join("\n")
        );

        println!("\nSecond pass: analyzing content of promising candidates...");

        // Second pass: analyze content of promising candidates
        for (path, score) in candidates {
            let content = match fs::read(&path) {
                Ok(content) => content,
                Err(_) => continue,
            };

            println!(
                "Analyzing content of {} (filename score: {:.2})",
                path.display(),
                score
            );

            // Convert to string (we know it's valid UTF-8 from pre-filtering)
            let content_str = String::from_utf8_lossy(&content);

            // Process file in chunks if necessary
            for chunk in content_str
                .chars()
                .collect::<Vec<char>>()
                .chunks(CHUNK_SIZE)
            {
                let chunk_str: String = chunk.iter().collect();
                if let Ok(Some(relevance)) = self.analyze_content(&path, &chunk_str, query).await {
                    println!("{}: {}", path.display(), relevance);
                    break; // Stop processing chunks once we find a match
                }
            }
        }

        Ok(())
    }
}
