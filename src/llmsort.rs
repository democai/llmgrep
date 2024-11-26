use anyhow::Result;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::parameters::FormatType;
use ollama_rs::Ollama;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

const MAX_FILE_SIZE: u64 = 1024 * 1024; // 1MB
const BATCH_SIZE: usize = 100;
const MIN_BINARY_CHECK_SIZE: usize = 1000;
const BINARY_THRESHOLD: usize = 300; // 30% of MIN_BINARY_CHECK_SIZE
const MAX_SORT_TRY_COUNT: usize = 3;

#[derive(Debug, Deserialize)]
struct FileScores {
    filenames: Vec<FileScore>,
}

#[derive(Debug, Deserialize)]
struct FileScore {
    filename: String,
    score: f32,
}

pub struct LlmSort {
    ollama: Ollama,
    model: String,
    verbose: bool,
}

impl LlmSort {
    pub async fn new(model: &str, verbose: bool) -> Result<Self> {
        let ollama = Ollama::default();
        Ok(LlmSort {
            ollama,
            model: model.to_string(),
            verbose,
        })
    }

    async fn analyze_filenames_batch(
        &self,
        files: &[(PathBuf, String)],
        query: &str,
    ) -> Result<Vec<f32>> {
        let filenames: Vec<_> = files.iter().map(|(_, name)| name).collect();

        let system_prompt = "You are a highly accurate filename analysis tool. Your task is to analyze filenames and estimate the probability they contain content matching a search query.

Instructions:
1. Evaluate each filename considering:
   - Naming conventions and semantics
   - File extensions and their typical content
   - Common code/documentation patterns
   - Word matches and related concepts
2. Assign a score from 0.0 (irrelevant) to 1.0 (highly relevant)
3. Return ONLY a valid JSON array of objects with 'filename' and 'score' fields, nothing else

Example:
Input: ['main.rs', 'auth.rs'] with query 'authentication'
Output: [{\"filename\":\"main.rs\",\"score\":0.3},{\"filename\":\"auth.rs\",\"score\":0.9}]";

        let prompt = format!(
            "Analyze these filenames: {:#?}\nQuery: '{}'\n\nRespond with ONLY a JSON array. Example format: [{{\"filename\":\"example.rs\",\"score\":0.5}}]",
            filenames, query
        );

        let mut request = GenerationRequest::new(self.model.clone(), prompt);
        request.system = Some(system_prompt.to_string());
        request.format = Some(FormatType::Json);

        let response = self.ollama.generate(request).await?;

        let scores: Vec<FileScore> = match serde_json::from_str::<FileScores>(&response.response) {
            Ok(scores) => scores.filenames,
            Err(e) => {
                if self.verbose {
                    eprintln!(
                        "JSON parsing error: {}. Response was: {}",
                        e,
                        response.response.trim()
                    );
                }
                // Try parsing again with a different format
                match serde_json::from_str::<Vec<FileScore>>(&response.response) {
                    Ok(scores) => scores,
                    Err(e2) => {
                        if self.verbose {
                            eprintln!(
                                "Second JSON parsing error: {}. Response was: {}",
                                e2, response.response
                            );
                        }
                        Vec::new()
                    }
                }
            }
        };

        // Match scores back to original filenames, defaulting to 0.0 for any missing scores
        let result = files
            .iter()
            .map(|(_, name)| {
                scores
                    .iter()
                    .find(|score| score.filename == *name)
                    .map_or_else(|| 0.0, |score| score.score)
            })
            .collect();

        Ok(result)
    }

    pub async fn collect_and_sort_candidates(
        &self,
        dir: &Path,
        ignore_paths: &[&str],
        query: &str,
    ) -> Result<Vec<(PathBuf, f32)>> {
        let candidates = self.collect_candidates(dir, ignore_paths).await?;

        // Process candidates in batches
        let mut scored_candidates = Vec::new();
        for chunk in candidates.chunks(BATCH_SIZE) {
            let batch: Vec<(PathBuf, String)> = chunk
                .iter()
                .map(|path| {
                    let filename = path
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_default();
                    (path.clone(), filename)
                })
                .collect();

            let scores = self.analyze_filenames_batch(&batch, query).await?;

            scored_candidates.extend(chunk.iter().cloned().zip(scores));
        }

        scored_candidates
            .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored_candidates)
    }

    pub async fn collect_sort_with_retry(
        &self,
        dir: &Path,
        ignore_paths: &[&str],
        query: &str,
    ) -> Result<Vec<(PathBuf, f32)>> {
        let mut try_count = 0;
        while try_count < MAX_SORT_TRY_COUNT {
            let candidates = self
                .collect_and_sort_candidates(dir, ignore_paths, query)
                .await?;

            if candidates.is_empty() {
                return Ok(Vec::new());
            }

            if candidates.iter().any(|(_, score)| *score > 0.0) {
                return Ok(candidates);
            }
            try_count += 1;
        }

        // If we get here, we failed to find any promising candidates
        Ok(Vec::new())
    }

    fn is_binary_file(content: &[u8]) -> bool {
        let check_size = content.len().min(MIN_BINARY_CHECK_SIZE);
        let non_ascii_count = content
            .iter()
            .take(check_size)
            .filter(|&&byte| byte == 0 || byte >= 128)
            .count();

        non_ascii_count > BINARY_THRESHOLD
    }

    fn should_ignore(&self, path: &Path, root: &Path, ignore_paths: &[&str]) -> bool {
        // Get path relative to root
        let rel_path = path.strip_prefix(root).unwrap_or(path);

        // Check if any component of the path matches ignore patterns
        for ignore in ignore_paths {
            let ignore_path = Path::new(ignore);

            // Check if the relative path starts with the ignore pattern
            if rel_path.starts_with(ignore_path) {
                return true;
            }
        }
        false
    }

    async fn collect_candidates(&self, dir: &Path, ignore_paths: &[&str]) -> Result<Vec<PathBuf>> {
        let mut candidates = Vec::new();

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            // Skip if path matches ignore patterns
            if self.should_ignore(&path, dir, ignore_paths) {
                continue;
            }

            if path.is_dir() {
                let mut sub_candidates =
                    Box::pin(self.collect_candidates(&path, ignore_paths)).await?;
                candidates.append(&mut sub_candidates);
                continue;
            }

            // Get metadata once and reuse
            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue, // Skip if we can't read metadata
            };

            // Skip files that are too large
            if metadata.len() > MAX_FILE_SIZE {
                continue;
            }

            // Read file content for binary check
            match fs::read(&path) {
                Ok(content) => {
                    // Skip binary files
                    if Self::is_binary_file(&content) {
                        continue;
                    }

                    // Skip if not valid UTF-8
                    if String::from_utf8(content.clone()).is_err() {
                        continue;
                    }

                    candidates.push(path);
                }
                Err(_) => continue, // Skip if we can't read the file
            }
        }

        Ok(candidates)
    }
}
