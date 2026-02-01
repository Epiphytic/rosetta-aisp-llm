# rosetta-aisp-llm

LLM-powered fallback for AISP conversion using Claude SDK.

Extends [rosetta-aisp](https://github.com/epiphytic/rosetta-aisp) with AI-powered conversion when deterministic Rosetta mappings have low confidence.

## Features

- **Hybrid Conversion**: Combines deterministic Rosetta mappings with LLM fallback
- **Confidence-Based Triggering**: Only uses LLM when confidence is below threshold
- **Multiple Model Support**: Choose between haiku, sonnet, or opus based on complexity
- **Async API**: Built with async/await for efficient I/O

## Installation

```toml
[dependencies]
rosetta-aisp-llm = "0.1"
```

## Usage

```rust
use rosetta_aisp_llm::{convert_with_fallback, ConversionOptionsExt};

#[tokio::main]
async fn main() {
    let prose = "Define a type User with valid credentials";

    // Simple conversion (no LLM fallback)
    let result = convert_with_fallback(prose, None).await;

    // With LLM fallback enabled
    let options = ConversionOptionsExt {
        enable_llm_fallback: true,
        confidence_threshold: Some(0.8),
        llm_model: Some("sonnet".to_string()),
        ..Default::default()
    };
    let result = convert_with_fallback(prose, Some(options)).await;

    println!("Output: {}", result.output);
    println!("Confidence: {}", result.confidence);
    println!("Used LLM: {}", result.used_fallback);
}
```

## Custom LLM Provider

Implement the `LlmProvider` trait to add support for other LLM providers:

```rust
use rosetta_aisp_llm::{LlmProvider, LlmResult, ConversionTier};
use async_trait::async_trait;
use anyhow::Result;

struct MyProvider;

#[async_trait]
impl LlmProvider for MyProvider {
    async fn convert(
        &self,
        prose: &str,
        tier: ConversionTier,
        unmapped: &[String],
        partial_output: Option<&str>,
    ) -> Result<LlmResult> {
        // Your implementation here
        todo!()
    }

    async fn is_available(&self) -> bool {
        true
    }
}
```

## CLI Tool

This crate includes the `rosetta` CLI for command-line conversions.

### Installation

```bash
cargo install rosetta-aisp-llm
```

Or build from source:

```bash
cargo build --release
# Binary will be at target/release/rosetta
```

### Commands

```bash
# Convert prose to AISP notation
rosetta convert -i "for all x in S, x equals y"

# Convert with LLM fallback enabled
rosetta convert -i "The quantum entanglement manifests correlation" --llm-fallback

# Force a specific tier (minimal, standard, full)
rosetta convert -i "Define x as 5" -t minimal

# Output as JSON
rosetta convert -i "for all users, allow access" -f json

# Convert AISP back to prose
rosetta to-prose -i "∀x∈S: x≡y"

# Detect appropriate conversion tier
rosetta detect-tier -i "Define a type User and prove validity"

# Look up a symbol for prose pattern
rosetta lookup "for all"

# Look up prose for a symbol
rosetta reverse "∀"

# List all symbols (optionally by category)
rosetta symbols
rosetta symbols -c quantifier

# List all categories
rosetta categories

# Test round-trip semantic preservation
rosetta round-trip -i "Define x as 5 and for all y in S, x equals y" -r 5
```

### Piping Input

All commands that accept `-i/--input` can also read from stdin:

```bash
echo "for all x, x implies y" | rosetta convert
cat spec.txt | rosetta convert --llm-fallback
```

### LLM Fallback Options

```bash
# Enable LLM fallback with custom threshold
rosetta convert -i "complex domain text" --llm-fallback --threshold 0.7

# Use different Claude models (haiku, sonnet, opus)
rosetta convert -i "text" --llm-fallback --model haiku
```

## Requirements

- Rust 1.85+
- Claude Code CLI (for `ClaudeFallback` provider and `--llm-fallback` flag)

## License

MIT
