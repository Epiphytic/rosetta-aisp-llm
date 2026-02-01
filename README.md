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

## Requirements

- Rust 1.85+
- Claude Code CLI (for `ClaudeFallback` provider)

## License

MIT
