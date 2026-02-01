//! LLM Provider Trait
//!
//! Defines the interface for LLM-based AISP conversion providers.

use anyhow::Result;
use async_trait::async_trait;
use rosetta_aisp::{ConversionResult, ConversionTier, TokenStats};

/// LLM provider trait for fallback conversions
///
/// Implement this trait to add support for different LLM providers.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Convert prose to AISP using LLM
    ///
    /// # Arguments
    ///
    /// * `prose` - The natural language text to convert
    /// * `tier` - The target conversion tier
    /// * `unmapped` - Phrases that couldn't be mapped deterministically
    /// * `partial_output` - Optional partial conversion from deterministic pass
    /// * `use_aisp_prompt` - Use minimalist AISP prompt instead of English
    async fn convert(
        &self,
        prose: &str,
        tier: ConversionTier,
        unmapped: &[String],
        partial_output: Option<&str>,
        use_aisp_prompt: bool,
    ) -> Result<LlmResult>;

    /// Check if provider is available
    async fn is_available(&self) -> bool;
}

/// LLM conversion result
#[derive(Debug, Clone)]
pub struct LlmResult {
    /// The converted AISP output
    pub output: String,
    /// The provider name (e.g., "claude")
    pub provider: String,
    /// The model used (e.g., "sonnet")
    pub model: String,
    /// Approximate tokens used (if available)
    pub tokens_used: Option<usize>,
}

impl LlmResult {
    /// Convert to ConversionResult
    pub fn to_conversion_result(self, tier: ConversionTier, input_len: usize) -> ConversionResult {
        ConversionResult {
            output: self.output.clone(),
            confidence: 0.95, // LLM output assumed high confidence
            unmapped: vec![],
            tier,
            tokens: TokenStats {
                input: input_len,
                output: self.output.len(),
                ratio: if input_len == 0 {
                    0.0
                } else {
                    (self.output.len() as f64 / input_len as f64 * 100.0).round() / 100.0
                },
            },
            used_fallback: true,
        }
    }
}
