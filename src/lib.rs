//! Rosetta AISP LLM Fallback
//!
//! Provides LLM-powered fallback for AISP conversion when deterministic
//! Rosetta mappings have low confidence. Uses Claude SDK for intelligent
//! prose-to-symbol translation.
//!
//! # Example
//!
//! ```no_run
//! use rosetta_aisp_llm::{convert_with_fallback, ConversionOptionsExt};
//!
//! # async fn example() {
//! let prose = "Define a type User with valid credentials";
//! let result = convert_with_fallback(prose, None).await;
//! println!("Output: {}", result.output);
//! println!("Used LLM: {}", result.used_fallback);
//! # }
//! ```

mod claude;
mod provider;

pub use claude::ClaudeFallback;
pub use provider::{LlmProvider, LlmResult};

// Re-export rosetta-aisp types for convenience
pub use rosetta_aisp::{
    AispConverter, ConversionOptions, ConversionResult, ConversionTier, RosettaStone, TokenStats,
};

/// Extended conversion options with LLM fallback support
#[derive(Debug, Clone, Default)]
pub struct ConversionOptionsExt {
    /// Force specific tier (auto-detect if None)
    pub tier: Option<ConversionTier>,
    /// Confidence threshold for LLM fallback (default: 0.8)
    pub confidence_threshold: Option<f64>,
    /// Enable LLM fallback
    pub enable_llm_fallback: bool,
    /// LLM model to use (default: sonnet)
    pub llm_model: Option<String>,
}

/// Convert prose to AISP with optional LLM fallback
///
/// This function first attempts deterministic conversion using rosetta-aisp.
/// If the confidence is below the threshold and LLM fallback is enabled,
/// it uses Claude to improve the conversion.
///
/// # Arguments
///
/// * `prose` - The natural language text to convert
/// * `options` - Optional configuration for conversion behavior
///
/// # Returns
///
/// A `ConversionResult` containing the AISP output and metadata
pub async fn convert_with_fallback(
    prose: &str,
    options: Option<ConversionOptionsExt>,
) -> ConversionResult {
    let opts = options.unwrap_or_default();

    // Convert using rosetta-aisp's ConversionOptions
    let base_options = ConversionOptions {
        tier: opts.tier,
        confidence_threshold: opts.confidence_threshold,
    };

    let result = AispConverter::convert(prose, Some(base_options));
    let threshold = opts.confidence_threshold.unwrap_or(0.8);

    // Check if LLM fallback is needed
    if opts.enable_llm_fallback && result.confidence < threshold {
        let provider = if let Some(model) = &opts.llm_model {
            ClaudeFallback::with_model(model)
        } else {
            ClaudeFallback::new()
        };

        if provider.is_available().await {
            if let Ok(llm_result) = provider
                .convert(prose, result.tier, &result.unmapped, Some(&result.output))
                .await
            {
                return llm_result.to_conversion_result(result.tier, prose.len());
            }
        }
    }

    result
}
