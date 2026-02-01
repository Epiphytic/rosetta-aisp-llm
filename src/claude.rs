//! Claude SDK Fallback
//!
//! Uses claude-agent-sdk-rs for LLM-based AISP conversion
//! when deterministic Rosetta mappings have low confidence.

use crate::provider::{LlmProvider, LlmResult};
use anyhow::Result;
use async_trait::async_trait;
use rosetta_aisp::{get_all_categories, symbol_to_prose, symbols_by_category, ConversionTier};

/// Generate symbol reference grouped by category
fn symbol_ref_grouped() -> String {
    let mut output = String::new();
    let categories = get_all_categories();

    for category in categories {
        output.push_str(&format!("\n### {}\n", category.to_uppercase()));
        let symbols = symbols_by_category(category);
        for symbol in symbols {
            if let Some(pattern) = symbol_to_prose(symbol) {
                output.push_str(&format!("- {}: {}\n", symbol, pattern));
            }
        }
    }
    output
}

/// System prompt for AISP conversion
fn system_prompt() -> String {
    let symbol_ref = symbol_ref_grouped();

    format!(
        r#"You are an AISP (AI Symbolic Programming) conversion specialist.

Convert natural language prose to AISP 5.1 symbolic notation using these rules:

## Symbol Reference (Rosetta Stone)
{symbol_ref}

## Output Format by Tier

### Minimal Tier
Direct symbol substitution only. Example:
Input: "Define x as 5"
Output: x≜5

### Standard Tier
Include header block with metadata:
```
𝔸5.1.[name]@[date]
γ≔[name]

⟦Λ:Funcs⟧{{
  [symbol conversion]
}}

⟦Ε⟧⟨δ≜0.70;τ≜◊⁺⟩
```

### Full Tier
Complete AISP document with all blocks:
```
𝔸5.1.[name]@[date]
γ≔[name].definitions
ρ≔⟨[name],types,rules⟩

⟦Ω:Meta⟧{{
  domain≜[name]
  version≜1.0.0
  ∀D∈AISP:Ambig(D)<0.02
}}

⟦Σ:Types⟧{{
  [inferred types]
}}

⟦Γ:Rules⟧{{
  [inferred rules]
}}

⟦Λ:Funcs⟧{{
  [symbol conversion]
}}

⟦Ε⟧⟨δ≜0.82;φ≜100;τ≜◊⁺⁺;⊢valid;∎⟩
```

## Rules
1. Output ONLY the AISP notation - no explanations
2. Preserve semantic meaning precisely
3. Use appropriate Unicode symbols from the reference
4. For ambiguous phrases, choose the most logical interpretation
5. Never hallucinate symbols not in the reference"#
    )
}

/// Create user prompt with context
fn create_user_prompt(
    prose: &str,
    tier: ConversionTier,
    unmapped: &[String],
    partial_output: Option<&str>,
) -> String {
    let mut prompt = format!(
        r#"Convert this prose to AISP ({} tier):

"{}""#,
        tier, prose
    );

    if !unmapped.is_empty() {
        prompt.push_str(&format!(
            "\n\nNote: These phrases couldn't be mapped deterministically: {}",
            unmapped.join(", ")
        ));
    }

    if let Some(partial) = partial_output {
        prompt.push_str(&format!("\n\nPartial conversion attempt:\n{}", partial));
    }

    prompt
}

/// Claude SDK fallback provider
///
/// Uses Claude models via the claude-agent-sdk-rs crate to convert
/// prose to AISP when deterministic conversion has low confidence.
pub struct ClaudeFallback {
    model: String,
}

impl Default for ClaudeFallback {
    fn default() -> Self {
        Self::new()
    }
}

impl ClaudeFallback {
    /// Create new Claude fallback with default model (sonnet)
    pub fn new() -> Self {
        Self {
            model: "sonnet".to_string(),
        }
    }

    /// Create with specific model
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }

    /// Use haiku for simple/fast conversions
    pub fn haiku() -> Self {
        Self::with_model("haiku")
    }

    /// Use sonnet for balanced conversions
    pub fn sonnet() -> Self {
        Self::with_model("sonnet")
    }

    /// Use opus for complex conversions
    pub fn opus() -> Self {
        Self::with_model("opus")
    }
}

#[async_trait]
impl LlmProvider for ClaudeFallback {
    async fn convert(
        &self,
        prose: &str,
        tier: ConversionTier,
        unmapped: &[String],
        partial_output: Option<&str>,
    ) -> Result<LlmResult> {
        use claude_agent_sdk_rs::{query, ClaudeAgentOptions, ContentBlock, Message, PermissionMode};

        let user_prompt = create_user_prompt(prose, tier, unmapped, partial_output);

        // Configure minimal Claude instance
        let options = ClaudeAgentOptions::builder()
            .model(&self.model)
            .system_prompt(system_prompt())
            .max_turns(1) // Single turn for conversion
            .permission_mode(PermissionMode::BypassPermissions)
            .tools(Vec::<String>::new()) // No tools needed
            .build();

        let messages = query(&user_prompt, Some(options)).await?;

        // Extract text response
        let mut output = String::new();
        let mut tokens_used = None;

        for message in messages {
            match message {
                Message::Assistant(msg) => {
                    for block in msg.message.content {
                        if let ContentBlock::Text(text) = block {
                            output.push_str(&text.text);
                        }
                    }
                }
                Message::Result(result) => {
                    if let Some(cost) = result.total_cost_usd {
                        // Rough token estimate from cost
                        tokens_used = Some((cost * 100000.0) as usize);
                    }
                }
                _ => {}
            }
        }

        Ok(LlmResult {
            output: output.trim().to_string(),
            provider: "claude".to_string(),
            model: self.model.clone(),
            tokens_used,
        })
    }

    async fn is_available(&self) -> bool {
        // Check if Claude Code CLI is available
        std::process::Command::new("claude")
            .arg("--version")
            .output()
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_prompt_generation() {
        let prompt = system_prompt();
        assert!(prompt.contains("AISP"));
        assert!(prompt.contains("Rosetta Stone"));
    }

    #[test]
    fn test_user_prompt_minimal() {
        let prompt = create_user_prompt("Define x as 5", ConversionTier::Minimal, &[], None);
        assert!(prompt.contains("Define x as 5"));
        assert!(prompt.contains("minimal"));
    }

    #[test]
    fn test_user_prompt_with_unmapped() {
        let prompt = create_user_prompt(
            "Define x as 5",
            ConversionTier::Standard,
            &["foo".to_string(), "bar".to_string()],
            None,
        );
        assert!(prompt.contains("foo"));
        assert!(prompt.contains("bar"));
    }
}
