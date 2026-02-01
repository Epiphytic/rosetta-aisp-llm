//! LLM Fallback Tests
//!
//! Tests for conversions that benefit from LLM fallback functionality.
//! These tests verify that low-confidence conversions can be improved
//! using the rosetta-aisp-llm integration.

use rosetta_aisp_llm::{
    convert_with_fallback, AispConverter, ClaudeFallback, ConversionOptions, ConversionOptionsExt,
    ConversionResult, ConversionTier, LlmProvider, RosettaStone,
};

/// Test cases with expected low deterministic confidence
/// These represent prose that doesn't map well to standard Rosetta patterns
const LOW_CONFIDENCE_CASES: &[&str] = &[
    "The quantum entanglement manifests probabilistic correlation",
    "Neural networks approximate arbitrary continuous functions",
    "Homomorphic encryption preserves algebraic structure",
    "Monadic composition lifts pure functions into effectful contexts",
];

/// Test cases that should work well without LLM
const HIGH_CONFIDENCE_CASES: &[&str] = &[
    "for all x in S, x equals y",
    "Define x as 5 and y as 10",
    "if valid then proceed else reject",
    "there exists a user such that admin is true",
];

#[tokio::test]
async fn test_llm_provider_availability() {
    let provider = ClaudeFallback::new();
    // This test just verifies the provider can be created and checked
    let _ = provider.is_available().await;
}

#[tokio::test]
async fn test_convert_with_fallback_disabled() {
    let prose = "Define x as 5";

    let options = ConversionOptionsExt {
        enable_llm_fallback: false,
        ..Default::default()
    };

    let result = convert_with_fallback(prose, Some(options)).await;

    // Without fallback, should use deterministic conversion
    assert!(!result.used_fallback);
    assert!(result.output.contains("≜"));
}

#[tokio::test]
async fn test_high_confidence_no_fallback_needed() {
    for prose in HIGH_CONFIDENCE_CASES {
        let options = ConversionOptionsExt {
            enable_llm_fallback: true,
            confidence_threshold: Some(0.5),
            ..Default::default()
        };

        let result = convert_with_fallback(prose, Some(options)).await;

        // High confidence cases should not need fallback
        // (though they may still use it if threshold is very high)
        println!(
            "Prose: '{}' -> confidence: {:.1}%, used_fallback: {}",
            prose,
            result.confidence * 100.0,
            result.used_fallback
        );

        // At minimum, output should not be empty
        assert!(!result.output.is_empty());
    }
}

#[tokio::test]
async fn test_low_confidence_with_fallback() {
    // Skip if Claude CLI is not available
    let provider = ClaudeFallback::new();
    if !provider.is_available().await {
        eprintln!("Skipping LLM fallback test: Claude CLI not available");
        return;
    }

    for prose in LOW_CONFIDENCE_CASES {
        let options = ConversionOptionsExt {
            enable_llm_fallback: true,
            confidence_threshold: Some(0.9), // High threshold to force fallback
            llm_model: Some("haiku".to_string()), // Use haiku for speed
            ..Default::default()
        };

        let result = convert_with_fallback(prose, Some(options)).await;

        println!(
            "Prose: '{}'\n  -> confidence: {:.1}%, used_fallback: {}",
            prose,
            result.confidence * 100.0,
            result.used_fallback
        );

        // Output should not be empty
        assert!(
            !result.output.is_empty(),
            "Output should not be empty for: {}",
            prose
        );

        // If fallback was used, confidence should be high
        if result.used_fallback {
            assert!(
                result.confidence >= 0.90,
                "LLM fallback should produce high confidence, got: {:.1}%",
                result.confidence * 100.0
            );
        }
    }
}

#[tokio::test]
async fn test_tier_preserved_with_fallback() {
    // Skip if Claude CLI is not available
    let provider = ClaudeFallback::new();
    if !provider.is_available().await {
        eprintln!("Skipping LLM fallback test: Claude CLI not available");
        return;
    }

    let prose = "Define a type User with id and name fields";

    for tier in [
        ConversionTier::Minimal,
        ConversionTier::Standard,
        ConversionTier::Full,
    ] {
        let options = ConversionOptionsExt {
            tier: Some(tier),
            enable_llm_fallback: true,
            confidence_threshold: Some(0.99), // Force fallback
            llm_model: Some("haiku".to_string()),
            use_aisp_prompt: false,
        };

        let result = convert_with_fallback(prose, Some(options)).await;

        // Tier should be preserved
        assert_eq!(
            result.tier, tier,
            "Tier should be preserved, expected {:?}, got {:?}",
            tier, result.tier
        );
    }
}

#[tokio::test]
async fn test_deterministic_fallback_consistency() {
    // Even without LLM, the fallback path should be consistent
    let prose = "for all users, if authenticated then allow";

    let options = ConversionOptionsExt {
        enable_llm_fallback: false,
        ..Default::default()
    };

    // Run twice to verify consistency
    let result1 = convert_with_fallback(prose, Some(options.clone())).await;
    let result2 = convert_with_fallback(prose, Some(options)).await;

    assert_eq!(
        result1.output, result2.output,
        "Deterministic conversion should be consistent"
    );
    assert_eq!(result1.confidence, result2.confidence);
    assert!(!result1.used_fallback);
    assert!(!result2.used_fallback);
}

#[tokio::test]
async fn test_unmapped_words_tracking() {
    let prose = "xyzabc undefined_concept foo_bar_baz";

    let options = ConversionOptionsExt {
        enable_llm_fallback: false,
        ..Default::default()
    };

    let result = convert_with_fallback(prose, Some(options)).await;

    // Should have some unmapped words
    println!("Unmapped: {:?}", result.unmapped);
    // Note: exact unmapped words depend on implementation
}

#[tokio::test]
async fn test_conversion_result_serialization() {
    let prose = "for all x in S";

    let result = convert_with_fallback(prose, None).await;

    // Should serialize to JSON
    let json = serde_json::to_string(&result).expect("Should serialize");
    assert!(json.contains("output"));
    assert!(json.contains("confidence"));
    assert!(json.contains("tier"));
    assert!(json.contains("used_fallback"));

    // Should deserialize back
    let parsed: ConversionResult = serde_json::from_str(&json).expect("Should deserialize");
    assert_eq!(parsed.output, result.output);
    assert_eq!(parsed.used_fallback, result.used_fallback);
}

#[test]
fn test_rosetta_stone_direct_conversion() {
    // Direct RosettaStone usage should work without async
    let (aisp, mapped_chars, _unmapped) = RosettaStone::convert("for all x in S");

    assert!(aisp.contains("∀"), "Should contain universal quantifier");
    assert!(aisp.contains("∈"), "Should contain element-of symbol");
    assert!(mapped_chars > 0, "Should have mapped some characters");

    // Round trip
    let prose = RosettaStone::to_prose(&aisp);
    assert!(
        prose.to_lowercase().contains("for all"),
        "Should convert back to 'for all'"
    );
}

#[test]
fn test_aisp_converter_without_llm() {
    // AispConverter should work independently of LLM
    let result = AispConverter::convert(
        "The user must authenticate before accessing the API",
        Some(ConversionOptions {
            tier: Some(ConversionTier::Standard),
            ..Default::default()
        }),
    );

    assert!(
        result.output.contains("⟦Ω:Meta⟧"),
        "Should have Meta block"
    );
    assert!(
        !result.used_fallback,
        "Should not use fallback without LLM"
    );
}
