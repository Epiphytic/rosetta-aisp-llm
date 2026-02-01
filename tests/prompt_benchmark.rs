//! Prompt Benchmark Tests
//!
//! Evaluates 4 quadrants of LLM fallback:
//! - haiku + English prompt
//! - haiku + AISP prompt
//! - sonnet + English prompt
//! - sonnet + AISP prompt
//!
//! Measures speed and accuracy for each combination.

use rosetta_aisp_llm::{
    convert_with_fallback, ClaudeFallback, ConversionOptionsExt, ConversionTier, LlmProvider,
};
use std::time::{Duration, Instant};

/// Test cases that require LLM fallback (low deterministic confidence)
const TEST_CASES: &[(&str, &[&str])] = &[
    // (input prose, expected symbols in output)
    ("Define x as 5", &["≜", "x", "5"]),
    ("for all x in S, x equals y", &["∀", "∈", "≡"]),
    ("if valid then proceed else reject", &["→"]),
    ("there exists a user such that admin is true", &["∃", "⊤"]),
    (
        "The quantum entanglement manifests probabilistic correlation",
        &["≈", "∝"], // May use approximate or proportional
    ),
];

/// More challenging test cases for comprehensive benchmark
const CHALLENGING_CASES: &[(&str, &[&str])] = &[
    // Complex logical statements
    (
        "For all users u, if u is authenticated and u has permission p, then u can access resource r",
        &["∀", "∧", "⇒", "∈"],
    ),
    // Dependent types
    (
        "Define a vector type parameterized by length n where n is a natural number",
        &["Π", "ℕ", "≜", "Vec"],
    ),
    // Category theory
    (
        "The functor F preserves composition, meaning F of g composed with f equals F of g composed with F of f",
        &["∘", "≡", "𝔽"],
    ),
    // Set theory with quantifiers
    (
        "There exists a unique element x in set S such that for all y in S, x is less than or equal to y",
        &["∃!", "∀", "∈", "≤"],
    ),
    // Lambda calculus
    (
        "Define the Y combinator as a function that takes f and returns f applied to the Y combinator applied to f",
        &["λ", "fix", "≜"],
    ),
    // Proof-carrying code
    (
        "The document is well-formed if and only if it has a valid header and at least two blocks",
        &["↔", "∧", "wf", "≥"],
    ),
    // Type inference
    (
        "If expression e has type A implies B, and expression e2 has type A, then applying e to e2 has type B",
        &["⊢", "→", ":"],
    ),
    // Error handling
    (
        "When ambiguity is detected greater than threshold 0.02, reject the input and request clarification",
        &["Ambig", ">", "⇒", "reject"],
    ),
];

/// Result of a single benchmark run
#[derive(Debug, Clone)]
struct BenchmarkResult {
    model: String,
    prompt_style: String,
    test_case: String,
    duration_ms: u128,
    output: String,
    symbols_found: usize,
    symbols_expected: usize,
    accuracy: f64,
    used_fallback: bool,
}

/// Run a single benchmark test
async fn run_benchmark(
    prose: &str,
    expected_symbols: &[&str],
    model: &str,
    use_aisp_prompt: bool,
) -> Option<BenchmarkResult> {
    let options = ConversionOptionsExt {
        tier: Some(ConversionTier::Minimal),
        enable_llm_fallback: true,
        confidence_threshold: Some(0.99), // Force fallback
        llm_model: Some(model.to_string()),
        use_aisp_prompt,
    };

    let start = Instant::now();
    let result = convert_with_fallback(prose, Some(options)).await;
    let duration = start.elapsed();

    // Count how many expected symbols appear in output
    let symbols_found = expected_symbols
        .iter()
        .filter(|s| result.output.contains(*s))
        .count();

    let accuracy = if expected_symbols.is_empty() {
        1.0
    } else {
        symbols_found as f64 / expected_symbols.len() as f64
    };

    Some(BenchmarkResult {
        model: model.to_string(),
        prompt_style: if use_aisp_prompt {
            "aisp".to_string()
        } else {
            "english".to_string()
        },
        test_case: prose.chars().take(40).collect(),
        duration_ms: duration.as_millis(),
        output: result.output,
        symbols_found,
        symbols_expected: expected_symbols.len(),
        accuracy,
        used_fallback: result.used_fallback,
    })
}

/// Run all benchmarks for a specific quadrant (basic cases)
async fn run_quadrant(
    model: &str,
    use_aisp_prompt: bool,
) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    for (prose, expected) in TEST_CASES {
        if let Some(result) = run_benchmark(prose, expected, model, use_aisp_prompt).await {
            results.push(result);
        }
    }

    results
}

/// Run challenging benchmarks for a specific quadrant
async fn run_challenging_quadrant(
    model: &str,
    use_aisp_prompt: bool,
) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    for (prose, expected) in CHALLENGING_CASES {
        if let Some(result) = run_benchmark(prose, expected, model, use_aisp_prompt).await {
            results.push(result);
        }
    }

    results
}

/// Print summary statistics for a quadrant
fn print_quadrant_summary(results: &[BenchmarkResult]) {
    if results.is_empty() {
        println!("  No results");
        return;
    }

    let total_time: u128 = results.iter().map(|r| r.duration_ms).sum();
    let avg_time = total_time / results.len() as u128;
    let avg_accuracy: f64 = results.iter().map(|r| r.accuracy).sum::<f64>() / results.len() as f64;
    let fallback_used = results.iter().filter(|r| r.used_fallback).count();

    println!(
        "  Avg time: {}ms | Avg accuracy: {:.1}% | Fallback used: {}/{}",
        avg_time,
        avg_accuracy * 100.0,
        fallback_used,
        results.len()
    );
}

#[tokio::test]
async fn benchmark_all_quadrants() {
    // Check if Claude CLI is available
    let provider = ClaudeFallback::new();
    if !provider.is_available().await {
        eprintln!("Skipping benchmark: Claude CLI not available");
        return;
    }

    println!("\n=== PROMPT BENCHMARK: 4 QUADRANTS ===\n");

    // Quadrant 1: haiku + english
    println!("## Quadrant 1: haiku + english");
    let q1 = run_quadrant("haiku", false).await;
    for r in &q1 {
        println!(
            "  [{}ms] {:.1}% acc | {}",
            r.duration_ms,
            r.accuracy * 100.0,
            r.test_case
        );
    }
    print_quadrant_summary(&q1);
    println!();

    // Quadrant 2: haiku + aisp
    println!("## Quadrant 2: haiku + aisp");
    let q2 = run_quadrant("haiku", true).await;
    for r in &q2 {
        println!(
            "  [{}ms] {:.1}% acc | {}",
            r.duration_ms,
            r.accuracy * 100.0,
            r.test_case
        );
    }
    print_quadrant_summary(&q2);
    println!();

    // Quadrant 3: sonnet + english
    println!("## Quadrant 3: sonnet + english");
    let q3 = run_quadrant("sonnet", false).await;
    for r in &q3 {
        println!(
            "  [{}ms] {:.1}% acc | {}",
            r.duration_ms,
            r.accuracy * 100.0,
            r.test_case
        );
    }
    print_quadrant_summary(&q3);
    println!();

    // Quadrant 4: sonnet + aisp
    println!("## Quadrant 4: sonnet + aisp");
    let q4 = run_quadrant("sonnet", true).await;
    for r in &q4 {
        println!(
            "  [{}ms] {:.1}% acc | {}",
            r.duration_ms,
            r.accuracy * 100.0,
            r.test_case
        );
    }
    print_quadrant_summary(&q4);
    println!();

    // Summary comparison
    println!("=== SUMMARY ===\n");

    let summaries = [
        ("haiku+english", &q1),
        ("haiku+aisp", &q2),
        ("sonnet+english", &q3),
        ("sonnet+aisp", &q4),
    ];

    println!("| Quadrant        | Avg Time | Avg Accuracy | Total Time |");
    println!("|-----------------|----------|--------------|------------|");

    for (name, results) in &summaries {
        if results.is_empty() {
            continue;
        }
        let total_time: u128 = results.iter().map(|r| r.duration_ms).sum();
        let avg_time = total_time / results.len() as u128;
        let avg_accuracy: f64 =
            results.iter().map(|r| r.accuracy).sum::<f64>() / results.len() as f64;

        println!(
            "| {:<15} | {:>6}ms | {:>10.1}% | {:>8}ms |",
            name,
            avg_time,
            avg_accuracy * 100.0,
            total_time
        );
    }

    // Determine winner
    println!("\n=== RECOMMENDATION ===\n");

    let best = summaries
        .iter()
        .filter(|(_, r)| !r.is_empty())
        .max_by(|(_, a), (_, b)| {
            // Score = accuracy * 100 - (time_penalty)
            // Prefer accuracy, but penalize slow times
            let score_a = a.iter().map(|r| r.accuracy).sum::<f64>() / a.len() as f64 * 100.0
                - (a.iter().map(|r| r.duration_ms).sum::<u128>() / a.len() as u128) as f64 * 0.01;
            let score_b = b.iter().map(|r| r.accuracy).sum::<f64>() / b.len() as f64 * 100.0
                - (b.iter().map(|r| r.duration_ms).sum::<u128>() / b.len() as u128) as f64 * 0.01;
            score_a.partial_cmp(&score_b).unwrap()
        });

    if let Some((name, _)) = best {
        println!("Best combination: {}", name);
    }
}

#[tokio::test]
async fn benchmark_haiku_english() {
    let provider = ClaudeFallback::new();
    if !provider.is_available().await {
        eprintln!("Skipping: Claude CLI not available");
        return;
    }

    println!("\n=== HAIKU + ENGLISH ===\n");
    let results = run_quadrant("haiku", false).await;
    for r in &results {
        println!("[{}ms] {}: {}", r.duration_ms, r.test_case, r.output);
    }
    print_quadrant_summary(&results);
}

#[tokio::test]
async fn benchmark_haiku_aisp() {
    let provider = ClaudeFallback::new();
    if !provider.is_available().await {
        eprintln!("Skipping: Claude CLI not available");
        return;
    }

    println!("\n=== HAIKU + AISP ===\n");
    let results = run_quadrant("haiku", true).await;
    for r in &results {
        println!("[{}ms] {}: {}", r.duration_ms, r.test_case, r.output);
    }
    print_quadrant_summary(&results);
}

#[tokio::test]
async fn benchmark_sonnet_english() {
    let provider = ClaudeFallback::new();
    if !provider.is_available().await {
        eprintln!("Skipping: Claude CLI not available");
        return;
    }

    println!("\n=== SONNET + ENGLISH ===\n");
    let results = run_quadrant("sonnet", false).await;
    for r in &results {
        println!("[{}ms] {}: {}", r.duration_ms, r.test_case, r.output);
    }
    print_quadrant_summary(&results);
}

#[tokio::test]
async fn benchmark_sonnet_aisp() {
    let provider = ClaudeFallback::new();
    if !provider.is_available().await {
        eprintln!("Skipping: Claude CLI not available");
        return;
    }

    println!("\n=== SONNET + AISP ===\n");
    let results = run_quadrant("sonnet", true).await;
    for r in &results {
        println!("[{}ms] {}: {}", r.duration_ms, r.test_case, r.output);
    }
    print_quadrant_summary(&results);
}

#[tokio::test]
async fn benchmark_challenging_all_quadrants() {
    // Check if Claude CLI is available
    let provider = ClaudeFallback::new();
    if !provider.is_available().await {
        eprintln!("Skipping challenging benchmark: Claude CLI not available");
        return;
    }

    println!("\n=== CHALLENGING BENCHMARK: 4 QUADRANTS ===\n");

    // Quadrant 1: haiku + english
    println!("## Quadrant 1: haiku + english (challenging)");
    let q1 = run_challenging_quadrant("haiku", false).await;
    for r in &q1 {
        println!(
            "  [{}ms] {:.1}% acc | {}",
            r.duration_ms,
            r.accuracy * 100.0,
            r.test_case
        );
    }
    print_quadrant_summary(&q1);
    println!();

    // Quadrant 2: haiku + aisp
    println!("## Quadrant 2: haiku + aisp (challenging)");
    let q2 = run_challenging_quadrant("haiku", true).await;
    for r in &q2 {
        println!(
            "  [{}ms] {:.1}% acc | {}",
            r.duration_ms,
            r.accuracy * 100.0,
            r.test_case
        );
    }
    print_quadrant_summary(&q2);
    println!();

    // Quadrant 3: sonnet + english
    println!("## Quadrant 3: sonnet + english (challenging)");
    let q3 = run_challenging_quadrant("sonnet", false).await;
    for r in &q3 {
        println!(
            "  [{}ms] {:.1}% acc | {}",
            r.duration_ms,
            r.accuracy * 100.0,
            r.test_case
        );
    }
    print_quadrant_summary(&q3);
    println!();

    // Quadrant 4: sonnet + aisp
    println!("## Quadrant 4: sonnet + aisp (challenging)");
    let q4 = run_challenging_quadrant("sonnet", true).await;
    for r in &q4 {
        println!(
            "  [{}ms] {:.1}% acc | {}",
            r.duration_ms,
            r.accuracy * 100.0,
            r.test_case
        );
    }
    print_quadrant_summary(&q4);
    println!();

    // Summary comparison
    println!("=== CHALLENGING SUMMARY ===\n");

    let summaries = [
        ("haiku+english", &q1),
        ("haiku+aisp", &q2),
        ("sonnet+english", &q3),
        ("sonnet+aisp", &q4),
    ];

    println!("| Quadrant        | Avg Time | Avg Accuracy | Total Time |");
    println!("|-----------------|----------|--------------|------------|");

    for (name, results) in &summaries {
        if results.is_empty() {
            continue;
        }
        let total_time: u128 = results.iter().map(|r| r.duration_ms).sum();
        let avg_time = total_time / results.len() as u128;
        let avg_accuracy: f64 =
            results.iter().map(|r| r.accuracy).sum::<f64>() / results.len() as f64;

        println!(
            "| {:<15} | {:>6}ms | {:>10.1}% | {:>8}ms |",
            name,
            avg_time,
            avg_accuracy * 100.0,
            total_time
        );
    }

    // Determine winner
    println!("\n=== CHALLENGING RECOMMENDATION ===\n");

    let best = summaries
        .iter()
        .filter(|(_, r)| !r.is_empty())
        .max_by(|(_, a), (_, b)| {
            let score_a = a.iter().map(|r| r.accuracy).sum::<f64>() / a.len() as f64 * 100.0
                - (a.iter().map(|r| r.duration_ms).sum::<u128>() / a.len() as u128) as f64 * 0.01;
            let score_b = b.iter().map(|r| r.accuracy).sum::<f64>() / b.len() as f64 * 100.0
                - (b.iter().map(|r| r.duration_ms).sum::<u128>() / b.len() as u128) as f64 * 0.01;
            score_a.partial_cmp(&score_b).unwrap()
        });

    if let Some((name, _)) = best {
        println!("Best combination for challenging cases: {}", name);
    }
}
