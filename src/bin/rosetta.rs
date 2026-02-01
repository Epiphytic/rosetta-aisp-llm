//! Rosetta CLI - Convert prose to AISP notation
//!
//! A command-line tool for bidirectional prose ↔ AISP conversion
//! with optional LLM fallback for improved accuracy.

use clap::{Parser, Subcommand, ValueEnum};
use rosetta_aisp_llm::{
    convert_with_fallback, AispConverter, ConversionOptions, ConversionOptionsExt, ConversionTier,
    RosettaStone,
};
use rosetta_aisp::{
    get_all_categories, prose_to_symbol, symbol_to_prose, symbols_by_category,
};
use std::io::{self, Read};

#[derive(Parser)]
#[command(name = "rosetta")]
#[command(author = "epiphytic")]
#[command(version = "0.2.0")]
#[command(about = "Convert natural language prose to AISP symbolic notation", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert prose to AISP notation
    Convert {
        /// Prose text to convert (reads from stdin if not provided)
        #[arg(short, long)]
        input: Option<String>,

        /// Force a specific conversion tier
        #[arg(short, long, value_enum)]
        tier: Option<TierArg>,

        /// Output format
        #[arg(short, long, value_enum, default_value = "text")]
        format: OutputFormat,

        /// Enable LLM fallback for low-confidence conversions
        #[arg(long)]
        llm_fallback: bool,

        /// Confidence threshold for LLM fallback (default: 0.8)
        #[arg(long, default_value = "0.8")]
        threshold: f64,

        /// LLM model to use (haiku, sonnet, opus)
        #[arg(long, default_value = "sonnet")]
        model: String,
    },

    /// Convert AISP notation back to prose
    ToProse {
        /// AISP notation to convert (reads from stdin if not provided)
        #[arg(short, long)]
        input: Option<String>,
    },

    /// Detect the appropriate conversion tier for prose
    DetectTier {
        /// Prose text to analyze (reads from stdin if not provided)
        #[arg(short, long)]
        input: Option<String>,
    },

    /// Look up a symbol for a prose pattern
    Lookup {
        /// Prose pattern to look up
        pattern: String,
    },

    /// Look up prose for a symbol
    Reverse {
        /// AISP symbol to look up
        symbol: String,
    },

    /// List all available symbols
    Symbols {
        /// Filter by category
        #[arg(short, long)]
        category: Option<String>,
    },

    /// Show all available categories
    Categories,

    /// Perform round-trip conversion to test semantic preservation
    RoundTrip {
        /// Prose text to test (reads from stdin if not provided)
        #[arg(short, long)]
        input: Option<String>,

        /// Number of round-trips to perform
        #[arg(short, long, default_value = "5")]
        rounds: usize,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum TierArg {
    Minimal,
    Standard,
    Full,
}

impl From<TierArg> for ConversionTier {
    fn from(tier: TierArg) -> Self {
        match tier {
            TierArg::Minimal => ConversionTier::Minimal,
            TierArg::Standard => ConversionTier::Standard,
            TierArg::Full => ConversionTier::Full,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    /// Plain text output
    Text,
    /// JSON output with metadata
    Json,
}

fn read_input(input: Option<String>) -> String {
    match input {
        Some(text) => text,
        None => {
            let mut buffer = String::new();
            io::stdin()
                .read_to_string(&mut buffer)
                .expect("Failed to read from stdin");
            buffer.trim().to_string()
        }
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Convert {
            input,
            tier,
            format,
            llm_fallback,
            threshold,
            model,
        } => {
            let prose = read_input(input);

            let result = if llm_fallback {
                let options = ConversionOptionsExt {
                    tier: tier.map(Into::into),
                    confidence_threshold: Some(threshold),
                    enable_llm_fallback: true,
                    llm_model: Some(model),
                };
                convert_with_fallback(&prose, Some(options)).await
            } else {
                let options = ConversionOptions {
                    tier: tier.map(Into::into),
                    confidence_threshold: Some(threshold),
                };
                AispConverter::convert(&prose, Some(options))
            };

            match format {
                OutputFormat::Text => {
                    println!("{}", result.output);
                    eprintln!();
                    eprintln!("---");
                    eprintln!("Tier: {:?}", result.tier);
                    eprintln!("Confidence: {:.1}%", result.confidence * 100.0);
                    eprintln!(
                        "Tokens: {} → {} ({:.2}x)",
                        result.tokens.input, result.tokens.output, result.tokens.ratio
                    );
                    if result.used_fallback {
                        eprintln!("LLM fallback: used");
                    }
                    if !result.unmapped.is_empty() {
                        eprintln!("Unmapped: {:?}", result.unmapped);
                    }
                }
                OutputFormat::Json => {
                    let json = serde_json::to_string_pretty(&result).expect("Failed to serialize");
                    println!("{}", json);
                }
            }
        }

        Commands::ToProse { input } => {
            let aisp = read_input(input);
            let prose = AispConverter::to_prose(&aisp);
            println!("{}", prose);
        }

        Commands::DetectTier { input } => {
            let prose = read_input(input);
            let tier = AispConverter::detect_tier(&prose);
            println!("{:?}", tier);
        }

        Commands::Lookup { pattern } => {
            match prose_to_symbol(&pattern) {
                Some(symbol) => println!("{}", symbol),
                None => {
                    eprintln!("No symbol found for pattern: {}", pattern);
                    std::process::exit(1);
                }
            }
        }

        Commands::Reverse { symbol } => {
            match symbol_to_prose(&symbol) {
                Some(prose) => println!("{}", prose),
                None => {
                    eprintln!("No prose found for symbol: {}", symbol);
                    std::process::exit(1);
                }
            }
        }

        Commands::Symbols { category } => {
            match category {
                Some(cat) => {
                    let symbols = symbols_by_category(&cat);
                    if symbols.is_empty() {
                        eprintln!("No symbols found for category: {}", cat);
                        eprintln!("Available categories: {:?}", get_all_categories());
                        std::process::exit(1);
                    }
                    for symbol in symbols {
                        if let Some(prose) = symbol_to_prose(symbol) {
                            println!("{} → {}", symbol, prose);
                        } else {
                            println!("{}", symbol);
                        }
                    }
                }
                None => {
                    for category in get_all_categories() {
                        println!("\n=== {} ===", category);
                        for symbol in symbols_by_category(category) {
                            if let Some(prose) = symbol_to_prose(symbol) {
                                println!("  {} → {}", symbol, prose);
                            } else {
                                println!("  {}", symbol);
                            }
                        }
                    }
                }
            }
        }

        Commands::Categories => {
            for category in get_all_categories() {
                println!("{}", category);
            }
        }

        Commands::RoundTrip { input, rounds } => {
            let original = read_input(input);
            let mut current = original.clone();

            println!("Original: {}", original);
            println!();

            for i in 1..=rounds {
                let (aisp, mapped_chars, _) = RosettaStone::convert(&current);
                let prose = RosettaStone::to_prose(&aisp);
                let similarity = RosettaStone::semantic_similarity(&original, &prose);
                let confidence = RosettaStone::confidence(current.len(), mapped_chars);

                println!("Round {} (confidence: {:.1}%, similarity: {:.1}%):", i, confidence * 100.0, similarity * 100.0);
                println!("  AISP: {}", aisp);
                println!("  Prose: {}", prose);
                println!();

                current = prose;
            }

            let final_similarity = RosettaStone::semantic_similarity(&original, &current);
            println!("Final semantic similarity: {:.1}%", final_similarity * 100.0);

            if final_similarity < 0.30 {
                eprintln!("Warning: Semantic drift exceeded acceptable threshold");
                std::process::exit(1);
            }
        }
    }
}
