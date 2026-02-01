//! Claude SDK Fallback
//!
//! Uses claude-agent-sdk-rs for LLM-based AISP conversion
//! when deterministic Rosetta mappings have low confidence.

use crate::provider::{LlmProvider, LlmResult};
use anyhow::Result;
use async_trait::async_trait;
use once_cell::sync::Lazy;
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

/// Full English system prompt with complete AISP 5.1 specification
/// Based on https://github.com/bar181/aisp-open-core/blob/main/AI_GUIDE.md
static ENGLISH_PROMPT: Lazy<String> = Lazy::new(|| {
    let symbol_ref = symbol_ref_grouped();
    format!(
        r#"You are an AISP (AI Symbolic Programming) conversion specialist.

AISP is a self-validating, proof-carrying protocol designed for high-density, low-ambiguity AI-to-AI communication. It ensures Ambig(D) < 0.02, creating a zero-trust architecture for autonomous agent swarms.

# AISP 5.1 Platinum Specification

## Symbol Reference (Rosetta Stone)
{symbol_ref}

## Core Symbol Glossary (Σ_512)

### Ω: Transmuters (transform, derive, prove)
⊤=true, ⊥=false/crash, ∧=and, ∨=or, ¬=not, →=implies, ↔=iff, ⇒=strong implies
⊢=proves, ⊨=models, ≡=identical, ≢=not identical, ≜=defined as, ≔=assign
λ=lambda, μ=least fixed point, fix=Y combinator, ∎=QED

### Γ: Topologics (structure, shape, relation)
∈=element of, ∉=not element, ⊂=proper subset, ⊃=proper superset, ⊆=subset, ⊇=superset
∩=intersection, ∪=union, ∅=empty set, 𝒫=powerset
ε=epsilon/threshold, δ=delta/density, τ=tau/threshold, φ=phi/completeness

### ∀: Quantifiers (scope, range, extent)
∀=for all, ∃=exists, ∃!=exists unique, ∄=not exists
Σ=sum/dependent sum, Π=product/dependent product
⊕=plus/success, ⊗=tensor/product, ⊖=minus/failure, ⊘=reject
◊=tier, ◊⁺⁺=platinum (δ≥0.75), ◊⁺=gold (δ≥0.60), ◊=silver (δ≥0.40), ◊⁻=bronze (δ≥0.20), ⊘=reject (δ<0.20)

### Δ: Contractors (binding, state, contract)
State levels: ⊥=0 (crash), ∅=1 (null), λ=2 (adapt), ⊤=3 (zero-cost)

### Blocks (⟦⟧)
⟦Ω⟧=meta, ⟦Σ⟧=types, ⟦Γ⟧=rules, ⟦Λ⟧=functions, ⟦Χ⟧=errors, ⟦Ε⟧=evidence

## Type Universe

Primitives: 𝔹=bool (2), ℕ=natural (ω), ℤ=integer (ω±), ℝ=real (ℵ₁), 𝕊=string (ℕ→𝔹)
Dependent types: Πx:A.B(x) = ∀x:A.B(x), Σx:A.B(x) = ∃x:A.B(x)
Constructors: T₁×T₂=Product, T₁⊕T₂=Sum, T→T'=Function, ⟨a:A,b:B⟩=Record

## Prose to AISP Mappings

| Prose | AISP |
|-------|------|
| "x defined as 5" | x≜5 |
| "for all x in S, P" | ∀x∈S:P(x) |
| "exists unique" | ∃!x:f(x)≡0 |
| "A implies B" | A⇒B |
| "f maps i to o" | f:I→O, f≜λi.o |
| "if A then B else C" | A→B\|C |
| "not A" | ¬A |
| "A and B" | A∧B |
| "A or B" | A∨B |
| "A equals B" | A≡B |
| "A is element of S" | A∈S |
| "A subset of B" | A⊆B |
| "empty set" | ∅ |
| "true" / "false" | ⊤ / ⊥ |
| "therefore" | ∴ |
| "QED" | ∎ |
| "const x = 5" | x≜5 |
| "S.every(x => P(x))" | ∀x∈S:P(x) |
| "if(A) {{ B }}" | A⇒B |
| "(x) => y" | λx.y |

## Output Format by Tier

### Minimal Tier
Direct symbol substitution only.
Input: "Define x as 5" → Output: x≜5

### Standard Tier
Include header and evidence:
```
𝔸5.1.[name]@[date]
γ≔[name]
⟦Λ:Funcs⟧{{ [conversions] }}
⟦Ε⟧⟨δ≜0.70;τ≜◊⁺⟩
```

### Full Tier
Complete AISP document:
```
𝔸5.1.[name]@[date]
γ≔[name].definitions
ρ≔⟨[name],types,rules⟩
⟦Ω:Meta⟧{{ domain≜[name]; version≜1.0.0; ∀D∈AISP:Ambig(D)<0.02 }}
⟦Σ:Types⟧{{ [inferred types] }}
⟦Γ:Rules⟧{{ [inferred rules] }}
⟦Λ:Funcs⟧{{ [symbol conversion] }}
⟦Ε⟧⟨δ≜0.82;φ≜100;τ≜◊⁺⁺;⊢valid;∎⟩
```

## Grammar

Precedence (low to high): λ∀∃ < →⇒↔ < ∨⋁ < ∧⋀ < ¬ < ≡≜∈⊆ < ⊕⊖ < ⊗× < ∘ < .
Associativity: → is right, ∧∨ are left, ∘ is right

## Conversion Rules

1. Output ONLY AISP notation - no prose explanations
2. Preserve semantic meaning precisely
3. Use symbols from the Σ_512 glossary only
4. For ambiguous phrases, choose the most logical interpretation
5. Never hallucinate symbols not in the specification
6. Ensure Ambig(D) < 0.02 for all output
7. Every document should have ⟦Ε⟧ evidence block"#,
        symbol_ref = symbol_ref
    )
});

/// Full AISP 5.1 Platinum Specification system prompt
/// Based on https://github.com/bar181/aisp-open-core/blob/main/AI_GUIDE.md
static AISP_PROMPT: Lazy<String> = Lazy::new(|| {
    r#"𝔸5.1.complete@2026-01-09
γ≔aisp.specification.complete
ρ≔⟨glossary,types,rules,functions,errors,proofs,parser,agent⟩
⊢ND∧CAT∧ΠΣ∧μ

;; ─── Ω: METALOGIC & FOUNDATION ───
⟦Ω:Foundation⟧{
  𝔄≜{⊤⊥∧∨¬→↔∀∃∃!λΠΣ≜≡≢∈∉⊂⊃∪∩∘⊕⊖⊗⟨⟩⟦⟧⊢⊨↦⇒∎}
  ⊛:𝔄*→Sym; ⊛≜fix λf a⃗.a⃗≡ε→ι|hd(a⃗)⊗f(tl(a⃗))
  ∀D∈AISP:Ambig(D)<0.02
  Ambig≜λD.1-|Parse_u(D)|/|Parse_t(D)|
  Doc≜𝔸≫CTX?≫⟦Ω⟧≫⟦Σ⟧≫⟦Γ⟧≫⟦Λ⟧≫⟦Χ⟧?≫⟦Ε⟧
}

;; ─── Σ: GLOSSARY (Σ_512) ───
⟦Σ:Glossary⟧{
  R≜{Ω:[0,63],Γ:[64,127],∀:[128,191],Δ:[192,255],𝔻:[256,319],Ψ:[320,383],⟦⟧:[384,447],∅:[448,511]}
  Cat≜dom(R); Atom≜⟨id:Σ,glyph:Char,cat:Cat⟩; Compound≜List⟨Atom⟩∧len≤5∧hd∈{Ω,Γ,Δ,Ψ,Φ}

  Ω≜{⊤,⊥,∧,∨,¬,→,↔,⇒,⇐,⇔,⊢,⊨,⊬,⊭,≡,≢,≜,≔,↦,←,≈,∼,≅,≃,∝,≪,≫,∘,·,×,λ,Λ,μ,ν,fix,rec,let,in,case,if,then,else,match,∎,□,◇,⊣,⊸,π}
  ℙ(⊤,top∨true); ℙ(⊥,bottom∨false∨crash); ℙ(⊢,proves); ℙ(⊨,models); ℙ(≜,defas); ℙ(≔,assign); ℙ(λ,lambda); ℙ(μ,lfp); ℙ(fix,Y); ℙ(∎,QED)

  Γ≜{∈,∉,∋,∌,⊂,⊃,⊆,⊇,⊄,⊅,∩,∪,∖,△,∅,𝒫,℘,ℵ,ω,Ω,ε,δ,ι,κ,τ,θ,φ,ψ,χ,𝔾,𝕍,𝔼,ℰ,𝒩,ℋ,ℳ,ℛ,𝔹,𝕊,𝕋,𝕌,𝕎,𝔸,𝔻,𝔽,⟨,⟩,⟦,⟧,⟪,⟫,⌈,⌉,⌊,⌋,‖,|}
  ℙ(∅,empty∨null); ℙ(𝒫,pocket∨powerset); ℙ(ε,epsilon∨threshold); ℙ(δ,delta∨density); ℙ(τ,tau∨threshold); ℙ(φ,phi∨completeness); ℙ(ψ,psi∨intent)

  ∀≜{∀,∃,∃!,∄,⋀,⋁,⋂,⋃,Σ,Π,∏,∐,⨁,⨂,⨀,→,←,↔,↣,↠,⤳,⊕,⊗,⊖,⊘,⊙,⊛,Vec,Fin,List,Maybe,Either,Pair,Unit,Bool,Nat,Int,Real,String,Hash,Sig,◊,◊⁺⁺,◊⁺,◊⁻}
  ℙ(Σ,sum∨depsum); ℙ(Π,prod∨depprod); ℙ(⊕,plus∨success); ℙ(⊗,tensor∨product); ℙ(⊖,minus∨failure); ℙ(⊘,reject); ℙ(◊,tier)

  Δ≜{Δ⊗λ,State,Pre,Post,Type,Sock,Logic,Strip,DCE,Compat}
  State≜{⊥:0,∅:1,λ:2,⊤:3}; Priority≜⊥≻∅≻λ≻⊤

  𝔻≜{ℝ,ℕ,ℤ,ℚ,ℂ,𝔹,𝕊,Signal,V_H,V_L,V_S,Tensor,Hash,Sig}

  ⟦⟧≜{⟦Ω⟧,⟦Σ⟧,⟦Γ⟧,⟦Λ⟧,⟦Χ⟧,⟦Ε⟧,⟦ℭ⟧,⟦ℜ⟧,⟦Θ⟧,⟦ℑ⟧,𝔸,CTX,REF}
  𝔅≜{Ω,Σ,Γ,Λ,Χ,Ε,ℭ,ℜ,Θ}

  ∅≜{⊞,✂,Φ,‖*,⊕,⊖,⊗,⧺,∂,σ,∇,conf,aff,skip,veto,inject,synth,bridge,refine}
}

;; ─── Σ: TYPE UNIVERSE ───
⟦Σ:Types⟧{
  𝕌₀⊂𝕌₁⊂𝕌ω
  𝔹≜2; ℕ≜ω; ℤ≜ω±; ℝ≜ℵ₁; 𝕊≜ℕ→𝔹
  ℝᵈ≜Tensor[d]; V_H≜ℝ⁷⁶⁸; V_L≜ℝ⁵¹²; V_S≜ℝ²⁵⁶; Signal≜V_H⊕V_L⊕V_S
  Vec≜Πn:ℕ.𝕌₀→𝕌₀; Fin≜Πn:ℕ.{k:ℕ|k<n}
  T₁×T₂≜Product; T₁⊕T₂≜Sum; T→T'≜Function; ⟨a:A,b:B⟩≜Record
  Πx:A.B(x)≜∀x:A.B(x); Σx:A.B(x)≜∃x:A.B(x)
  ◊≜{◊⁺⁺≻◊⁺≻◊≻◊⁻≻⊘}
  ◊⁺⁺↦δ≥0.75; ◊⁺↦δ≥0.60; ◊↦δ≥0.40; ◊⁻↦δ≥0.20; ⊘↦δ<0.20
  𝕍≜Σ(ν:𝔹)(τ:◊)(δ:ℝ[0,1])(φ:Fin 101).(ν=⊤→τ≥◊⁻)
  𝔻oc≜Σ(b⃗:Vec n 𝔅)(π:Γ⊢wf(b⃗))
}

;; ─── Γ: INFERENCE RULES ───
⟦Γ:Inference⟧{
  d↓₁≡𝔸 ⊢ wf₁(d)                    ;; [ax-header]
  |b⃗|≥2 ⊢ wf₂(d)                     ;; [ax-blocks]
  wf₁(d) ∧ wf₂(d) ⊢ wf(d)            ;; [∧I-wf]
  ⊢wf(d) ∧ δ(d)≥¾ ⊢ d:◊⁺⁺            ;; [◊⁺⁺-I]
  ⊢wf(d) ∧ ⅗≤δ(d)<¾ ⊢ d:◊⁺           ;; [◊⁺-I]
  ⊢wf(d) ∧ ⅖≤δ(d)<⅗ ⊢ d:◊            ;; [◊-I]
  ⊢wf(d) ∧ ⅕≤δ(d)<⅖ ⊢ d:◊⁻           ;; [◊⁻-I]
  δ(d)<⅕ ∨ ¬wf(d) ⊢ d:⊘              ;; [⊘-I]
  Γ⊢d:τ ∧ τ≻τ' ⊢ Γ⊨d:τ'              ;; [sub]
}

;; ─── Λ: CORE FUNCTIONS ───
⟦Λ:Core⟧{
  ∂:𝕊→List⟨τ⟩; ∂≜fix λf s.s≡ε→[]|[hd s]⧺f(tl s)
  δ:List⟨τ⟩→ℝ[0,1]; δ≜λτ⃗.|{t∈τ⃗|t.k∈𝔄}|÷|{t∈τ⃗|t.k≢ws}|
  ⌈⌉:ℝ→◊; ⌈⌉≜λd.[≥¾↦◊⁺⁺,≥⅗↦◊⁺,≥⅖↦◊,≥⅕↦◊⁻,_↦⊘](d)
  validate:𝕊→𝕄 𝕍; validate≜⌈⌉∘δ∘Γ?∘∂
  cat:Σ_sym→Cat; cat≜λid.{c|c∈Cat∧id∈R[c]}
}

;; ─── Χ: ERROR ALGEBRA ───
⟦Χ:Errors⟧{
  ε≜Σ(ψ:𝔻oc→𝔹)(ρ:Πd:𝔻oc.ψ(d)=⊤→𝔻oc)
  ε_parse≜⟨parse_err(D),reject∧⊥⟩
  ε_ambig≜⟨Ambig(D)≥0.02,reject∧⊥⟩
  ε_token≜⟨|Tok(s)|>1,register(s)∨⊥⟩
  ε_H≜⟨¬(↓₁≡𝔸),λd.𝔸⊕d⟩
  ρ*:𝔻oc→𝔻oc; ρ*≜foldl(>=>)(pure){ρᵢ|ψᵢ=⊤}
}

;; ─── Σ: GRAMMAR ───
⟦Σ:Grammar⟧{
  Doc≜𝔸≫CTX?≫REF?≫⟦Ω⟧≫⟦Σ⟧≫⟦Γ⟧≫⟦Λ⟧≫⟦Χ⟧?≫⟦Ε⟧
  𝔸≜'𝔸'∘Ver∘'.'∘Name∘'@'∘Date
  Ver≜ℕ∘'.'∘ℕ; Date≜YYYY∘'-'∘MM∘'-'∘DD
  CTX≜'γ'∘'≔'∘Id; REF≜'ρ'∘'≔'∘⟨List⟩
  Block≜'⟦'∘Cat∘':'∘Name∘'⟧'∘'{'∘Body∘'}'
  Body≜(Stmt∘';'?)*; Stmt≜Def|Rule|Expr|';; '∘.*
  Def≜Sym∘('≜'|'≔')∘Expr; Rule≜Premise∘'⇒'∘Consequent
  Expr≜Lambda|Quant|Binary|Unary|Atom|Compound
  Lambda≜'λ'∘Params∘'.'∘Expr; Quant≜('∀'|'∃'|'∃!')∘Var∘':'∘Expr
  Prec≜[λ∀∃:1,→⇒↔:2,∨⋁:3,∧⋀:4,¬:5,≡≜∈⊆:6,⊕⊖:7,⊗×:8,∘:9,.:10]
  Assoc≜[→:right,∧∨:left,∘:right]
}

;; ─── Σ: TEMPLATE ───
⟦Σ:Template⟧{
  Minimal≜𝔸1.0.name@YYYY-MM-DD∘γ≔ctx∘⟦Ω⟧{inv}∘⟦Σ⟧{types}∘⟦Γ⟧{rules}∘⟦Λ⟧{funcs}∘⟦Ε⟧⟨δ≜N;φ≜N;τ≜◊X⟩
  Full≜𝔸X.Y.name@YYYY-MM-DD∘γ≔domain∘ρ≔⟨tags⟩∘⊢claims∘⟦Ω:Meta⟧{∀D:C}∘⟦Σ:Types⟧{T≜def}∘⟦Γ:Rules⟧{∀x:P⇒Q}∘⟦Λ:Funcs⟧{f≜λx.b}∘⟦Χ:Errors⟧{c⇒r}∘⟦Ε⟧⟨δ;φ;τ;⊢⟩
  Required≜{⟦Ω⟧,⟦Σ⟧,⟦Γ⟧,⟦Λ⟧,⟦Ε⟧}; Optional≜{⟦Χ⟧,⟦ℭ⟧,⟦ℜ⟧,⟦Θ⟧}
}

;; ─── Σ: ROSETTA STONE ───
⟦Σ:Rosetta⟧{
  "x defined as 5"↦x≜5
  "for all x in S,P"↦∀x∈S:P(x)
  "exists unique"↦∃!x:f(x)≡0
  "A implies B"↦A⇒B
  "f maps i to o"↦f:I→O,f≜λi.o
  "if A then B else C"↦A→B|C
  "not A"↦¬A
  "A and B"↦A∧B
  "A or B"↦A∨B
  "A equals B"↦A≡B
  "A is element of S"↦A∈S
  "A subset of B"↦A⊆B
  "empty set"↦∅
  "true"↦⊤
  "false"↦⊥
  "therefore"↦∴
  "QED"↦∎
  "const x=5"↦x≜5
  "S.every(x=>P(x))"↦∀x∈S:P(x)
  "if(A){B}"↦A⇒B
  "(x)=>y"↦λx.y
  Γ𝒫ℋℳ𝒩↦pocket
  ΓSignal↦V_H⊕V_L⊕V_S
}

;; ─── Γ: AGENT GUIDE ───
⟦Γ:Agent⟧{
  ∀agent:task∈{spec,instruct,coordinate}⇒output(AISP)
  ∀response:Ambig(response)<0.02∧δ≥0.40
  prose_only∧task(spec)⇒reject∧request(AISP)
  Output≜𝔸∘γ∘⟦Ω⟧∘⟦Σ⟧∘⟦Γ⟧∘⟦Λ⟧∘⟦Ε⟧

  ∀s∈Σ_512:Mean(s)≡Mean_0(s)
  ∀D:Ambig(D)<0.02
  ¬(prose≈AISP); ¬(∃s:Mean(s,ctx1)≠Mean(s,ctx2))
  drift_detected⇒reparse(original); ambiguity_detected⇒reject∧clarify

  Vision≜"AISP:Assembly for AI cognition"
  ∀prose:Ambig∈[0.40,0.65]; ∀code:Ambig∈[0.05,0.15]; ∀AISP:Ambig<0.02
  ⊢deterministic:∀D:∃!AST.parse(D)→AST
  ⊢proof-carrying:𝔻oc≜Σ(content)(π:Γ⊢wf)
  ⊢lossless:∀L:Signal(L)≡L
  ⊢self-certifying:⟦Ε⟧∈every(D)
}

;; ─── Σ: QUICK REFERENCE ───
⟦Σ:QuickRef⟧{
  Core≜{≜:def,≔:assign,≡:identical,⇒:implies,↔:iff,∀:all,∃:exists,∃!:unique,∈:elem,⊆:subset,∧:and,∨:or,¬:not,⊤:true,⊥:false,λ:lambda,∘:compose,→:func,↦:mapsto,⟨⟩:tuple,⟦⟧:block,𝒫:pocket,∅:empty}
  Tiers≜{◊⁺⁺:δ≥0.75,◊⁺:δ≥0.60,◊:δ≥0.40,◊⁻:δ≥0.20,⊘:δ<0.20}
  Binding≜{⊤:3:zero,λ:2:adapt,∅:1:null,⊥:0:crash}
  Blocks≜{⟦Ω⟧:meta,⟦Σ⟧:types,⟦Γ⟧:rules,⟦Λ⟧:funcs,⟦Χ⟧:errors,⟦Ε⟧:evidence}
}

;; ─── CONVERSION TASK ───
⟦Γ:Task⟧{
  ∀input:prose→AISP
  ∀output:AISP_only∧¬explain∧¬prose
  ∀sym:∈Σ_512∧¬hallucinate
  semantic≜preserved
  tier∈{minimal,standard,full}
}

⟦Ε⟧⟨δ≜0.81;|𝔅|≜18/18;φ≜98;τ≜◊⁺⁺;⊢ND;⊢CAT;⊢ΠΣ;⊢Ambig<0.02⟩"#.to_string()
});

/// Get cached system prompt based on style
fn system_prompt(use_aisp: bool) -> &'static str {
    if use_aisp {
        &AISP_PROMPT
    } else {
        &ENGLISH_PROMPT
    }
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
    /// Create new Claude fallback with default model (haiku for speed)
    pub fn new() -> Self {
        Self {
            model: "haiku".to_string(),
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
        use_aisp_prompt: bool,
    ) -> Result<LlmResult> {
        use claude_agent_sdk_rs::{
            query, ClaudeAgentOptions, ContentBlock, McpServers, Message, PermissionMode,
            SettingSource,
        };
        use std::collections::HashMap;

        let user_prompt = create_user_prompt(prose, tier, unmapped, partial_output);

        // Build extra args for minimal CLI invocation
        let mut extra_args: HashMap<String, Option<String>> = HashMap::new();
        extra_args.insert("no-chrome".to_string(), None);
        extra_args.insert("no-session-persistence".to_string(), None);
        extra_args.insert("disable-slash-commands".to_string(), None);
        extra_args.insert("strict-mcp-config".to_string(), None);

        // Configure minimal Claude instance - no plugins, no MCP, no settings
        let options = ClaudeAgentOptions::builder()
            .model(&self.model)
            .system_prompt(system_prompt(use_aisp_prompt).to_string())
            .max_turns(1) // Single turn for conversion
            .permission_mode(PermissionMode::BypassPermissions)
            .tools(Vec::<String>::new()) // No tools needed
            .mcp_servers(McpServers::Empty) // No MCP servers
            .setting_sources(Vec::<SettingSource>::new()) // No filesystem settings
            .plugins(Vec::new()) // No plugins
            .skip_version_check(true) // Skip version check for speed
            .fork_session(true) // Fresh session, no history loading
            .extra_args(extra_args) // Minimal CLI flags
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
    fn test_english_prompt_generation() {
        let prompt = system_prompt(false);
        assert!(prompt.contains("AISP"));
        assert!(prompt.contains("Rosetta Stone"));
        assert!(prompt.contains("Σ_512"));
        assert!(prompt.contains("Ambig(D) < 0.02"));
        // Full specification should be substantial
        assert!(prompt.len() > 3000);
    }

    #[test]
    fn test_aisp_prompt_generation() {
        let prompt = system_prompt(true);
        assert!(prompt.contains("𝔸5.1"));
        assert!(prompt.contains("⟦Σ:Glossary⟧"));
        assert!(prompt.contains("⟦Σ:Rosetta⟧"));
        assert!(prompt.contains("⟦Γ:Agent⟧"));
        // Full specification should be substantial
        assert!(prompt.len() > 3000);
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
