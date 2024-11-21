use std::{hint::black_box, vec};

use llguidance_parser::{
    api::{GrammarWithLexer, ParserLimits, TopLevelGrammar},
    toktrie::{InferenceCapabilities, TokEnv, TokenId},
    Constraint, GrammarBuilder, TokenParser,
};

use lazy_static::lazy_static;

/// Check that the grammar generates the expected output.
///
/// Output is a list of strings, each of which is a sequence of tokens.
/// Tokens in the string are separated with "‧".
/// Strings at even positions are "forced tokens", and strings at odd positions
/// are "generated tokens".
/// We check that the grammars forces the forced tokens (first of which is the
/// prompt), and that it allows in the mask the generated tokens.
///
/// These tests are "recorded" by passing "test_trace": true in the llguidance
/// request and post-processing.
fn check_grammar(tok_env: &TokEnv, grammar: TopLevelGrammar, output: &[&str]) {
    println!("\nChecking grammar");

    let parser = TokenParser::from_llguidance_json(
        tok_env.clone(),
        grammar,
        llguidance_parser::Logger::new(0, 2),
        InferenceCapabilities {
            ff_tokens: true, // can the engine append multiple tokens?
            backtrack: true, // can the engine remove generated tokens?

            conditional_ff_tokens: false, // not used
            fork: false,                  // not used
        },
        ParserLimits::default(),
        vec![],
    )
    .unwrap();
    let mut constraint = Constraint::new(parser);

    let prompt = constraint.process_prompt(tok_env.tokenize(""));
    check_eq(tok_env, "prompt", &prompt, output[0]);

    let mut idx = 1;
    let mut gen_tokens = tokenize_trace(tok_env, output[idx]);

    for _ in 0..200 {
        let res = constraint.compute_mask().unwrap();

        if res.is_stop() {
            assert!(idx >= output.len() - 1, "Expected more output at {}", idx);
            assert!(gen_tokens.is_empty(), "Expected more tokens to generate");
            break;
        }

        let mut bt: u32;
        let mut toks: Vec<TokenId>;

        if let Some(mask) = &res.sample_mask {
            if gen_tokens.is_empty() {
                panic!("No more tokens to generate");
            }
            let tok = gen_tokens.remove(0);
            assert!(mask.is_allowed(tok), "Token {} not allowed", tok);
            let res = constraint.commit_token(Some(tok)).unwrap();
            bt = res.backtrack;
            toks = res.ff_tokens.clone();
            if toks.is_empty() || toks[0] != tok {
                if output[idx + 1].starts_with("1↶") {
                    assert!(bt == 0 || res.ff_tokens.is_empty());
                    bt = 1;
                } else {
                    panic!("Expected token {} got {}", tok, toks[0]);
                }
            } else if toks.len() > 1 {
                toks.remove(0);
            } else {
                assert!(bt == 0);
                assert!(toks.len() == 1);
                continue;
            }
        } else {
            let res = constraint.commit_token(None).unwrap();
            bt = res.backtrack;
            toks = res.ff_tokens.clone();
        }

        idx += 1;
        let mut expected = output[idx];
        if expected.contains("↶") {
            let parts: Vec<&str> = expected.split("↶").collect();
            assert!(parts.len() == 2);
            expected = parts[1];
            assert!(
                bt == parts[0].parse::<u32>().unwrap(),
                "Expected backtrack {} got {}",
                parts[0],
                bt
            );
        }
        check_eq(tok_env, &format!("step {}", idx), &toks, expected);
        idx += 1;
        if idx < output.len() {
            gen_tokens = tokenize_trace(tok_env, output[idx]);
        }
    }
}

fn check_eq(tok_env: &TokEnv, label: &str, tokens: &[TokenId], expected_tokens: &str) {
    println!("Checking {}: {:?}", label, expected_tokens);
    let trie = tok_env.tok_trie();
    let actual_tokens = trie.test_trace_tokens(tokens);
    assert_eq!(
        actual_tokens, expected_tokens,
        "Tokens mismatch in {}\n  {}\n  {}",
        label, actual_tokens, expected_tokens
    );
}

fn tokenize_trace(tok_env: &TokEnv, s: &str) -> Vec<TokenId> {
    let trie = tok_env.tok_trie();
    println!("Tokenizing {:?}", s);
    let mut result = Vec::new();
    for word in s.split("‧") {
        if word == "≺EOS≻" {
            result.push(trie.eos_token());
            continue;
        }
        let tt = trie.greedy_tokenize(word.as_bytes());
        assert!(
            tt.len() == 1,
            "Expected single token for {:?} got {:?}",
            word,
            tt
        );
        result.push(tt[0]);
    }
    result
}

lazy_static! {
    static ref TOK_ENV: TokEnv = {
        toktrie_hf_tokenizers::ByteTokenizerEnv::from_name("microsoft/Phi-3.5-mini-instruct", None)
            .unwrap()
            .to_env()
    };
}

fn check_lark_grammar(lark: &str, output: &[&str]) {
    let grm = TopLevelGrammar::from_lark(lark.to_string());
    check_grammar(&TOK_ENV, grm, output);
}

fn test_llparser() {
    check_lark_grammar(
        r#"
            start: "Q: Are dolphins fish?\nA: " ANSWER "\nQ: Are sharks fish?\nA: " ANSWER
            ANSWER: "Yes" | "No"
        "#,
        &[
            "Q‧:‧ Are‧ dol‧ph‧ins‧ fish‧?‧\n‧A‧:",
            " No", // note the prefix space - moved by token healing
            "\n‧Q‧:‧ Are‧ sh‧arks‧ fish‧?‧\n‧A‧:",
            " Yes",
        ],
    );

    check_lark_grammar(
        r#"
            start: "Power frequency is " NUMBER "Hz; voltage is " NUMBER "V"
            NUMBER: /[0-9]+/
        "#,
        &[
            "Power‧ frequency‧ is‧ ",
            "5‧0‧Hz", // no EoS needed on 50Hz
            ";‧ voltage‧ is‧ ",
            "2‧2‧0‧V",
        ],
    );

    check_lark_grammar(
        r#"
            start: "Q: 7 * 8\nA: " NUMBER
            NUMBER: /[0-9]+/
        "#,
        &["Q‧:‧ ‧7‧ *‧ ‧8‧\n‧A‧:‧ ", "5‧6‧≺EOS≻"],
    );
}

fn main() {
    test_llparser();

    let mut builder = GrammarBuilder::new();

    builder.add_grammar(GrammarWithLexer::default());
    let n0 = builder.gen_rx(".*", "\n");
    let n1 = builder.string("\n");
    let n2 = builder.join(&[n0, n1]);
    builder.set_start_node(n2);

    let grammar = builder.finalize().unwrap();
    let obj_str = "this is\na test";

    let tok_env: TokEnv =
        toktrie_hf_tokenizers::ByteTokenizerEnv::from_name("microsoft/Phi-3.5-mini-instruct", None)
            .unwrap()
            .to_env();

    let tokens = tok_env.tokenize(&obj_str);

    let stderr_log_level = 2;
    let buffer_log_level = 0;

    let parser = TokenParser::from_llguidance_json(
        tok_env.clone(),
        grammar,
        llguidance_parser::Logger::new(buffer_log_level, stderr_log_level),
        InferenceCapabilities {
            ff_tokens: true,  // can the engine append multiple tokens?
            backtrack: false, // can the engine remove generated tokens?

            conditional_ff_tokens: false, // not used
            fork: false,                  // not used
        },
        ParserLimits::default(),
        vec![],
    )
    .unwrap();
    let mut constraint = Constraint::new(parser);

    // enable sending parser results back via the logs (constraint.flush_logs())
    constraint.log_json_progress = true;

    let trie = tok_env.tok_trie();

    eprintln!("Parsing tokens: {}", trie.tokens_dbg(&tokens));

    let mut idx = 0;
    while idx < tokens.len() {
        let res = constraint.compute_mask().unwrap();

        if res.is_stop() {
            // stop sequence
            break;
        }

        let sampled_token = if let Some(mask) = &res.sample_mask {
            // Simulate sampling - it should use the mask and temperature
            black_box(mask);
            black_box(constraint.temperature);
            let sampled_token = tokens[idx];

            let p_stats = constraint.parser.last_step_stats();
            println!(
                "SAMPLE {}: {} {}; stats: {} lex, {} items, {} us",
                idx,
                sampled_token,
                tok_env.tok_trie().token_dbg(sampled_token),
                p_stats.lexer_cost,
                p_stats.all_items,
                p_stats.compute_time_us,
            );
            Some(sampled_token)
        } else {
            // sampling not required
            println!("NO SAMPLE");
            None
        };

        let splice = constraint.commit_token(sampled_token).unwrap();
        if splice.stop {
            // stop sequence
            break;
        }

        assert!(splice.backtrack == 0); // we didn't allow backtracking in InferenceCaps

        // if this fails, our test data is broken
        if tokens[idx..idx + splice.ff_tokens.len()] != splice.ff_tokens {
            panic!(
                "BAD TEST: ff_tokens mismatch:\n{}\n{}",
                trie.tokens_dbg(&tokens[idx..idx + splice.ff_tokens.len()]),
                trie.tokens_dbg(&splice.ff_tokens)
            );
        }

        if splice.ff_tokens.len() > 1 {
            println!("FF: {}", trie.tokens_dbg(&splice.ff_tokens));
        }

        idx += splice.ff_tokens.len();
    }

    // the stop reason should be likely also sent to the user
    println!("Stop reason: {:?}", constraint.parser.stop_reason());

    println!("Max step stats: {:?}", constraint.parser.max_step_stats());
}
