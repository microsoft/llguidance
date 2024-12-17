use lazy_static::lazy_static;
use llguidance::{
    api::{GrammarWithLexer, ParserLimits, TopLevelGrammar},
    toktrie::{InferenceCapabilities, TokEnv, TokenId},
    Constraint, TokenParser,
};

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
fn check_grammar(
    tok_env: &TokEnv,
    prompt_str: &str,
    grammar: TopLevelGrammar,
    output: &[&str],
    temp: f32,
) -> Constraint {
    let parser = TokenParser::from_llguidance_json(
        tok_env.clone(),
        grammar,
        llguidance::Logger::new(0, 2),
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

    let prompt = constraint.process_prompt(tok_env.tokenize(prompt_str));
    check_eq(tok_env, "prompt", &prompt, output[0]);

    let mut idx = 1;
    let mut gen_tokens = tokenize_trace(tok_env, output[idx]);
    let mut seen_temp = temp == 0.0;

    for _ in 0..200 {
        let res = constraint.compute_mask().unwrap();

        if let Some(t) = res.temperature {
            assert!(
                t == temp || t == 0.0,
                "Expected temperature {} got {}",
                temp,
                t
            );
            if t == temp {
                seen_temp = true;
            }
        }

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

            let tok = gen_tokens[0];
            assert!(
                mask.is_allowed(tok),
                "Token {} {} not allowed",
                tok,
                tok_env.tok_trie().token_dbg(tok)
            );

            let num_ok = constraint.validate_tokens_raw(&gen_tokens).unwrap();
            if num_ok < gen_tokens.len() {
                panic!(
                    "Expected {} tokens to be allowed; got {}; {}",
                    gen_tokens.len(),
                    num_ok,
                    tok_env.tok_trie().tokens_dbg(&gen_tokens)
                );
            }
            gen_tokens.remove(0);

            let res = constraint.commit_token(Some(tok)).unwrap();
            bt = res.backtrack;
            toks = res.ff_tokens.clone();
            if toks.is_empty() || toks[0] != tok {
                if idx + 1 < output.len() && output[idx + 1].starts_with("1↶") {
                    // fast-forward with fake backtrack
                    assert!(bt == 0 || res.ff_tokens.is_empty());
                    bt = 1;
                    // go to forced byte checking
                } else {
                    if toks.is_empty() {
                        panic!("Expected {}; got nothing", tok);
                    } else {
                        panic!("Expected token {} got {}", tok, toks[0]);
                    }
                }
            } else if toks.len() > 1 {
                // we got fast-forwarded to the next entry,
                // delete the generated tokens and leave the rest for forced
                // bytes checking below
                toks.remove(0);
                // go to forced byte checking
            } else {
                assert!(bt == 0);
                assert!(toks.len() == 1);
                continue; // normal path
            }
        } else {
            let res = constraint.commit_token(None).unwrap();
            bt = res.backtrack;
            toks = res.ff_tokens.clone();
        }

        // forced byte checking
        assert!(gen_tokens.is_empty(), "Expected more tokens to generate");

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

    assert!(seen_temp, "Expected temperature {} not seen", temp);

    constraint
}

fn check_eq(tok_env: &TokEnv, label: &str, tokens: &[TokenId], expected_tokens: &str) {
    let trie = tok_env.tok_trie();
    let actual_tokens = trie.test_trace_tokens(tokens);
    println!(
        "Checking {}: exp:{:?} got:{:?}",
        label, expected_tokens, actual_tokens
    );
    assert_eq!(
        actual_tokens, expected_tokens,
        "Tokens mismatch in {}",
        label
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
            "Expected single token for {:?} got {}",
            word,
            trie.test_trace_tokens(&tt)
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

pub fn check_lark_grammar_prompt(lark: &str, prompt_str: &str, output: &[&str]) -> Constraint {
    let grm = TopLevelGrammar::from_lark(lark.to_string());
    println!("\nChecking grammar:\n{}\nagainst: {:?}", lark, output);
    check_grammar(&TOK_ENV, prompt_str, grm, output, 0.0)
}

pub fn check_lark_grammar(lark: &str, output: &[&str]) -> Constraint {
    check_lark_grammar_prompt(lark, "", output)
}

pub fn check_lark_grammar_nested(lark: &str, sub_lark: &str, output: &[&str]) -> Constraint {
    let temp = lark
        .find("temperature=")
        .map(|i| {
            let i = i + "temperature=".len();
            let mut end = i;
            while end < lark.len()
                && (lark.as_bytes()[end].is_ascii_digit() || lark.as_bytes()[end] == b'.')
            {
                end += 1;
            }
            lark[i..end].parse::<f32>().unwrap()
        })
        .unwrap_or(0.0);
    let mut top_grm = TopLevelGrammar::from_lark(lark.to_string());
    let mut sub_grm = GrammarWithLexer::from_lark(sub_lark.to_string());
    sub_grm.name = Some("sub".to_string());
    top_grm.grammars.push(sub_grm);
    println!(
        "\nChecking nested grammars:\n{}\nNested:\n{}\nagainst: {:?}",
        lark, sub_lark, output
    );
    check_grammar(&TOK_ENV, "", top_grm, output, temp)
}

pub fn check_lark_json(lark: &str, json_schema: serde_json::Value, output: &[&str]) -> Constraint {
    let schema_str = serde_json::to_string_pretty(&json_schema).unwrap();
    let mut top_grm = TopLevelGrammar::from_lark(lark.to_string());
    let mut sub_grm = GrammarWithLexer::from_json_schema(json_schema);
    sub_grm.name = Some("sub".to_string());
    top_grm.grammars.push(sub_grm);
    println!(
        "\nChecking lark+json:\n{}\nNested:\n{}\nagainst: {:?}",
        lark, schema_str, output
    );
    check_grammar(&TOK_ENV, "", top_grm, output, 0.0)
}

pub fn check_capture(c: &Constraint, name: &str, expected: &str) {
    if let Some(bytes) = c.parser.get_capture(name) {
        let actual = String::from_utf8_lossy(bytes);
        assert_eq!(actual, expected, "Capture {} mismatch", name);
    } else {
        panic!("Capture {} not found", name);
    }
}

pub fn print_tokenized(s: &str) {
    let trie = TOK_ENV.tok_trie();
    let tokens = TOK_ENV.tokenize(s);
    println!("{:?}", trie.test_trace_tokens(&tokens));
}
