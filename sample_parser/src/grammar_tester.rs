use std::{hint::black_box, vec};

use llguidance_parser::{
    api::{GrammarWithLexer, ParserLimits},
    toktrie::{InferenceCapabilities, TokEnv},
    Constraint, GrammarBuilder, TokenParser,
};

fn main() {
    let mut builder = GrammarBuilder::new();

    builder.add_grammar(GrammarWithLexer::default());
    let n0 = builder.gen_rx(".*", "\n");
    let n1 = builder.string("\n");
    let n2 = builder.join(&[n0, n1]);
    builder.set_start_node(n2);

    let schema = builder.finalize().unwrap();
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
        schema,
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
