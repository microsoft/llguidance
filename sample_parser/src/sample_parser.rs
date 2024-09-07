use std::{env, fs::File, hint::black_box, io::Read, vec};

use llguidance_parser::{
    api::{ParserLimits, TopLevelGrammar},
    toktrie::{InferenceCapabilities, TokEnv},
    Constraint, TokenParser,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <schema.ll.json> <sample.json>", args[0]);
        std::process::exit(1);
    }

    let schema_file = read_file_to_string(&args[1]);
    let schema: TopLevelGrammar =
        serde_json::from_str(&schema_file).expect("Invalid JSON in schema");
    let obj_str = read_file_to_string(&args[2]);

    // you can implement TokEnv yourself, if you have the tokenizer
    // see the ByteTokenizerEnv for an example
    let tok_env: TokEnv =
        toktrie_hf_tokenizers::ByteTokenizerEnv::from_name("microsoft/Phi-3.5-mini-instruct", None)
            .unwrap()
            .to_env();

    let tokens = tok_env.tokenize(&obj_str);

    // set to 2 for more output; 1 is warnings only
    let stderr_log_level = 1;

    // typically set to 2, to send info-level output to the user
    let buffer_log_level = 2;

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

            println!(
                "SAMPLE {}: {} {}",
                idx,
                sampled_token,
                tok_env.tok_trie().token_dbg(sampled_token)
            );
            Some(sampled_token)
        } else {
            // sampling not required
            println!("NO SAMPLE");
            None
        };

        let splice = constraint.commit_token(sampled_token).unwrap();
        if splice.is_stop() {
            // stop sequence
            break;
        }

        let splice = splice.unconditional_splice().unwrap();

        assert!(splice.backtrack == 0); // we didn't allow backtracking in InferenceCaps

        // The splice contains the tokens (possibly more than one since we enabled ff_tokens
        // in InferenceCaps) that the parser wants to append to the output.

        // if this fails, our test data is broken
        assert!(
            tokens[idx..idx + splice.ff_tokens.len()] == splice.ff_tokens,
            "BAD TEST: ff_tokens mismatch"
        );

        if splice.ff_tokens.len() > 1 {
            println!("FF: {}", trie.tokens_dbg(&splice.ff_tokens));
        }

        idx += splice.ff_tokens.len();

        // send output to the user
        send_output(&constraint.flush_logs());
    }

    // flush any output
    send_output(&constraint.flush_logs());
    // the stop reason should be likely also sent to the user
    println!("Stop reason: {:?}", constraint.parser.stop_reason());
}

fn read_file_to_string(filename: &str) -> String {
    let mut file = File::open(filename).expect("Unable to open file");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Unable to read file");
    content
}

fn send_output(user_output: &str) {
    // enable if you want to see the output
    if false {
        println!("{}", user_output);
    }
}
