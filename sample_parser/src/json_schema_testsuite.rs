use anyhow::{bail, Result};
use core::str;
use llguidance::{
    api::{ParserLimits, TopLevelGrammar},
    toktrie::{InferenceCapabilities, TokEnv},
    Constraint, JsonCompileOptions, TokenParser,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{env, fs::File, io::Read, vec};

#[derive(Debug, Serialize, Deserialize)]
struct JsonTest {
    description: String,
    schema: Value,
    tests: Vec<JsonTestSequence>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonTestSequence {
    description: String,
    data: Value,
    valid: bool,
}

impl JsonTestSequence {
    fn run_for(&self, obj_str: &str, tok_env: &TokEnv, mut constraint: Constraint) -> Result<()> {
        let tokens = tok_env.tokenize(obj_str);
        let trie = tok_env.tok_trie();

        let mut idx = 0;
        while idx < tokens.len() {
            // println!("idx: {} {}", idx, trie.token_dbg(tokens[idx]));

            let res = constraint.compute_mask()?;

            if res.is_stop() {
                if self.valid {
                    bail!("premature stop in valid test");
                } else {
                    bail!("premature stop in invalid test"); // ??
                }
            }

            let sampled_token = if let Some(mask) = &res.sample_mask {
                let sampled_token = tokens[idx];
                if !mask.is_allowed(sampled_token) {
                    if self.valid {
                        bail!(
                            "sampled token {} not allowed by mask",
                            trie.token_dbg(sampled_token)
                        );
                    } else {
                        return Ok(());
                    }
                }

                // let p_stats = constraint.parser.last_step_stats();
                Some(sampled_token)
            } else {
                None
            };

            let splice = constraint.commit_token(sampled_token)?;
            if splice.stop {
                if self.valid {
                    if idx + 1 < tokens.len() {
                        bail!("premature stop in valid test (commit)");
                    } else {
                        return Ok(());
                    }
                } else {
                    bail!("premature stop in invalid test (commit)"); // ??
                }
            }

            assert!(splice.backtrack == 0); // we didn't allow backtracking in InferenceCaps

            if tokens[idx..idx + splice.ff_tokens.len()] != splice.ff_tokens {
                bail!(
                    "BAD TEST: ff_tokens mismatch:\n{}\n{}",
                    trie.tokens_dbg(&tokens[idx..idx + splice.ff_tokens.len()]),
                    trie.tokens_dbg(&splice.ff_tokens)
                );
            }

            idx += splice.ff_tokens.len();
        }

        let accept = constraint.parser.is_accepting();

        if self.valid {
            if accept {
                Ok(())
            } else {
                bail!("unexpected end of test");
            }
        } else {
            bail!(
                "unexpected end of test for invalid test (accept={})",
                accept
            );
        }
    }

    fn run(&self, grm: &TopLevelGrammar, tok_env: &TokEnv) -> Result<()> {
        let stderr_log_level = 1;
        let buffer_log_level = 0;
        let parser = TokenParser::from_llguidance_json(
            tok_env.clone(),
            grm.clone(),
            llguidance::Logger::new(buffer_log_level, stderr_log_level),
            InferenceCapabilities {
                ff_tokens: false, // can the engine append multiple tokens?
                backtrack: false, // can the engine remove generated tokens?

                conditional_ff_tokens: false, // not used
                fork: false,                  // not used
            },
            ParserLimits::default(),
            vec![],
        )?;
        let constraint = Constraint::new(parser);

        let obj_str = serde_json::to_string_pretty(&self.data).unwrap();
        match self.run_for(&obj_str, tok_env, constraint) {
            Ok(_) => Ok(()),
            Err(e) => {
                bail!("{}\n{:?}", e, obj_str)
            }
        }
    }
}

impl JsonTest {
    fn run(&self, tok_env: &TokEnv) -> Result<()> {
        let opts = JsonCompileOptions::default();
        let grm = opts.json_to_llg(self.schema.clone())?;
        let mut first_err = Ok(());
        for t in &self.tests {
            let r = t.run(&grm, tok_env);
            if first_err.is_ok() && r.is_err() {
                first_err = r;
            }
        }
        first_err
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <file.json...>", args[0]);
        std::process::exit(1);
    }

    let tok_env: TokEnv =
        toktrie_hf_tokenizers::ByteTokenizerEnv::from_name("meta-llama/Llama-3.2-1B", None)
            .unwrap()
            .to_env();

    let t0 = std::time::Instant::now();
    for arg in &args[1..] {
        let schema_file = read_file_to_string(arg);
        let val: Vec<JsonTest> =
            serde_json::from_str(&schema_file).expect("Invalid JSON in schema");
        for (idx, t) in val.iter().enumerate() {
            print!("Running test: {} ({}) #{} ", arg, t.description, idx);
            match t.run(&tok_env) {
                Ok(_) => println!("OK"),
                Err(e) => println!("ERROR: {}", e),
            }
        }
    }

    let elapsed = t0.elapsed();
    println!("Total time: {} ms", elapsed.as_millis());
}

fn read_file_to_string(filename: &str) -> String {
    let mut file = File::open(filename).expect("Unable to open file");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Unable to read file");
    content
}
