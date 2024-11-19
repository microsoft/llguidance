use std::{env, fs::File, io::Read, vec};

use llguidance_parser::{
    api::ParserLimits,
    toktrie::{InferenceCapabilities, TokEnv},
    Constraint, JsonCompileOptions, TokenParser,
};
use serde_json::Value;

fn test_file(tok_env: TokEnv, file: &str) {
    let schema_file = read_file_to_string(file);
    let opts = JsonCompileOptions::default();
    let val: Value = serde_json::from_str(&schema_file).expect("Invalid JSON in schema");

    if schema_file.len() < 512 && val["$ref"].is_string() {
        eprintln!("{} ref-only", file);
        return;
    }

    let schema = opts.json_to_llg(&val);

    let schema = match schema {
        Ok(schema) => schema,
        Err(e) => {
            eprintln!("{} Error: {}", file, e);
            return;
        }
    };

    let parser = TokenParser::from_llguidance_json(
        tok_env,
        schema,
        llguidance_parser::Logger::new(0, 1),
        InferenceCapabilities {
            ff_tokens: true,
            backtrack: false,
            conditional_ff_tokens: false,
            fork: false,
        },
        ParserLimits::default(),
        vec![],
    );

    match parser {
        Ok(parser) => {
            let mut constraint = Constraint::new(parser);
            constraint.compute_mask().unwrap();
            eprintln!("{} OK", file);
        }
        Err(e) => {
            eprintln!("{} Error: {}", file, e);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <json-schema.json|folder>...", args[0]);
        std::process::exit(1);
    }

    let mut files = vec![];
    for arg in &args[1..] {
        if arg.ends_with(".json") {
            files.push(arg.to_string());
        } else {
            let dir = std::fs::read_dir(arg).expect("Unable to read directory");
            for entry in dir {
                let entry = entry.expect("Unable to read entry");
                let path = entry.path();
                if path.is_file() && path.to_str().unwrap().ends_with(".json") {
                    files.push(path.to_str().unwrap().to_string());
                }
            }
        }
    }

    let tok_env: TokEnv =
        toktrie_hf_tokenizers::ByteTokenizerEnv::from_name("microsoft/Phi-3.5-mini-instruct", None)
            .unwrap()
            .to_env();

    for file in files {
        test_file(tok_env.clone(), &file);
    }
}

fn read_file_to_string(filename: &str) -> String {
    let mut file = File::open(filename).expect("Unable to open file");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Unable to read file");
    content
}
