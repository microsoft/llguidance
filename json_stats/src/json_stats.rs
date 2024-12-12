use anyhow::{anyhow, bail, Result};
use clap::Parser;
use json_stats::SchemaStats;
use jsonschema::Validator;
use llguidance::{
    api::ParserLimits,
    toktrie::{InferenceCapabilities, TokEnv},
    Constraint, JsonCompileOptions, TokenParser,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    sync::Arc,
};

use rayon::prelude::*;

struct DummyResolver {}
impl jsonschema::Retrieve for DummyResolver {
    fn retrieve(
        &self,
        uri: &jsonschema::Uri<&str>,
    ) -> std::result::Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err(anyhow!("external resolver disabled (url: {})", uri).into())
    }
}

#[derive(Parser, Debug, Default)]
#[command(version, about, long_about = None)]
pub struct CliOptions {
    #[arg(long, short = 'c')]
    llg_compile: bool,

    #[arg(long, short = 't')]
    llg_test: bool,

    #[arg(long)]
    remove_broken_tests: bool,

    #[arg(long)]
    skip_synth: bool,

    #[arg(long)]
    additional_features: bool,

    // .json files or folders with .json files
    #[arg(value_name = "FILES")]
    files: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct LlgResult {
    id: String,
    ttfm_us: usize,
    num_valid: usize,
    num_invalid: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    compile_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parser_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonTest {
    description: String,
    meta: Option<JsonMetaInfo>,
    schema: Value,
    tests: Vec<JsonTestSequence>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct JsonMetaInfo {
    pub full_size: usize,
    pub stripped_size: usize,
    pub features: Vec<String>,
    #[serde(default)]
    pub raw_features: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct JsonFileInfo {
    pub id: String,
    pub meta: JsonMetaInfo,
    pub num_valid_tests: usize,
    pub num_invalid_tests: usize,
    pub size_valid_tests: usize,
    pub size_invalid_tests: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonTestSequence {
    description: String,
    valid: bool,
    #[serde(skip)]
    broken: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    rust_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    python_error: Option<String>,
    data: Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SchemaRes {
    file_name: String,
    full_size: usize,
    ttfm_us: usize,
    file_info: JsonFileInfo,
    llg_result: Option<LlgResult>,

    test_valid_error: Option<String>,
    test_invalid_error: Option<String>,
    schema_error: Option<String>,
}

#[derive(Clone)]
struct TestEnv {
    cli: Arc<CliOptions>,
    tok_env: TokEnv,
    file_name: String,
}

impl TestEnv {
    fn run_llg_test(&self, parser: &TokenParser, t: &JsonTestSequence) -> Result<()> {
        let mut parser = parser.clone();
        parser.start_without_prompt();

        let dstr = serde_json::to_string(&t.data).unwrap();
        let tokens = self.tok_env.tokenize(&dstr);
        let trie = self.tok_env.tok_trie();

        for (tidx, &token) in tokens.iter().enumerate() {
            let ok = parser.validate_token(token)?;
            if !ok {
                if t.valid {
                    bail!(
                        "token not accepted at {}",
                        trie.tokens_dbg(&tokens[0..tidx + 1])
                    )
                } else {
                    return Ok(());
                }
            }
            let bt = parser.consume_token(token)?;
            assert!(bt == 0);
        }

        if parser.is_accepting() {
            if !t.valid {
                bail!("incorrect accept");
            }
        } else {
            if t.valid {
                bail!("parser not accepting at the end");
            }
        }

        Ok(())
    }

    fn run_llg_compile(&self, test_file: &JsonTest) -> LlgResult {
        let opts = JsonCompileOptions::default();
        let mut res = LlgResult::default();

        let t0 = std::time::Instant::now();
        let schema = opts.json_to_llg(test_file.schema.clone());

        let schema = match schema {
            Ok(schema) => schema,
            Err(e) => {
                res.compile_error = Some(format!("{e}"));
                limit_string(&mut res.compile_error);
                // eprintln!("{} Error Compile: {}", file, e);
                return res;
            }
        };

        let parser = TokenParser::from_llguidance_json(
            self.tok_env.clone(),
            schema,
            llguidance::Logger::new(0, 0),
            InferenceCapabilities {
                ff_tokens: false,
                backtrack: false,
                conditional_ff_tokens: false,
                fork: false,
            },
            ParserLimits::default(),
            vec![],
        );

        let parser = match parser {
            Ok(parser) => {
                let mut constraint = Constraint::new(parser.clone());
                constraint.compute_mask().unwrap();
                res.ttfm_us = t0.elapsed().as_micros() as usize;
                parser
                // eprintln!("{} OK", file);
            }
            Err(e) => {
                // eprintln!("{} Error Parser: {}", self.file_name, e);
                res.parser_error = Some(format!("{e}"));
                limit_string(&mut res.parser_error);
                return res;
            }
        };

        if self.cli.llg_test {
            for (idx, t) in test_file.tests.iter().enumerate() {
                if let Err(e) = self.run_llg_test(&parser, t) {
                    res.validation_error = Some(format!("test #{idx}: {e}"));
                    limit_string(&mut res.validation_error);
                } else {
                    if t.valid {
                        res.num_valid += 1;
                    } else {
                        res.num_invalid += 1;
                    }
                }
            }
        }

        res
    }

    fn run_test(&self) -> SchemaRes {
        let file_name = &self.file_name;
        let schema_file = read_file_to_string(file_name);
        let mut test_file: JsonTest = serde_json::from_str(&schema_file)
            .expect(format!("Invalid JSON in schema file {}", file_name).as_str());

        let mut stats = SchemaRes::default();
        stats.file_name = file_name.clone();
        stats.full_size = serde_json::to_string(&test_file.schema).unwrap().len();

        let uuid_regex = regex::Regex::new(r"^(?P<time_low>[0-9a-fA-F]{8})-(?P<time_mid>[0-9a-fA-F]{4})-(?P<time_high_and_version>[0-9a-fA-F]{4})-(?P<clock_seq_and_reserved>[0-9a-fA-F]{2})(?P<clock_seq_low>[0-9a-fA-F]{2})-(?P<node>[0-9a-fA-F]{12})$"
    ).unwrap();
        let iri_regex = regex::Regex::new(
        r"^(?P<scheme>[A-Za-z][A-Za-z0-9+\-.]*):(?:\/\/(?P<authority>[^\s/?#]+))?(?P<path>[^\s?#]*)(?:\?(?P<query>[^\s#]*))?(?:#(?P<fragment>\S*))?$"
    ).unwrap();
        let duration_regex = regex::Regex::new(
        r"^P(?:(?P<dur_date>(?:(?P<dur_year>[0-9]+Y(?:[0-9]+M(?:[0-9]+D)?)?)|(?P<dur_month>[0-9]+M(?:[0-9]+D)?)|(?P<dur_day>[0-9]+D))(?:T(?:(?P<dur_hour>[0-9]+H(?:[0-9]+M(?:[0-9]+S)?)?)|(?P<dur_minute>[0-9]+M(?:[0-9]+S)?)|(?P<dur_second>[0-9]+S)))?)|(?P<dur_time>T(?:(?P<dur_hour2>[0-9]+H(?:[0-9]+M(?:[0-9]+S)?)?)|(?P<dur_minute2>[0-9]+M(?:[0-9]+S)?)|(?P<dur_second2>[0-9]+S)))|(?P<dur_week>[0-9]+W))$"
    ).unwrap();

        let mut schema = test_file.schema.clone();
        if !schema["$schema"].is_string() {
            schema["$schema"] = json!("http://json-schema.org/draft-07/schema#");
        }

        stats.file_info.id = file_name.split('/').last().unwrap().to_string();

        match Validator::options()
            .with_retriever(DummyResolver {})
            .should_validate_formats(true)
            .with_format("uuid", move |value| uuid_regex.is_match(value))
            .with_format("iri", move |value| iri_regex.is_match(value))
            .with_format("duration", move |value| duration_regex.is_match(value))
            // .with_draft(jsonschema::Draft::Draft202012)
            .build(&schema)
        {
            Ok(v) => {
                for (idx, t) in test_file.tests.iter_mut().enumerate() {
                    t.rust_error = None;
                    let res = v.validate(&t.data);
                    if t.valid {
                        stats.file_info.num_valid_tests += 1;
                        stats.file_info.size_valid_tests +=
                            serde_json::to_string(&t.data).unwrap().len();
                    } else {
                        stats.file_info.num_invalid_tests += 1;
                        stats.file_info.size_invalid_tests +=
                            serde_json::to_string(&t.data).unwrap().len();
                    }
                    match res {
                        Ok(_) if t.valid => {}
                        Err(e) if !t.valid => {
                            t.rust_error = Some(format!("{e}"));
                        }
                        Ok(_) => {
                            eprintln!("{} {idx} Error: Expected invalid, got valid", file_name);
                            t.rust_error = Some("Expected invalid, got valid".to_string());
                            stats.test_invalid_error =
                                Some("Expected invalid, got valid".to_string());
                        }
                        Err(e) => {
                            eprintln!("{} {idx} Error Validating: {}", file_name, e);
                            t.broken = true;
                            stats.test_valid_error = Some(format!("{e}"));
                        }
                    }

                    limit_string(&mut t.python_error);
                    limit_string(&mut t.rust_error);
                }
            }
            Err(e) => {
                eprintln!("{} Error Creating Validator: {}", file_name, e);
                stats.schema_error = Some(format!("{e}"));
            }
        }

        if self.cli.remove_broken_tests {
            test_file.tests.retain(|t| !t.broken);
        }

        {
            let sch_stats =
                SchemaStats::for_file(file_name, &test_file.schema, self.cli.additional_features);
            let (mut raw_features, mut features): (Vec<_>, Vec<_>) = sch_stats
                .features
                .keys()
                .cloned()
                .partition(|f| is_non_semantic_feature(f));
            features.sort();
            raw_features.sort();
            let meta = JsonMetaInfo {
                full_size: sch_stats.full_size,
                stripped_size: sch_stats.stripped_size,
                features,
                raw_features,
            };
            test_file.meta = Some(meta.clone());
            stats.file_info.meta = meta;
        }

        save_json_to_file(file_name, &test_file);

        if self.cli.llg_compile {
            let mut llg = self.run_llg_compile(&test_file);
            llg.id = stats.file_info.id.clone();
            stats.llg_result = Some(llg);
        }

        stats
    }
}

fn main() {
    let mut options = CliOptions::parse();
    if options.llg_test {
        options.llg_compile = true;
    }
    let options = Arc::new(options);

    let mut files = vec![];
    for arg in &options.files {
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

    files.sort();

    if options.skip_synth {
        files.retain(|f| !f.contains("Handwritten") && !f.contains("Synthesized"));
    }

    let tok_env: TokEnv =
        toktrie_hf_tokenizers::ByteTokenizerEnv::from_name("microsoft/Phi-3.5-mini-instruct", None)
            .unwrap()
            .to_env();

    let t0 = std::time::Instant::now();
    let par = true; // options.llg_test == false;
    let do_file = |file: &String| {
        let env = TestEnv {
            tok_env: tok_env.clone(),
            file_name: file.to_string(),
            cli: options.clone(),
        };
        env.run_test()
    };
    let results = if par {
        files.par_iter().map(do_file).collect::<Vec<_>>()
    } else {
        files.iter().map(do_file).collect::<Vec<_>>()
    };

    let mut total = TotalStats::default();
    let mut all_stats = HashMap::new();
    let mut num_files_by_feature: HashMap<String, usize> = HashMap::new();
    let mut num_files_by_raw_feature: HashMap<String, usize> = HashMap::new();
    let mut all_file_info = vec![];
    let mut llg_results = vec![];

    for (file, s) in files.iter().zip(results.into_iter()) {
        all_stats.insert(file.clone(), s.clone());

        all_file_info.push(s.file_info.clone());

        for f in s.file_info.meta.raw_features {
            *num_files_by_raw_feature.entry(f).or_insert(0) += 1;
        }
        for f in s.file_info.meta.features {
            *num_files_by_feature.entry(f).or_insert(0) += 1;
        }

        total.num_valid_tests += s.file_info.num_valid_tests;
        total.num_invalid_tests += s.file_info.num_invalid_tests;

        total.num_files += 1;
        total.full_size += s.full_size;

        if let Some(llg) = s.llg_result {
            if llg.compile_error.is_some() {
                total.num_llg_compile_error += 1;
            }
            if llg.parser_error.is_some() {
                total.num_llg_parser_error += 1;
            }
            if let Some(msg) = llg.validation_error.as_ref() {
                if msg.contains("consider making your grammar left-recursive") {
                    total.num_llg_parser_limits += 1;
                } else if msg.contains("incorrect accept") {
                    total.num_llg_invalidation_error += 1;
                    eprintln!("{} Error Invalidation: {}", s.file_name, msg);
                } else {
                    total.num_llg_validation_error += 1;
                    eprintln!("{} Error Validation: {}", s.file_name, msg);
                }
            } else {
                if llg.num_valid > 0 {
                    total.num_llg_correct_schemas += 1;
                }
            }

            total.num_llg_valid_tests += llg.num_valid;
            total.num_llg_invalid_tests += llg.num_invalid;

            llg_results.push(llg);
        }

        if s.schema_error.is_some() {
            total.num_schema_error += 1;
        } else if s.test_valid_error.is_some() {
            total.num_valid_error += 1;
        } else if s.test_invalid_error.is_some() {
            total.num_invalid_error += 1;
        }
        total.ttfm_us += s.ttfm_us;
        total.max_ttfm_us = total.max_ttfm_us.max(s.ttfm_us);
    }
    println!("{}", serde_json::to_string_pretty(&total).unwrap());

    println!("Total time: {}ms", t0.elapsed().as_millis());

    save_json_to_file("tmp/test_total.json", &total);
    save_json_to_file("tmp/test_all_stats.json", &all_stats);
    save_json_to_file(
        "../../JSONSchemaBench-plain/metainfo/all_test_info.json",
        &all_file_info,
    );

    if llg_results.len() > 0 {
        save_json_to_file("tmp/llg_results.json", &llg_results);
    }

    save_sorted_json_to_file("tmp/num_files_with_feature.json", &num_files_by_feature);
    save_sorted_json_to_file(
        "tmp/num_files_with_raw_feature.json",
        &num_files_by_raw_feature,
    );
}

fn save_json_to_file<T: Serialize>(filename: &str, data: &T) {
    let mut file =
        File::create(filename).expect(format!("Unable to create file {}", filename).as_str());
    file.write_all(serde_json::to_string_pretty(data).unwrap().as_bytes())
        .expect(format!("Unable to write file {}", filename).as_str());
    // eprintln!("Saved to {}", filename);
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TotalStats {
    num_files: usize,

    num_llg_compile_error: usize,
    num_llg_parser_error: usize,
    num_llg_validation_error: usize,
    num_llg_invalidation_error: usize,
    num_llg_parser_limits: usize,
    num_llg_correct_schemas: usize,

    num_valid_tests: usize,
    num_invalid_tests: usize,

    num_llg_valid_tests: usize,
    num_llg_invalid_tests: usize,

    num_schema_error: usize,
    num_fixed_schema_error: usize,
    num_valid_error: usize,
    num_invalid_error: usize,
    ttfm_us: usize,
    max_ttfm_us: usize,
    full_size: usize,
    stripped_size: usize,
}

fn read_file_to_string(filename: &str) -> String {
    let mut file = File::open(filename).expect("Unable to open file");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Unable to read file");
    content
}

fn save_sorted_json_to_file(filename: &str, data: &HashMap<String, usize>) {
    let mut data: Vec<_> = data.iter().collect();
    data.sort_by(|a, b| b.1.cmp(&a.1));
    let data = Value::Object(
        data.iter()
            .map(|(k, v)| ((*k).clone(), Value::Number((**v as u64).into())))
            .collect::<serde_json::Map<_, _>>(),
    );
    save_json_to_file(filename, &data);
}

fn is_non_semantic_feature(feature: &str) -> bool {
    feature.starts_with("_meta:")
        || feature.starts_with("type:")
        || feature.starts_with("_nested:")
        || feature.ends_with(":trivial")
        || feature == "$id"
        || feature == "$schema"
        || feature == "definitions"
        || feature == "$defs"
        || feature == "defs"
        || feature == "id"
        // these are very widely supported and almost always used; they are not interesting
        || feature == "_boolSchema"
        || feature == "type"
        || feature == "properties"
        || feature == "required"
        || feature == "_requiredEmpty"
        // these are covered by @minmax... features
        || feature == "minimum"
        || feature == "maximum"
        || feature == "exclusiveMinimum"
        || feature == "exclusiveMaximum"
        || feature == "minLength"
        || feature == "maxLength"
        || feature == "minItems"
        || feature == "maxItems"
        || feature == "minProperties"
        || feature == "maxProperties"
}

fn limit_string(sp: &mut Option<String>) {
    if let Some(s) = sp {
        if s.len() > 1100 {
            *sp = Some(format!(
                "{}.. ({} more)",
                &String::from_utf8_lossy(&s.as_bytes()[..1024]),
                s.len() - 1024
            ));
        }
    }
}
