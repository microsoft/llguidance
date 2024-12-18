use anyhow::{anyhow, bail, Result};
use clap::Parser;
use json_stats::SchemaStats;
use jsonschema::Validator;
use llguidance::{
    earley::regexvec::LexerStats,
    toktrie::{InferenceCapabilities, TokEnv},
    Constraint, JsonCompileOptions, ParserFactory, TokenParser,
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

    #[arg(long, short = 'm')]
    llg_masks: bool,

    #[arg(long, short = 's')]
    llg_slicer: bool,

    #[arg(long)]
    llg_no_forcing: bool,

    #[arg(long)]
    num_threads: Option<usize>,

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

const MASK_STEPS: usize = 16;

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct LlgResult {
    id: String,

    #[serde(skip_serializing_if = "is_zero")]
    ttfm_us: usize,
    #[serde(skip_serializing_if = "is_zero")]
    masks_us: usize,
    #[serde(skip_serializing_if = "is_zero")]
    max_mask_us: usize,
    #[serde(skip_serializing_if = "is_zero")]
    slicer_leftover_us: usize,

    one: usize,

    num_tokens: usize,
    num_tests: usize,
    num_valid_tests: usize,
    num_invalid_tests: usize,

    avg_parser_items: usize,
    max_avg_parser_items: usize,
    sum_parser_items: usize,
    max_sum_parser_items: usize,
    max_parser_items: usize,
    max_lexer_cost: u64,
    max_lexer_states: usize,
    lexer_cost: u64,

    lexer_stats: LexerStats,

    #[serde(skip)]
    slow_mask_count: [usize; MASK_STEPS],
    #[serde(skip)]
    slow_mask_us: [usize; MASK_STEPS],

    #[serde(skip)]
    slow_mask_count_a: [usize; MASK_STEPS],
    #[serde(skip)]
    slow_mask_us_a: [usize; MASK_STEPS],

    #[serde(skip_serializing_if = "Option::is_none")]
    compile_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parser_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_error: Option<String>,
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}

impl LlgResult {
    pub fn clear_timings(&mut self) {
        self.ttfm_us = 0;
        self.masks_us = 0;
        self.max_mask_us = 0;
        self.slicer_leftover_us = 0;
    }
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
    factory: Arc<ParserFactory>,
    file_name: String,
}

fn json_sum(curr: &mut Value, v: &Value) {
    assert!(curr.is_object());
    assert!(v.is_object());
    let v = v.as_object().unwrap();
    for (k, v) in v.iter() {
        if let Some(v) = v.as_i64() {
            let c = &curr[k];
            let v2 = if c.is_null() {
                v
            } else {
                if k.starts_with("max_") {
                    std::cmp::max(c.as_i64().unwrap(), v)
                } else {
                    c.as_i64().unwrap() + v
                }
            };
            curr[k] = json!(v2);
        }
    }
}

impl TestEnv {
    fn run_llg_test_inner(
        &self,
        stats: &mut LlgResult,
        parser: &mut TokenParser,
        t: &JsonTestSequence,
    ) -> Result<()> {
        let dstr = serde_json::to_string(&t.data).unwrap();
        let tokens = self.tok_env.tokenize(&dstr);
        let trie = self.tok_env.tok_trie();
        let masks = self.cli.llg_masks;

        stats.num_tests += 1;

        // println!("tokenized: {}", trie.tokens_dbg(&tokens));

        for (tidx, &token) in tokens.iter().enumerate() {
            // eprintln!("WILL TEST {}: {}", tidx, trie.token_dbg(token));

            stats.num_tokens += 1;

            let ok = if masks {
                let t0 = std::time::Instant::now();
                let m = parser.compute_mask()?; // .unwrap_or_else(|_| trie.alloc_token_set());
                let us = t0.elapsed().as_micros() as usize;
                let pstats = parser.last_step_stats();

                // && pstats.lexer_cost < 7 * us as u64
                if false && us > 100 {
                    // MASK,us,lexer_cost,slices,items,rows,cached_rows
                    eprintln!(
                        "{},{},{},{},{},{},{}",
                        if us > 1000 { "SLOW" } else { "OK" },
                        us,
                        pstats.lexer_cost,
                        pstats.slices_applied,
                        pstats.all_items,
                        pstats.rows,
                        pstats.cached_rows,
                    );
                    eprintln!("{}", parser.parser.lexer_stats());

                    // eprintln!("{:?}", pstats);
                }

                stats.sum_parser_items += pstats.all_items;
                stats.max_parser_items = std::cmp::max(stats.max_parser_items, pstats.all_items);
                stats.max_lexer_cost = std::cmp::max(stats.max_lexer_cost, pstats.lexer_cost);
                stats.lexer_cost += pstats.lexer_cost;

                let step = us.next_power_of_two().trailing_zeros() as usize;
                let step = std::cmp::min(step, MASK_STEPS - 1);

                stats.slow_mask_count[step] += 1;
                stats.slow_mask_us[step] += us;

                // assert!(pstats.slices_applied <= 1);

                let is_big = m.num_set() >= 120_000;
                let sliced = pstats.slices_applied > 0;
                let cond_a = is_big && sliced;
                if cond_a {
                    stats.slow_mask_count_a[step] += 1;
                    stats.slow_mask_us_a[step] += us;
                }

                stats.max_mask_us = std::cmp::max(stats.max_mask_us, us);
                m.is_allowed(token)
            } else {
                parser.validate_token(token)?
            };

            if !ok {
                if t.valid {
                    bail!(
                        "token not accepted at {}",
                        trie.tokens_dbg(&tokens[0..tidx + 1])
                            .replace("\\\"", "“")
                            .replace("\"", "")
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
                bail!(
                    "incorrect accept; expected {}",
                    t.rust_error
                        .clone()
                        .unwrap_or_else(|| t.python_error.clone().unwrap_or("???".to_string()))
                );
            }
        } else {
            if t.valid {
                bail!("parser not accepting at the end");
            }
        }

        Ok(())
    }

    fn run_llg_test(
        &self,
        stats: &mut LlgResult,
        parser: &TokenParser,
        t: &JsonTestSequence,
    ) -> Result<()> {
        // if self.cli.llg_masks && !t.valid {
        //     return Ok(());
        // }

        let mut parser = parser.deep_clone();
        parser.start_without_prompt();

        let r = self.run_llg_test_inner(stats, &mut parser, t);

        let m = parser.parser.metrics_mut();
        stats.slicer_leftover_us += m.slicer_leftover_us;

        let lx = parser.parser.lexer_stats();
        stats.max_lexer_states = std::cmp::max(stats.max_lexer_states, lx.num_states);

        r
    }

    fn run_llg_compile(&self, test_file: &JsonTest) -> LlgResult {
        let opts = JsonCompileOptions::default();
        let mut res = LlgResult::default();

        let t0 = std::time::Instant::now();
        let schema = opts.json_to_llg(test_file.schema.clone());

        let mut schema = match schema {
            Ok(schema) => schema,
            Err(e) => {
                res.compile_error = Some(format!("{e}"));
                limit_string(&mut res.compile_error);
                // eprintln!("{} Error Compile: {}", file, e);
                return res;
            }
        };

        if self.cli.llg_no_forcing {
            schema.grammars[0].no_forcing = true;
        }

        let parser = self.factory.create_parser(schema);

        let parser = match parser {
            Ok(parser) => {
                let mut constraint = Constraint::new(parser.clone());
                constraint.compute_mask().unwrap();
                res.ttfm_us = t0.elapsed().as_micros() as usize;
                res.one = 1;
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

        res.lexer_stats = parser.parser.lexer_stats();

        if self.cli.llg_test {
            for (idx, t) in test_file.tests.iter().enumerate() {
                let t0 = std::time::Instant::now();
                if let Err(e) = self.run_llg_test(&mut res, &parser, t) {
                    res.validation_error = Some(format!("test #{idx}: {e}"));
                    limit_string(&mut res.validation_error);
                } else {
                    if t.valid {
                        res.num_valid_tests += 1;
                    } else {
                        res.num_invalid_tests += 1;
                    }
                }
                res.masks_us += t0.elapsed().as_micros() as usize;
            }

            if res.num_tokens > 0 {
                res.avg_parser_items = res.sum_parser_items / res.num_tokens;
                res.max_avg_parser_items = res.sum_parser_items / res.num_tokens;
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
    let jsb_data = std::env::var("JSB_DATA").expect("JSB_DATA environment variable not set");

    let mut options = CliOptions::parse();
    if options.llg_masks {
        options.llg_test = true;
    }
    if options.llg_test {
        options.llg_compile = true;
    }

    // set max thread numbers
    let num_cores = std::thread::available_parallelism().unwrap().get();
    let num_threads = options
        .num_threads
        .unwrap_or_else(|| std::cmp::min(num_cores, 40));
    if num_threads > 1 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap();
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

    // "microsoft/Phi-3.5-mini-instruct"
    let tok_env: TokEnv = toktrie_hf_tokenizers::ByteTokenizerEnv::from_name(
        "meta-llama/Llama-3.1-8B-Instruct",
        None,
    )
    .unwrap()
    .to_env();

    let mut slices = vec![
        r#"[^"\\\x00-\x1F\x7F]{1,30}"#.to_string(),
        r#"[^"\\\x00-\x1F\x7F]+"#.to_string(),

        // stats counting
        // r#"[\x00-\x1F\x7F](.|\n)*"#.to_string(), // easy reject
        // r#"[^"]*(\t|\n)"#.to_string(),
        // r#"(.|\n)*[\\"](.|\n)*"#.to_string(),
    ];
    if !options.llg_slicer {
        slices.clear();
    }

    let mut factory = ParserFactory::new(
        &tok_env,
        InferenceCapabilities {
            ff_tokens: false,
            backtrack: false,
            conditional_ff_tokens: false,
            fork: false,
        },
        &slices,
    );
    factory.quiet();
    let factory = Arc::new(factory);

    save_text_to_file("tmp/slices.txt", &factory.slicer().stats(false));
    save_text_to_file("tmp/slices_tokens.txt", &factory.slicer().stats(true));

    let t0 = std::time::Instant::now();
    let par = num_threads > 1;
    let do_file = |file: &String| {
        let env = TestEnv {
            tok_env: tok_env.clone(),
            factory: factory.clone(),
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
    let mut histogram = MaskHistogram::default();
    let mut histogram_a = MaskHistogram::default();
    let mut llg_totals = json!({});

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
            let log_err = !options.llg_masks;

            if llg.compile_error.is_some() {
                total.llg.num_compile_error += 1;
            }
            if llg.parser_error.is_some() {
                total.llg.num_parser_error += 1;
            }
            if let Some(msg) = llg.validation_error.as_ref() {
                if msg.contains("consider making your grammar left-recursive")
                    || msg.contains("try avoiding single-byte/short lexemes")
                {
                    total.llg.num_parser_limits += 1;
                } else if msg.contains("incorrect accept") {
                    total.llg.num_invalidation_error += 1;
                    if log_err {
                        eprintln!("{} Error Invalidation: {}", s.file_name, msg);
                    }
                } else {
                    total.llg.num_validation_error += 1;
                    if log_err {
                        eprintln!("{} Error Validation: {}", s.file_name, msg);
                    }
                }
            } else {
                if llg.num_valid_tests > 0 {
                    total.llg.num_correct_schemas += 1;
                }
            }

            total.llg.num_tokens += llg.num_tokens;
            json_sum(&mut llg_totals, &serde_json::to_value(&llg).unwrap());

            if llg.ttfm_us > 0 {
                total.llg.num_parsers += 1;
            }

            total.llg.ttfm_us += llg.ttfm_us;
            total.llg.mask_us_total += llg.masks_us;
            total.llg.max_mask_us = std::cmp::max(total.llg.max_mask_us, llg.max_mask_us);

            for i in 0..MASK_STEPS {
                histogram.count[i] += llg.slow_mask_count[i];
                histogram.us[i] += llg.slow_mask_us[i];
                histogram_a.count[i] += llg.slow_mask_count_a[i];
                histogram_a.us[i] += llg.slow_mask_us_a[i];
                total.llg.mask_us_total_a += llg.slow_mask_us_a[i];
                total.llg.num_masks_a += llg.slow_mask_count_a[i];
            }

            llg_results.push(llg);
        }

        if s.schema_error.is_some() {
            total.num_schema_error += 1;
        } else if s.test_valid_error.is_some() {
            total.num_valid_error += 1;
        } else if s.test_invalid_error.is_some() {
            total.num_invalid_error += 1;
        }
    }

    if total.llg.num_parsers > 0 {
        total.llg.ttfm_us /= total.llg.num_parsers;
        total.llg.mask_us = total.llg.mask_us_total / total.llg.num_tokens;
        total.llg.num_threads = num_threads;
        total.llg.mask_us_total_a_frac = total.llg.mask_us_total_a * 1000 / total.llg.mask_us_total;
        total.llg.num_masks_a_frac = total.llg.num_masks_a * 1000 / total.llg.num_tokens;
    }

    let mut histogram_csv = format!(
        "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
        "up_to", "sum", "sum_a", "s", "s_a", "count", "count_a"
    );
    let mut hist_sum = 0;
    let mut hist_sum_a = 0;
    if total.llg.mask_us_total > 0 {
        for i in 0..MASK_STEPS {
            histogram.perc[i] = histogram.us[i] * 1000 / total.llg.mask_us_total;
            hist_sum += histogram.us[i];
            hist_sum_a += histogram_a.us[i];
            histogram_csv.push_str(&format!(
                "{:10} {:10.3} {:10.3} {:10.3} {:10.3} {:10} {:10}\n",
                1 << i,
                (hist_sum as f64 / 1000_000.0),
                (hist_sum_a as f64 / 1000_000.0),
                (histogram.us[i] as f64 / 1000_000.0),
                (histogram_a.us[i] as f64 / 1000_000.0),
                histogram.count[i],
                histogram_a.count[i],
            ));
        }
    }

    println!("{}", serde_json::to_string_pretty(&total).unwrap());
    println!(
        "LLG: {}",
        serde_json::to_string_pretty(&llg_totals).unwrap()
    );

    println!("Total time: {}ms", t0.elapsed().as_millis());

    save_text_to_file("tmp/mask_histogram.csv", &histogram_csv);
    save_json_to_file("tmp/test_total.json", &total);
    save_json_to_file("tmp/test_all_stats.json", &all_stats);
    save_json_to_file(
        format!("{}/metainfo/all_stats.json", jsb_data).as_str(),
        &all_file_info,
    );

    if llg_results.len() > 0 {
        save_json_to_file("tmp/llg_results.json", &llg_results);
        for r in llg_results.iter_mut() {
            r.clear_timings();
        }
        save_json_to_file("tmp/llg_results_timeless.json", &llg_results);
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

fn save_text_to_file(filename: &str, data: &str) {
    let mut file =
        File::create(filename).expect(format!("Unable to create file {}", filename).as_str());
    file.write_all(data.as_bytes())
        .expect(format!("Unable to write file {}", filename).as_str());
    // eprintln!("Saved to {}", filename);
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct MaskHistogram {
    count: [usize; MASK_STEPS],
    us: [usize; MASK_STEPS],
    perc: [usize; MASK_STEPS],
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct LlgTotalStats {
    num_compile_error: usize,
    num_parser_error: usize,
    num_validation_error: usize,
    num_invalidation_error: usize,
    num_parser_limits: usize,
    num_correct_schemas: usize,
    num_tokens: usize,
    num_parsers: usize,
    num_threads: usize,
    ttfm_us: usize,
    max_mask_us: usize,
    mask_us: usize,
    mask_us_total: usize,
    mask_us_total_a: usize,
    num_masks_a: usize,
    mask_us_total_a_frac: usize,
    num_masks_a_frac: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TotalStats {
    num_files: usize,
    num_valid_tests: usize,
    num_invalid_tests: usize,
    num_schema_error: usize,
    num_fixed_schema_error: usize,
    num_valid_error: usize,
    num_invalid_error: usize,
    full_size: usize,
    stripped_size: usize,
    llg: LlgTotalStats,
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
