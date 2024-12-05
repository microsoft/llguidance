use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{Read, Write},
};

use anyhow::{bail, Result};
use llguidance::{
    api::ParserLimits,
    toktrie::{InferenceCapabilities, TokEnv},
    Constraint, JsonCompileOptions, TokenParser,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SchemaStats {
    features: HashMap<String, usize>,
    skipped_kw: HashMap<String, usize>,
    skipped_kw_filtered: HashMap<String, usize>,
    skipped_data: Vec<(String, Value)>,
    full_size: usize,
    stripped_size: usize,
    ttfm_us: usize,

    strip_error: Option<String>,
    compile_error: Option<String>,
    parser_error: Option<String>,
}

fn is_directly_nested_schema(kw: &str) -> bool {
    match kw {
        "allOf"
        | "anyOf"
        | "oneOf"
        | "not"
        | "items"
        | "additionalProperties"
        | "additionalItems"
        | "contains"
        | "propertyNames"
        | "dependentSchemas"
        | "prefixItems"
        | "unevaluatedItems"
        | "unevaluatedProperties"
        | "if"
        | "then"
        | "else" => true,
        _ => false,
    }
}

fn is_map_to_schema(kw: &str) -> bool {
    match kw {
        "properties" | "patternProperties" | "definitions" | "defs" | "$defs" => true,
        _ => false,
    }
}

fn is_schema_kw(kw: &str) -> bool {
    match kw {
        "$ref"
        | "$schema"
        | "additionalItems"
        | "additionalProperties"
        | "allOf"
        | "anyOf"
        | "dependencies"
        | "enum"
        | "exclusiveMaximum"
        | "exclusiveMinimum"
        | "format"
        | "items"
        | "maxItems"
        | "maxLength"
        | "maxProperties"
        | "maximum"
        | "minItems"
        | "minLength"
        | "minProperties"
        | "minimum"
        | "multipleOf"
        | "not"
        | "oneOf"
        | "pattern"
        | "patternProperties"
        | "properties"
        | "required"
        | "type"
        | "uniqueItems"
        | "id"
        | "$id"
        | "const"
        | "contains"
        | "propertyNames"
        | "contentEncoding"
        | "contentMediaType"
        | "else"
        | "if"
        | "then"
        | "$anchor"
        | "$defs"
        | "$recursiveAnchor"
        | "$recursiveRef"
        | "dependentRequired"
        | "dependentSchemas"
        | "maxContains"
        | "minContains"
        | "prefixItems"
        | "unevaluatedItems"
        | "unevaluatedProperties"
        | "$dynamicAnchor"
        | "$dynamicRef" => true,
        // added
        _ => is_directly_nested_schema(kw) || is_map_to_schema(kw),
    }
}

impl SchemaStats {
    fn incr(&mut self, feature: &str) {
        self.features
            .entry(feature.to_string())
            .and_modify(|e| *e += 1)
            .or_insert(1);
    }

    fn incr_feature(&mut self, kw: &str, val: &Value) {
        self.incr(kw);
        match kw {
            "additionalItems" | "additionalProperties" => {
                if val.is_object() {
                    self.incr(&format!("{}:object", kw));
                }
            }
            "type" => {
                if let Some(types) = val.as_array() {
                    self.incr("type_array");
                    for t in types {
                        if let Some(t) = t.as_str() {
                            self.incr(&format!("type:{}", t));
                        }
                    }
                } else if let Some(type_id) = val.as_str() {
                    self.incr(&format!("type:{}", type_id));
                }
            }
            "format" => {
                if let Some(format) = val.as_str() {
                    self.incr(&format!("format:{}", format));
                }
            }
            _ => {}
        }
    }

    fn map_schema(&mut self, schema: &Value) -> Result<Value> {
        match schema {
            Value::Bool(_) => {
                self.incr("bool_schema");
                Ok(schema.clone())
            }
            Value::Object(obj) => {
                let mut new_obj = serde_json::Map::new();
                for (k, v) in obj.iter() {
                    if !is_schema_kw(k) {
                        if v["$schema"].is_string() {
                            new_obj.insert(k.clone(), v.clone());
                            continue;
                        }
                        let size = serde_json::to_string(v).unwrap().len() + k.len() + 5;
                        self.skipped_kw
                            .entry(k.clone())
                            .and_modify(|e| *e += size)
                            .or_insert(size);
                        match k.as_str() {
                            "title" | "description" | "default" | "examples" | "example"
                            | "translation" | "readonly" | "defaultValue" | "_format" | "_id" => {
                                self.incr(&format!("meta:{}", k));
                            }
                            _ => {
                                self.skipped_kw_filtered
                                    .entry(k.clone())
                                    .and_modify(|e| *e += size)
                                    .or_insert(size);
                                self.skipped_data.push((k.clone(), v.clone()));
                            }
                        }
                        continue;
                    }
                    self.incr_feature(k, v);
                    let new_v = if is_directly_nested_schema(k) {
                        if let Some(seq) = v.as_array() {
                            let seq = seq
                                .iter()
                                .map(|v| self.map_schema(v))
                                .collect::<Result<_>>()?;
                            Value::Array(seq)
                        } else {
                            self.map_schema(v)?
                        }
                    } else if is_map_to_schema(k) {
                        if let Some(obj) = v.as_object() {
                            let obj = obj
                                .iter()
                                .map(|(k, v)| {
                                    let new_v = self.map_schema(v)?;
                                    Ok((k.clone(), new_v))
                                })
                                .collect::<Result<_>>()?;
                            Value::Object(obj)
                        } else {
                            bail!("Expected object for {}", k);
                        }
                    } else if k == "dependencies" {
                        if let Some(obj) = v.as_object() {
                            let obj = obj
                                .iter()
                                .map(|(k, v)| {
                                    let new_v = if v.is_array() {
                                        v.clone()
                                    } else {
                                        self.map_schema(v)?
                                    };
                                    Ok((k.clone(), new_v))
                                })
                                .collect::<Result<_>>()?;
                            Value::Object(obj)
                        } else {
                            bail!("Expected object for {}", k);
                        }
                    } else {
                        v.clone()
                    };
                    new_obj.insert(k.clone(), new_v);
                }
                Ok(Value::Object(new_obj))
            }
            _ => bail!(
                "Expected object or bool schema; got {}",
                serde_json::to_string(schema).unwrap()
            ),
        }
    }
}

fn test_file(tok_env: TokEnv, file: &str) -> SchemaStats {
    let schema_file = read_file_to_string(file);
    let opts = JsonCompileOptions::default();
    let val: Value = serde_json::from_str(&schema_file).expect("Invalid JSON in schema");

    let mut stats = SchemaStats::default();
    stats.full_size = serde_json::to_string(&val).unwrap().len();
    match stats.map_schema(&val) {
        Ok(val) => {
            stats.stripped_size = serde_json::to_string(&val).unwrap().len();
        }
        Err(e) => {
            eprintln!("{} Error Stats: {}", file, e);
            stats.strip_error = Some(format!("{e}"));
        }
    }

    let t0 = std::time::Instant::now();
    let schema = opts.json_to_llg(val);

    let schema = match schema {
        Ok(schema) => schema,
        Err(e) => {
            // eprintln!("{} Error Compile: {}", file, e);
            stats.compile_error = Some(format!("{e}"));
            return stats;
        }
    };

    let parser = TokenParser::from_llguidance_json(
        tok_env,
        schema,
        llguidance::Logger::new(0, 1),
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
            // eprintln!("{} OK", file);
        }
        Err(e) => {
            eprintln!("{} Error Parser: {}", file, e);
            stats.parser_error = Some(format!("{e}"));
        }
    }

    stats.ttfm_us = t0.elapsed().as_micros() as usize;

    stats
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

    let mut total = TotalStats::default();
    let mut skipped_data = vec![];
    let mut skipped_kw_size: HashMap<String, usize> = HashMap::new();
    let mut files_with_feature = HashMap::new();
    for (idx, file) in files.iter().enumerate() {
        if idx % 1000 == 0 {
            eprintln!("{} / {}", idx, files.len());
        }
        let s = test_file(tok_env.clone(), &file);
        total.num_files += 1;
        total.full_size += s.full_size;
        total.stripped_size += s.stripped_size;
        for (k, v) in s.skipped_data {
            skipped_data.push(json!({
                "file": file,
                "key": k,
                "value": v,
            }));
        }
        for (k, v) in s.skipped_kw_filtered {
            skipped_kw_size
                .entry(k)
                .and_modify(|e| *e += v)
                .or_insert(v);
        }
        for (k, _) in s.features {
            files_with_feature
                .entry(k)
                .and_modify(|e| *e += 1)
                .or_insert(1);
        }
        if s.compile_error.is_some() {
            total.num_compile_error += 1;
        }
        total.ttfm_us += s.ttfm_us;
        total.max_ttfm_us = total.max_ttfm_us.max(s.ttfm_us);
    }
    println!("{}", serde_json::to_string_pretty(&total).unwrap());

    save_sorted_json_to_file("tmp/num_files_with_feature.json", &files_with_feature);
    save_sorted_json_to_file("tmp/skipped_size.json", &skipped_kw_size);
    save_json_to_file("tmp/total.json", &total);
    save_json_to_file("tmp/skipped_data.json", &skipped_data);
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

fn save_json_to_file<T: Serialize>(filename: &str, data: &T) {
    let mut file =
        File::create(filename).expect(format!("Unable to create file {}", filename).as_str());
    file.write_all(serde_json::to_string_pretty(data).unwrap().as_bytes())
        .expect(format!("Unable to write file {}", filename).as_str());
    eprintln!("Saved to {}", filename);
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TotalStats {
    num_files: usize,
    num_compile_error: usize,
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
