use anyhow::{anyhow, bail, ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

use crate::{
    api::{GrammarWithLexer, RegexSpec},
    GrammarBuilder, NodeRef,
};

#[derive(Debug, Default, Clone)]
pub struct CompileOptions {
    compact: bool,
    validate: bool,
}

impl CompileOptions {
    pub fn new(compact: bool, validate: bool) -> Self {
        Self { compact, validate }
    }
}

fn to_compact_json(target: &serde_json::Value) -> String {
    serde_json::to_string(target).unwrap()
}

const KEYWORDS: [&str; 10] = [
    "anyOf",
    "oneOf",
    "allOf",
    "$ref",
    "const",
    "enum",
    "type",
    "pattern",
    "minLength",
    "maxLength",
];
const DEFS_KEYS: [&str; 2] = ["$defs", "definitions"];
const IGNORED_KEYS: [&str; 8] = [
    "$schema",
    "$id",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "discriminator",
];

fn looks_like_schema(map: &HashMap<String, serde_json::Value>) -> bool {
    map.contains_key("type")
        || map.contains_key("anyOf")
        || map.contains_key("allOf")
        || map.contains_key("oneOf")
        || map.contains_key("enum")
        || map.contains_key("$ref")
}

fn limited_str(node: &Value) -> String {
    let s = node.to_string();
    if s.len() > 100 {
        format!("{}...", &s[..100])
    } else {
        s
    }
}

fn validate_json_node_keys(node: &Value) -> Result<()> {
    let node = node
        .as_object()
        .ok_or_else(|| anyhow!("Expected object as json schema, got: {}", limited_str(node)))?;

    for key in node.keys() {
        let key = &key.as_str();
        if KEYWORDS.contains(key) || IGNORED_KEYS.contains(key) || DEFS_KEYS.contains(key) {
            continue;
        }
        bail!("Unknown key in JSON schema: {:?}", key);
    }

    Ok(())
}

pub struct Compiler {
    builder: GrammarBuilder,
    options: CompileOptions,
    definitions: HashMap<String, NodeRef>,
}

fn regex(rx: &str) -> RegexSpec {
    RegexSpec::Regex(rx.to_string())
}

impl Compiler {
    pub fn new(options: CompileOptions) -> Self {
        Self {
            builder: GrammarBuilder::new(),
            options,
            definitions: HashMap::new(),
        }
    }

    pub fn run(&mut self, schema: &Value) -> Result<()> {
        if self.options.validate {
            // validate schema using some JSON schema validation library
        }

        self.builder.add_grammar(GrammarWithLexer {
            greedy_skip_rx: if self.options.compact {
                Some(regex(r"[\x20\x0A\x0D\x09]+"))
            } else {
                None
            },
            ..GrammarWithLexer::default()
        });

        let mut defs_key = None;

        for dk in DEFS_KEYS {
            if schema[dk].is_object() {
                ensure!(
                    defs_key.is_none(),
                    "Multiple definitions sections found in schema"
                );
                defs_key = Some(dk.to_string());
            }
        }

        if let Some(defs_key) = &defs_key {
            for (ref_, _) in schema[defs_key].as_object().unwrap() {
                let placeholder = self.builder.placeholder();
                self.definitions.insert(ref_.clone(), placeholder);
            }
        }

        let root = self.gen_json(schema)?;
        self.builder.set_start_node(root);

        if let Some(defs_key) = &defs_key {
            for (ref_, ref_schema) in schema[defs_key].as_object().unwrap() {
                let pl = self.definitions[ref_];
                let compiled = self.gen_json(ref_schema)?;
                self.builder.set_placeholder(pl, compiled);
            }
        }

        Ok(())
    }

    fn process_any_of(&mut self, obj: &Value) -> Result<NodeRef> {
        let arr = obj
            .as_array()
            .ok_or_else(|| anyhow!("Expected array in anyOf, got: {}", limited_str(obj)))?
            .iter()
            .map(|json_schema| self.gen_json(json_schema))
            .collect::<Result<Vec<_>>>()?;
        Ok(self.builder.select(&arr))
    }

    fn gen_json(&mut self, json_schema: &Value) -> Result<NodeRef> {
        validate_json_node_keys(json_schema)?;

        if let Some(any_of) = json_schema.get("anyOf") {
            return self.process_any_of(any_of);
        }
        if let Some(any_of) = json_schema.get("oneOf") {
            return self.process_any_of(any_of);
        }

        if let Some(all_of) = json_schema.get("allOf") {
            let all_of_list = all_of
                .as_array()
                .ok_or_else(|| anyhow!("Expected array in allOf, got: {}", limited_str(all_of)))?;
            if all_of_list.len() != 1 {
                bail!("Only support allOf with exactly one item");
            }
            return self.gen_json(&all_of_list[0]);
        }

        todo!()
    }

}
