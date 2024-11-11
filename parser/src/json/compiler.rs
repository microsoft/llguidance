use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use std::{collections::HashMap, vec};

use super::formats::lookup_format;
use super::numeric::{rx_float_range, rx_int_range};
use crate::{
    api::{GrammarWithLexer, RegexSpec, TopLevelGrammar},
    GrammarBuilder, NodeRef,
};

// TODO: grammar size limit
// TODO: array maxItems etc limits
// TODO: schemastore/src/schemas/json/BizTalkServerApplicationSchema.json - this breaks 1M fuel on lexer, why?!

#[derive(Debug, Clone)]
pub struct JsonCompileOptions {
    pub item_separator: String,
    pub key_separator: String,
    pub whitespace_flexible: bool,
}

fn to_compact_json(target: &serde_json::Value) -> String {
    serde_json::to_string(target).unwrap()
}

const TYPES: [&str; 7] = ["null", "boolean", "integer", "number", "string", "array", "object"];
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
const IGNORED_KEYS: [&str; 19] = [
    "$schema",
    "$id",
    "id",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "authors",
    "deprecationMessage",
    "enumDescriptions",
    "example",
    "postActions",
    "readOnly",
    "markdownDescription",
    "deprecated",
    "dependencies",
    "discriminator", // we hope it's part of the grammar anyways
    "required",      // TODO: implement and remove from ignored list
];
// these are also just ignored
const DEFS_KEYS: [&str; 4] = ["$defs", "definitions", "defs", "refs"];

const ARRAY_KEYS: [&str; 4] = ["items", "prefixItems", "minItems", "maxItems"];
const OBJECT_KEYS: [&str; 3] = ["properties", "additionalProperties", "required"];
const STRING_KEYS: [&str; 4] = ["minLength", "maxLength", "pattern", "format"];
const NUMERIC_KEYS: [&str; 4] = ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"];

const CHAR_REGEX: &str = r#"(\\([\"\\\/bfnrt]|u[a-fA-F0-9]{4})|[^\"\\\x00-\x1F\x7F])"#;

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
        .ok_or_else(|| anyhow!("Expected object as json schema, got: {}", limited_str(node)))
        .unwrap();

    for key in node.keys() {
        let key = &key.as_str();
        if KEYWORDS.contains(key) || IGNORED_KEYS.contains(key) || DEFS_KEYS.contains(key) || ARRAY_KEYS.contains(key) || OBJECT_KEYS.contains(key) || STRING_KEYS.contains(key) || NUMERIC_KEYS.contains(key) {
            continue;
        }
        if key.starts_with("x-") || key.starts_with("$xsd-") {
            continue;
        }
        bail!("Unknown key in JSON schema: {:?}", key);
    }

    Ok(())
}

struct Compiler {
    builder: GrammarBuilder,
    options: JsonCompileOptions,
    definitions: HashMap<String, NodeRef>,
    pending_definitions: Vec<(String, NodeRef)>,

    any_cache: Option<NodeRef>,
    lexeme_cache: HashMap<String, NodeRef>,
}

macro_rules! cache {
    ($field:expr, $gen:expr) => {
        if $field.is_none() {
            $field = Some($gen);
        }
        return $field.unwrap();
    };
}

impl Default for JsonCompileOptions {
    fn default() -> Self {
        Self {
            item_separator: ",".to_string(),
            key_separator: ":".to_string(),
            whitespace_flexible: true,
        }
    }
}

impl JsonCompileOptions {
    pub fn json_to_llg(&self, schema: &Value) -> Result<TopLevelGrammar> {
        let mut compiler = Compiler::new(self.clone());
        #[cfg(feature = "jsonschema_validation")]
        {
            use crate::json_validation::validate_schema;
            validate_schema(schema)?;
        }

        compiler.execute(schema)?;
        compiler.builder.finalize()
    }

    pub fn json_to_llg_no_validate(&self, schema: &Value) -> Result<TopLevelGrammar> {
        let mut compiler = Compiler::new(self.clone());
        compiler.execute(schema)?;
        compiler.builder.finalize()
    }
}

fn mk_regex(rx: &str) -> RegexSpec {
    RegexSpec::Regex(rx.to_string())
}

trait OptionalField {
    fn opt_u64(&self, key: &str) -> Result<Option<u64>>;
    fn opt_f64(&self, key: &str) -> Result<Option<f64>>;
    fn opt_str(&self, key: &str) -> Result<Option<&str>>;
    fn opt_array(&self, key: &str) -> Result<Option<&Vec<Value>>>;
    #[allow(dead_code)]
    fn opt_bool(&self, key: &str) -> Result<Option<bool>>;
    fn opt_object(&self, key: &str) -> Result<Option<&serde_json::Map<String, Value>>>;
}

fn expected_err(key: &str, val: &Value, expected: &str) -> anyhow::Error {
    anyhow!(
        "Expected {} for field {:?}, got: {}",
        expected,
        key,
        limited_str(val)
    )
}

impl OptionalField for Value {
    fn opt_u64(&self, key: &str) -> Result<Option<u64>> {
        if let Some(val) = self.get(key) {
            val.as_u64()
                .ok_or_else(|| expected_err(key, val, "unsigned integer"))
                .map(Some)
        } else {
            Ok(None)
        }
    }

    fn opt_f64(&self, key: &str) -> Result<Option<f64>> {
        if let Some(val) = self.get(key) {
            val.as_f64()
                .ok_or_else(|| expected_err(key, val, "float"))
                .map(Some)
        } else {
            Ok(None)
        }
    }

    fn opt_str(&self, key: &str) -> Result<Option<&str>> {
        if let Some(val) = self.get(key) {
            val.as_str()
                .ok_or_else(|| expected_err(key, val, "string"))
                .map(Some)
        } else {
            Ok(None)
        }
    }

    fn opt_array(&self, key: &str) -> Result<Option<&Vec<Value>>> {
        if let Some(val) = self.get(key) {
            val.as_array()
                .ok_or_else(|| expected_err(key, val, "array"))
                .map(Some)
        } else {
            Ok(None)
        }
    }

    fn opt_bool(&self, key: &str) -> Result<Option<bool>> {
        if let Some(val) = self.get(key) {
            val.as_bool()
                .ok_or_else(|| expected_err(key, val, "boolean"))
                .map(Some)
        } else {
            Ok(None)
        }
    }

    fn opt_object(&self, key: &str) -> Result<Option<&serde_json::Map<String, Value>>> {
        if let Some(val) = self.get(key) {
            val.as_object()
                .ok_or_else(|| expected_err(key, val, "object"))
                .map(Some)
        } else {
            Ok(None)
        }
    }
}

impl Compiler {
    pub fn new(options: JsonCompileOptions) -> Self {
        Self {
            builder: GrammarBuilder::new(),
            options,
            definitions: HashMap::new(),
            pending_definitions: vec![],
            lexeme_cache: HashMap::new(),
            any_cache: None,
        }
    }

    pub fn execute(&mut self, schema: &Value) -> Result<()> {
        self.builder.add_grammar(GrammarWithLexer {
            greedy_skip_rx: if self.options.whitespace_flexible {
                Some(mk_regex(r"[\x20\x0A\x0D\x09]+"))
            } else {
                None
            },
            ..GrammarWithLexer::default()
        });

        let root = self.gen_json(schema)?;
        self.builder.set_start_node(root);

        while let Some((path0, pl)) = self.pending_definitions.pop() {
            // path is #/foo/bar/baz, first split into elements
            let path = path0.trim_start_matches("#/");
            let path = path.split('/').collect::<Vec<_>>();
            let mut node = schema;
            for elem in path {
                node = &node[elem];
            }
            if node.is_null() {
                bail!("Definition not found: {}", path0);
            }

            let compiled = self.gen_json(node)?;
            self.builder.set_placeholder(pl, compiled);
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
        if json_schema.as_bool() == Some(true) {
            return Ok(self.gen_json_any());
        }

        if json_schema.as_bool() == Some(false) {
            bail!("'false' not supported as schema here");
        }

        if let Some(json_schema) = json_schema.as_object() {
            // TODO: should be sufficient to have only ignored keys here
            if json_schema.is_empty() {
                return Ok(self.gen_json_any());
            }
        }

        // eprintln!("gen_json: {}", limited_str(json_schema));
        validate_json_node_keys(json_schema)?;

        // Process anyOf
        if let Some(any_of) = json_schema.get("anyOf") {
            return self.process_any_of(any_of);
        }

        // Process oneOf (same handling as anyOf for now)
        if let Some(one_of) = json_schema.get("oneOf") {
            return self.process_any_of(one_of);
        }

        // Process allOf
        if let Some(all_of_list) = json_schema.opt_array("allOf")? {
            if all_of_list.len() != 1 {
                bail!("Only support allOf with exactly one item");
            }
            return self.gen_json(&all_of_list[0]);
        }

        // Process $ref
        if let Some(reference) = json_schema.get("$ref") {
            let ref_str = reference.as_str().ok_or_else(|| {
                anyhow!("Expected string in $ref, got: {}", limited_str(reference))
            })?;
            return self.get_definition(ref_str);
        }

        // Process const
        if let Some(const_value) = json_schema.get("const") {
            return self.gen_json_const(const_value);
        }

        // Process enum
        if let Some(enum_array) = json_schema.opt_array("enum")? {
            let options = enum_array
                .iter()
                .map(|opt| self.gen_json_const(opt))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(self.builder.select(&options));
        }

        // Process type
        if let Some(tp) = json_schema.opt_str("type")? {
            return self.gen_json_type(tp, json_schema)
        }

        let tps = TYPES.iter().map(|s| Value::String(s.to_string())).collect::<Vec<_>>();
        let types = match json_schema.opt_array("type")? {
            Some(types) => types,
            None => &tps,
        };

        let nodes = types
            .iter()
            .map(|v| {
                let tp = v.as_str().ok_or_else(|| {
                    anyhow!("Expected string in type list, got: {}", limited_str(v))
                })?;
                self.gen_json_type(tp, json_schema)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self.builder.select(&nodes))
    }

    fn gen_json_const(&mut self, const_value: &Value) -> Result<NodeRef> {
        // Recursively build a grammar for a constant value (just to play nicely with separators and whitespace flexibility)
        match const_value {
            Value::Object(values) => {
                let properties = values.iter()
                    .map(|(k, v)| {
                        (k.clone(), json!({"const": v}))
                    })
                    .collect::<serde_json::Map<_, _>>();
                self.gen_json_object(&properties, &Value::Bool(false), properties.keys().cloned().collect())
            },
            Value::Array(values) => {
                let n_items = values.len() as u64;
                let prefix_items = values.iter().map(|v| json!({"const": v})).collect::<Vec<_>>();
                self.gen_json_array(&prefix_items, &Value::Bool(false), n_items, Some(n_items))
            },
            _ => {
                // let serde_json dump simple values
                let const_str = to_compact_json(const_value);
                Ok(self.builder.string(&const_str))
            }
        }
    }

    fn gen_json_type(&mut self, target_type_str: &str, json_schema: &Value) -> Result<NodeRef> {
        match target_type_str {
            "null" => return Ok(self.builder.string("null")),
            "boolean" => return Ok(self.lexeme(r"true|false")),
            "integer" | "number" => {
                let mut minimum = json_schema.opt_f64("minimum")?;
                let mut maximum = json_schema.opt_f64("maximum")?;
                let exclusive_minimum: bool = match json_schema.get("exclusiveMinimum") {
                    // Draft4-style boolean
                    Some(Value::Bool(b)) => *b,
                    // Draft 2020-12 style number
                    Some(Value::Number(n)) => {
                        let n = n.as_f64()
                            .ok_or_else(|| expected_err("exclusiveMinimum", &Value::Number(n.clone()), "unsigned integer"))?;
                        match minimum {
                            Some(min) if n < min => {
                                false
                            },
                            _ => {
                                minimum = Some(n);
                                true
                            }
                        }
                    },
                    _ => false,
                };
                let exclusive_maximum: bool = match json_schema.get("exclusiveMaximum") {
                    // Draft4-style boolean
                    Some(Value::Bool(b)) => *b,
                    // Draft 2020-12 style number
                    Some(Value::Number(n)) => {
                        let n = n.as_f64()
                            .ok_or_else(|| expected_err("exclusiveMaximum", &Value::Number(n.clone()), "unsigned integer"))?;
                        match maximum {
                            Some(max) if n > max => {
                                false
                            },
                            _ => {
                                maximum = Some(n);
                                true
                            }
                        }
                    },
                    _ => false,
                };
                if target_type_str == "integer" {
                    return self.json_int(minimum, maximum, exclusive_minimum, exclusive_maximum);
                } else {
                    return self.json_number(minimum, maximum, exclusive_minimum, exclusive_maximum);
                }
            },
            "string" => {
                let min_length = json_schema.opt_u64("minLength")?.unwrap_or(0);
                let max_length = json_schema.opt_u64("maxLength")?;
                let pattern = json_schema.opt_str("pattern")?;
                let format = json_schema.opt_str("format")?;
                return self.gen_json_string(min_length, max_length, pattern, format);
            }
            "array" => {
                let empty = vec![];
                let prefix_items = json_schema.opt_array("prefixItems")?.unwrap_or(&empty);
                let item_schema = json_schema.get("items").unwrap_or(&Value::Bool(true));
                let min_items = json_schema.opt_u64("minItems")?.unwrap_or(0);
                let max_items = json_schema.opt_u64("maxItems")?;
                return self.gen_json_array(prefix_items, item_schema, min_items, max_items);
            }
            "object" => {
                let empty = serde_json::Map::default();
                let properties = json_schema.opt_object("properties")?.unwrap_or(&empty);
                let additional_properties = json_schema
                    .get("additionalProperties")
                    .unwrap_or(&Value::Bool(true));
                let required: Vec<String> = json_schema.opt_array("required")?
                    .map(|values| {
                        values.iter()
                            .map(|v| {
                                v.as_str()
                                    .map(|s| s.to_string()) // Convert &str to String
                                    .ok_or_else(|| expected_err("required", v, "array of string"))
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .unwrap_or_else(|| Ok(vec![]))?;
                return self.gen_json_object(properties, additional_properties, required);
            }
            _ => bail!("Unsupported type in schema: {}", target_type_str),
        }
    }

    fn lexeme(&mut self, rx: &str) -> NodeRef {
        if self.lexeme_cache.contains_key(rx) {
            return self.lexeme_cache[rx];
        }
        let r = self.builder.lexeme(mk_regex(rx), false);
        self.lexeme_cache.insert(rx.to_string(), r);
        r
    }

    fn json_int(&mut self, minimum: Option<f64>, maximum: Option<f64>, exclusive_minimum: bool, exclusive_maximum: bool) -> Result<NodeRef> {
        let minimum = match (minimum, exclusive_minimum) {
            (Some(min_val), true) => {
                if min_val.fract() != 0.0 {
                    Some(min_val.ceil())
                } else {
                    Some(min_val + 1.0)
                }
            }
            (Some(min_val), false) => Some(min_val.ceil()),
            _ => None,
        }.map(|val| val as i64);
        let maximum = match (maximum, exclusive_maximum) {
            (Some(max_val), true) => {
                if max_val.fract() != 0.0 {
                    Some(max_val.floor())
                } else {
                    Some(max_val - 1.0)
                }
            },
            (Some(max_val), false) => Some(max_val.floor()),
            _ => None,
        }.map(|val| val as i64);
        // TODO: handle errors in rx_int_range; currently it just panics
        let rx = rx_int_range(minimum, maximum);
        Ok(self.lexeme(&rx))
    }

    fn json_number(&mut self, minimum: Option<f64>, maximum: Option<f64>, exclusive_minimum: bool, exclusive_maximum: bool) -> Result<NodeRef> {
        // TODO: handle errors in rx_float_range; currently it just panics
        let rx = rx_float_range(minimum, maximum, !exclusive_minimum, !exclusive_maximum);
        Ok(self.lexeme(&rx))
    }

    fn json_simple_string(&mut self) -> NodeRef {
        self.lexeme(&format!("\"{}*\"", CHAR_REGEX))
    }

    fn get_definition(&mut self, reference: &str) -> Result<NodeRef> {
        if let Some(definition) = self.definitions.get(reference) {
            return Ok(*definition);
        }
        let r = self.builder.placeholder();
        self.definitions.insert(reference.to_string(), r);
        self.pending_definitions.push((reference.to_string(), r));
        Ok(r)
    }

    fn gen_json_any(&mut self) -> NodeRef {
        cache!(self.any_cache, {
            let json_any = self.builder.placeholder();
            self.any_cache = Some(json_any); // avoid infinite recursion
            let all_jsons = json!([
                {"type": "null"},
                {"type": "boolean"},
                {"type": "integer"},
                {"type": "number"},
                {"type": "string"},
                {"type": "array", "items": true},
                {"type": "object", "additionalProperties": true},
            ]);
            let ch = all_jsons
                .as_array()
                .unwrap()
                .iter()
                .map(|json_schema| self.gen_json(json_schema))
                .collect::<Result<Vec<_>>>()
                .unwrap();
            let inner = self.builder.select(&ch);
            self.builder.set_placeholder(json_any, inner);
            json_any
        });
    }

    fn gen_json_object(
        &mut self,
        properties: &serde_json::Map<String, Value>,
        additional_properties: &Value,
        required: Vec<String>,
    ) -> Result<NodeRef> {
        let mut illegal_keys: Vec<String> = vec![];
        let mut items: Vec<(NodeRef, bool)> = vec![];
        for name in properties.keys().chain(required.iter().filter(|n| !properties.contains_key(n.as_str()))) {
            let property_schema = properties.get(name).unwrap_or(additional_properties);
            let is_required = required.contains(name);
            if property_schema == &Value::Bool(false) {
                if is_required {
                    bail!("Required property has 'false' schema: {}", name);
                }
                illegal_keys.push(name.clone());
                continue;
            }
            let name = self.builder.string(&format!("\"{}\"", name));
            let colon = self.builder.string(&self.options.key_separator);
            let the_rest = self.gen_json(property_schema)?;
            let item = self.builder.join(&[name, colon, the_rest]);
            items.push((item, is_required));
        }
        if additional_properties != &Value::Bool(false) {
            // TODO: illegal keys and NOT(properties keys)
            let name = self.json_simple_string();
            let colon = self.builder.string(&self.options.key_separator);
            let the_rest = self.gen_json(additional_properties)?;
            let item = self.builder.join(&[name, colon, the_rest]);
            let seq = self.sequence(item);
            items.push((seq, false));
        }
        let opener = self.builder.string("{");
        let inner = self.ordered_sequence(&items, false);
        let closer = self.builder.string("}");
        Ok(self.builder.join(&[opener, inner, closer]))
    }


    fn ordered_sequence(&mut self, items: &[(NodeRef, bool)], prefixed: bool) -> NodeRef {
        if items.is_empty() {
            return self.builder.string("");
        }
        let comma = self.builder.string(&self.options.item_separator);
        let (item, required) = items[0];
        let rest = &items[1..];

        match (prefixed, required) {
            (true, true) => {
                // If we know we have preceeding elements, we can safely just add a (',' + e)
                let rest_seq = self.ordered_sequence(rest, true);
                self.builder.join(&[comma, item, rest_seq])
            },
            (true, false) => {
                // If we know we have preceeding elements, we can safely just add an optional(',' + e)
                // TODO optimization: if the rest is all optional, we can nest the rest in the optional
                let comma_item = self.builder.join(&[comma, item]);
                let optional_comma_item = self.builder.optional(comma_item);
                let rest_seq = self.ordered_sequence(rest, true);
                self.builder.join(&[optional_comma_item, rest_seq])
            },
            (false, true) => {
                // No preceeding elements, so we just add the element (no comma)
                let rest_seq = self.ordered_sequence(rest, true);
                self.builder.join(&[item, rest_seq])
            },
            (false, false) => {
                // No preceeding elements, but our element is optional. If we add the element, the remaining
                // will be prefixed, else they are not.
                // TODO: same nested optimization as above
                let prefixed_rest = self.ordered_sequence(rest, true);
                let unprefixed_rest = self.ordered_sequence(rest, false);
                let opts = [
                    self.builder.join(&[item, prefixed_rest]),
                    unprefixed_rest,
                ];
                self.builder.select(&opts)
            },
        }
    }

    fn sequence(&mut self, item: NodeRef) -> NodeRef {
        let comma = self.builder.string(&self.options.item_separator);
        let item_comma = self.builder.join(&[item, comma]);
        let item_comma_star = self.builder.zero_or_more(item_comma);
        self.builder.join(&[item_comma_star, item])
    }

    fn gen_json_string(
        &mut self,
        min_length: u64,
        max_length: Option<u64>,
        regex: Option<&str>,
        format: Option<&str>,
    ) -> Result<NodeRef> {

        let mut regex = regex;

        if let Some(format) = format {
            if regex.is_some() {
                bail!("Cannot specify both a regex and a format for a JSON string");
            }
            if let Some(r) = lookup_format(format) {
                regex = Some(r);
            } else {
                bail!("Unknown format: {}", format)
            };
        }

        if min_length == 0 && max_length.is_none() && regex.is_none() {
            return Ok(self.json_simple_string());
        }

        if let Some(regex) = regex {
            if min_length > 0 || max_length.is_some() {
                bail!("If a pattern is specified, minLength and maxLength must be unspecified.");
            }
            // the regex has implicit ^...$ anyways
            let regex = regex.trim_start_matches('^').trim_end_matches('$');
            let node = self.builder.lexeme(mk_regex(regex), true);
            Ok(node)
        } else {
            Ok(self.lexeme(&format!(
                "\"{}{{{},{}}}\"",
                CHAR_REGEX,
                min_length,
                max_length.map_or("".to_string(), |v| v.to_string())
            )))
        }
    }

    fn gen_json_array(
        &mut self,
        prefix_items: &[Value],
        item_schema: &Value,
        min_items: u64,
        max_items: Option<u64>,
    ) -> Result<NodeRef> {
        let anything_goes = json!({});
        let item_schema = if item_schema.as_bool() == Some(true) {
            &anything_goes
        } else {
            item_schema
        };
        let item_schema_is_false = item_schema.as_bool() == Some(false);

        if item_schema_is_false && prefix_items.len() < min_items as usize {
            bail!(
                "PrefixItems has too few elements ({}) to satisfy minItems ({}) but no extra items were allowed",
                prefix_items.len(),
                min_items
            );
        }

        if let Some(max_items_value) = max_items {
            if max_items_value < min_items {
                bail!(
                    "maxItems ({}) can't be less than minItems ({})",
                    max_items_value,
                    min_items
                );
            }
        }

        let mut required_items = vec![];
        let mut optional_items = vec![];

        // If max_items is None, we can add an infinite tail of items later
        let n_to_add = max_items.map_or(prefix_items.len().max(min_items as usize), |max| {
            max as usize
        });

        if let Some(item_arr) = item_schema.as_array() {
            for item in item_arr {
                required_items.push(self.gen_json(item)?);
            }
        } else {
            let item_schema_compiled = if item_schema_is_false {
                None
            } else {
                Some(self.gen_json(item_schema)?)
            };

            for i in 0..n_to_add {
                let item = if i < prefix_items.len() {
                    self.gen_json(&prefix_items[i])?
                } else if let Some(compiled) = &item_schema_compiled {
                    compiled.clone()
                } else {
                    break;
                };

                if i < min_items as usize {
                    required_items.push(item);
                } else {
                    optional_items.push(item);
                }
            }

            if max_items.is_none() && !item_schema_is_false {
                // Add an infinite tail of items
                optional_items.push(self.sequence(item_schema_compiled.unwrap()));
            }
        }

        let mut grammars: Vec<NodeRef> = vec![self.builder.string("[")];
        let comma = self.builder.string(&self.options.item_separator);

        if !required_items.is_empty() {
            grammars.push(required_items[0]);
            for item in &required_items[1..] {
                grammars.push(comma);
                grammars.push(*item);
            }
        }

        if !optional_items.is_empty() {
            let first = optional_items[0];
            let tail = optional_items
                .into_iter()
                .skip(1)
                .rev()
                .fold(first, |acc, item| {
                    let j = self.builder.join(&[comma, item, acc]);
                    self.builder.optional(j)
                });

            if !required_items.is_empty() {
                let j = self.builder.join(&[comma, tail]);
                grammars.push(self.builder.optional(j));
            } else {
                grammars.push(self.builder.optional(tail));
            }
        }

        grammars.push(self.builder.string("]"));
        Ok(self.builder.join(&grammars))
    }
}
