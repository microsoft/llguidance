use anyhow::{anyhow, bail, Result};
use indexmap::IndexMap;
use serde_json::{json, Value};
use std::{collections::HashMap, vec};

use super::formats::lookup_format;
use super::numeric::{rx_float_range, rx_int_range};
use super::schema::{build_schema, Schema};
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

fn json_dumps(target: &serde_json::Value) -> String {
    serde_json::to_string(target).unwrap()
}

#[derive(Debug)]
struct UnsatisfiableSchemaError {
    message: String,
}

impl std::fmt::Display for UnsatisfiableSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unsatisfiable schema: {}", self.message)
    }
}

const CHAR_REGEX: &str = r#"(\\([\"\\\/bfnrt]|u[a-fA-F0-9]{4})|[^\"\\\x00-\x1F\x7F])"#;

fn check_number_bounds(
    minimum: Option<f64>,
    maximum: Option<f64>,
    exclusive_minimum: bool,
    exclusive_maximum: bool,
) -> Result<()> {
    if let (Some(min), Some(max)) = (minimum, maximum) {
        if min > max {
            return Err(anyhow!(UnsatisfiableSchemaError {
                message: format!("minimum ({}) is greater than maximum ({})", min, max),
            }));
        }
        if min == max && (exclusive_minimum || exclusive_maximum) {
            let minimum_repr = if exclusive_minimum {
                "exclusiveMinimum"
            } else {
                "minimum"
            };
            let maximum_repr = if exclusive_maximum {
                "exclusiveMaximum"
            } else {
                "maximum"
            };
            return Err(anyhow!(UnsatisfiableSchemaError {
                message: format!(
                    "{} ({}) is equal to {} ({})",
                    minimum_repr, min, maximum_repr, max
                ),
            }));
        }
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
        return Ok($field.unwrap());
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

        let (compiled_schema, definitions) = build_schema(schema)?;

        let root = self.gen_json(&compiled_schema)?;
        self.builder.set_start_node(root);

        while let Some((path, pl)) = self.pending_definitions.pop() {
            let schema = definitions
                .get(&path)
                .ok_or_else(|| anyhow!("Definition not found: {}", path))?;
            let compiled = self.gen_json(schema)?;
            self.builder.set_placeholder(pl, compiled);
        }

        Ok(())
    }

    fn gen_json(&mut self, json_schema: &Schema) -> Result<NodeRef> {
        match json_schema {
            Schema::Any => self.gen_json_any(),
            Schema::Unsatisfiable { reason } => Err(anyhow!(UnsatisfiableSchemaError {
                message: reason.to_string(),
            })),
            Schema::Null => Ok(self.builder.string("null")),
            Schema::Boolean => Ok(self.lexeme(r"true|false")),
            Schema::Number {
                minimum,
                maximum,
                exclusive_minimum,
                exclusive_maximum,
                integer,
            } => {
                let (minimum, exclusive_minimum) = match (minimum, exclusive_minimum) {
                    (Some(min), Some(xmin)) => {
                        if xmin >= min {
                            (Some(*xmin), true)
                        } else {
                            (Some(*min), false)
                        }
                    }
                    (Some(min), None) => (Some(*min), false),
                    (None, Some(xmin)) => (Some(*xmin), true),
                    (None, None) => (None, false),
                };
                let (maximum, exclusive_maximum) = match (maximum, exclusive_maximum) {
                    (Some(max), Some(xmax)) => {
                        if xmax <= max {
                            (Some(*xmax), true)
                        } else {
                            (Some(*max), false)
                        }
                    }
                    (Some(max), None) => (Some(*max), false),
                    (None, Some(xmax)) => (Some(*xmax), true),
                    (None, None) => (None, false),
                };
                if *integer {
                    self.json_int(minimum, maximum, exclusive_minimum, exclusive_maximum)
                } else {
                    self.json_number(minimum, maximum, exclusive_minimum, exclusive_maximum)
                }
            }
            Schema::String {
                min_length,
                max_length,
                pattern,
                format,
            } => self.gen_json_string(
                *min_length,
                *max_length,
                pattern.as_deref(),
                format.as_deref(),
            ),
            Schema::Array {
                min_items,
                max_items,
                prefix_items,
                items,
            } => self.gen_json_array(
                prefix_items,
                items.as_deref().unwrap_or(&Schema::Any),
                *min_items,
                *max_items,
            ),
            Schema::Object {
                properties,
                additional_properties,
                required,
            } => self.gen_json_object(
                properties,
                additional_properties.as_deref().unwrap_or(&Schema::Any),
                required.iter().cloned().collect(),
            ),
            Schema::Const { value } => self.gen_json_const(value.clone()),
            Schema::Enum { options } => self.gen_json_enum(options.clone()),
            Schema::AnyOf { options } => self.process_any_of(options.clone()),
            Schema::OneOf { options } => self.process_any_of(options.clone()),
            Schema::Ref { uri, .. } => self.get_definition(uri),
        }
    }

    fn process_any_of(&mut self, options: Vec<Schema>) -> Result<NodeRef> {
        let options = options
            .iter()
            .map(|v| self.gen_json(v))
            .collect::<Result<Vec<_>>>()?;
        Ok(self.builder.select(&options))
    }

    fn gen_json_enum(&mut self, options: Vec<Value>) -> Result<NodeRef> {
        let options = options
            .into_iter()
            .map(|v| self.gen_json_const(v))
            .collect::<Result<Vec<_>>>()?;
        Ok(self.builder.select(&options))
    }

    fn gen_json_const(&mut self, const_value: Value) -> Result<NodeRef> {
        // Recursively build a grammar for a constant value (just to play nicely with separators and whitespace flexibility)
        match const_value {
            Value::Object(values) => {
                let properties = IndexMap::from_iter(
                    values
                        .into_iter()
                        .map(|(k, v)| (k, Schema::Const { value: v })),
                );
                let required = properties.keys().cloned().collect();
                self.gen_json_object(&properties, &Schema::false_schema(), required)
            }
            Value::Array(values) => {
                let n_items = values.len() as u64;
                let prefix_items = values
                    .into_iter()
                    .map(|v| Schema::Const { value: v })
                    .collect::<Vec<_>>();
                self.gen_json_array(
                    &prefix_items,
                    &Schema::false_schema(),
                    n_items,
                    Some(n_items),
                )
            }
            _ => {
                // let serde_json dump simple values
                let const_str = json_dumps(&const_value);
                Ok(self.builder.string(&const_str))
            }
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

    fn json_int(
        &mut self,
        minimum: Option<f64>,
        maximum: Option<f64>,
        exclusive_minimum: bool,
        exclusive_maximum: bool,
    ) -> Result<NodeRef> {
        check_number_bounds(minimum, maximum, exclusive_minimum, exclusive_maximum)?;
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
        }
        .map(|val| val as i64);
        let maximum = match (maximum, exclusive_maximum) {
            (Some(max_val), true) => {
                if max_val.fract() != 0.0 {
                    Some(max_val.floor())
                } else {
                    Some(max_val - 1.0)
                }
            }
            (Some(max_val), false) => Some(max_val.floor()),
            _ => None,
        }
        .map(|val| val as i64);
        // TODO: handle errors in rx_int_range; currently it just panics
        let rx = rx_int_range(minimum, maximum);
        Ok(self.lexeme(&rx))
    }

    fn json_number(
        &mut self,
        minimum: Option<f64>,
        maximum: Option<f64>,
        exclusive_minimum: bool,
        exclusive_maximum: bool,
    ) -> Result<NodeRef> {
        check_number_bounds(minimum, maximum, exclusive_minimum, exclusive_maximum)?;
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

    fn gen_json_any(&mut self) -> Result<NodeRef> {
        cache!(self.any_cache, {
            let json_any = self.builder.placeholder();
            self.any_cache = Some(json_any); // avoid infinite recursion
            let options = vec![
                self.builder.string("null"),
                self.builder.lexeme(mk_regex(r"true|false"), false),
                self.json_number(None, None, false, false)?,
                self.json_simple_string(),
                self.gen_json_array(&[], &Schema::Any, 0, None)?,
                self.gen_json_object(&IndexMap::new(), &Schema::Any, vec![])?,
            ];
            let inner = self.builder.select(&options);
            self.builder.set_placeholder(json_any, inner);
            json_any
        });
    }

    fn gen_json_object(
        &mut self,
        properties: &IndexMap<String, Schema>,
        additional_properties: &Schema,
        required: Vec<String>,
    ) -> Result<NodeRef> {
        let mut taken_names: Vec<String> = vec![];
        let mut items: Vec<(NodeRef, bool)> = vec![];
        for name in properties.keys().chain(
            required
                .iter()
                .filter(|n| !properties.contains_key(n.as_str())),
        ) {
            let property_schema = properties.get(name).unwrap_or(additional_properties);
            let is_required = required.contains(name);
            // Quote (and escape) the name
            let quoted_name = json_dumps(&json!(name));
            let property = match self.gen_json(property_schema) {
                Ok(node) => node,
                Err(e) => match e.downcast_ref::<UnsatisfiableSchemaError>() {
                    // If it's not an UnsatisfiableSchemaError, just propagate it normally
                    None => return Err(e),
                    // Property is optional; don't raise UnsatisfiableSchemaError but mark name as taken
                    Some(_) if !is_required => {
                        taken_names.push(quoted_name);
                        continue;
                    }
                    // Property is required; add context and propagate UnsatisfiableSchemaError
                    Some(_) => {
                        return Err(e.context(UnsatisfiableSchemaError {
                            message: format!("required property '{}' is unsatisfiable", name),
                        }));
                    }
                },
            };
            let name = self.builder.string(&quoted_name);
            taken_names.push(quoted_name);
            let colon = self.builder.string(&self.options.key_separator);
            let item = self.builder.join(&[name, colon, property]);
            items.push((item, is_required));
        }

        match self.gen_json(additional_properties) {
            Err(e) => {
                if e.downcast_ref::<UnsatisfiableSchemaError>().is_none() {
                    // Propagate errors that aren't UnsatisfiableSchemaError
                    return Err(e);
                }
                // Ignore UnsatisfiableSchemaError for additionalProperties
            }
            Ok(property) => {
                let name = if taken_names.is_empty() {
                    self.json_simple_string()
                } else {
                    let taken_name_ids = taken_names
                        .iter()
                        .map(|n| self.builder.regex.literal(n.to_string()))
                        .collect::<Vec<_>>();
                    let taken = self.builder.regex.select(taken_name_ids);
                    let not_taken = self.builder.regex.not(taken);
                    let valid = self
                        .builder
                        .regex
                        .regex(r#""([^"\\]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*""#.to_string());
                    let valid_and_not_taken = self.builder.regex.and(vec![valid, not_taken]);
                    let rx = RegexSpec::RegexId(valid_and_not_taken);
                    self.builder.lexeme(rx, false)
                };
                let colon = self.builder.string(&self.options.key_separator);
                let item = self.builder.join(&[name, colon, property]);
                let seq = self.sequence(item);
                items.push((seq, false));
            }
        }
        let opener = self.builder.string("{");
        let inner = self.ordered_sequence(&items, false, &mut HashMap::new());
        let closer = self.builder.string("}");
        Ok(self.builder.join(&[opener, inner, closer]))
    }

    fn ordered_sequence<'a>(
        &mut self,
        items: &'a [(NodeRef, bool)],
        prefixed: bool,
        cache: &mut HashMap<(&'a [(NodeRef, bool)], bool), NodeRef>,
    ) -> NodeRef {
        // Cache to reduce number of nodes from O(n^2) to O(n)
        if let Some(node) = cache.get(&(items, prefixed)) {
            return node.clone();
        }
        if items.is_empty() {
            return self.builder.string("");
        }
        let comma = self.builder.string(&self.options.item_separator);
        let (item, required) = items[0];
        let rest = &items[1..];

        let node = match (prefixed, required) {
            (true, true) => {
                // If we know we have preceeding elements, we can safely just add a (',' + e)
                let rest_seq = self.ordered_sequence(rest, true, cache);
                self.builder.join(&[comma, item, rest_seq])
            }
            (true, false) => {
                // If we know we have preceeding elements, we can safely just add an optional(',' + e)
                // TODO optimization: if the rest is all optional, we can nest the rest in the optional
                let comma_item = self.builder.join(&[comma, item]);
                let optional_comma_item = self.builder.optional(comma_item);
                let rest_seq = self.ordered_sequence(rest, true, cache);
                self.builder.join(&[optional_comma_item, rest_seq])
            }
            (false, true) => {
                // No preceding elements, so we just add the element (no comma)
                let rest_seq = self.ordered_sequence(rest, true, cache);
                self.builder.join(&[item, rest_seq])
            }
            (false, false) => {
                // No preceding elements, but our element is optional. If we add the element, the remaining
                // will be prefixed, else they are not.
                // TODO: same nested optimization as above
                let prefixed_rest = self.ordered_sequence(rest, true, cache);
                let unprefixed_rest = self.ordered_sequence(rest, false, cache);
                let opts = [self.builder.join(&[item, prefixed_rest]), unprefixed_rest];
                self.builder.select(&opts)
            }
        };
        cache.insert((items, prefixed), node.clone());
        node
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
        if let Some(max_length) = max_length {
            if min_length > max_length {
                return Err(anyhow!(UnsatisfiableSchemaError {
                    message: format!(
                        "minLength ({}) is greater than maxLength ({})",
                        min_length, max_length
                    ),
                }));
            }
        }

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
        prefix_items: &[Schema],
        item_schema: &Schema,
        min_items: u64,
        max_items: Option<u64>,
    ) -> Result<NodeRef> {
        let mut max_items = max_items;

        if let Some(max_items) = max_items {
            if min_items > max_items {
                return Err(anyhow!(UnsatisfiableSchemaError {
                    message: format!(
                        "minItems ({}) is greater than maxItems ({})",
                        min_items, max_items
                    ),
                }));
            }
        }

        let additional_item_grm = match self.gen_json(item_schema) {
            Ok(node) => Some(node),
            Err(e) => match e.downcast_ref::<UnsatisfiableSchemaError>() {
                // If it's not an UnsatisfiableSchemaError, just propagate it normally
                None => return Err(e),
                // Item is optional; don't raise UnsatisfiableSchemaError
                Some(_) if prefix_items.len() >= min_items as usize => None,
                // Item is required; add context and propagate UnsatisfiableSchemaError
                Some(_) => {
                    return Err(e.context(UnsatisfiableSchemaError {
                        message: format!("required item is unsatisfiable"),
                    }));
                }
            },
        };

        let mut required_items = vec![];
        let mut optional_items = vec![];

        // If max_items is None, we can add an infinite tail of items later
        let n_to_add = max_items.map_or(prefix_items.len().max(min_items as usize), |max| {
            max as usize
        });

        for i in 0..n_to_add {
            let item = if i < prefix_items.len() {
                match self.gen_json(&prefix_items[i]) {
                    Ok(node) => node,
                    Err(e) => match e.downcast_ref::<UnsatisfiableSchemaError>() {
                        // If it's not an UnsatisfiableSchemaError, just propagate it normally
                        None => return Err(e),
                        // Item is optional; don't raise UnsatisfiableSchemaError.
                        // Set max_items to the current index, as we can't satisfy any more items.
                        Some(_) if i >= min_items as usize => {
                            max_items = Some(i as u64);
                            break;
                        }
                        // Item is required; add context and propagate UnsatisfiableSchemaError
                        Some(_) => {
                            return Err(e.context(UnsatisfiableSchemaError {
                                message: format!(
                                    "prefixItems[{}] is unsatisfiable but minItems is {}",
                                    i, min_items
                                ),
                            }));
                        }
                    },
                }
            } else if let Some(compiled) = &additional_item_grm {
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

        if max_items.is_none() && !additional_item_grm.is_none() {
            // Add an infinite tail of items
            optional_items.push(self.sequence(additional_item_grm.unwrap()));
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
            let tail =
                optional_items
                    .into_iter()
                    .skip(1)
                    .rev()
                    .fold(self.builder.empty(), |acc, item| {
                        let j = self.builder.join(&[comma, item, acc]);
                        self.builder.optional(j)
                    });
            let tail = self.builder.join(&[first, tail]);

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
