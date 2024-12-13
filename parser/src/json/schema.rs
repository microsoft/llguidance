use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    mem,
    rc::Rc,
};

use anyhow::{anyhow, bail, Result};
use derivre::RegexAst;
use indexmap::{IndexMap, IndexSet};
use referencing::{Draft, Registry, Resolver, ResourceRef};
use serde_json::Value;

use super::formats::lookup_format;

const DEFAULT_ROOT_URI: &str = "json-schema:///";
const DEFAULT_DRAFT: Draft = Draft::Draft202012;
const TYPES: [&str; 6] = ["null", "boolean", "number", "string", "array", "object"];

// Keywords that are implemented in this module
const IMPLEMENTED: [&str; 23] = [
    // Core
    "anyOf",
    "oneOf",
    "allOf",
    "$ref",
    "const",
    "enum",
    "type",
    // Array
    "items",
    "additionalItems",
    "prefixItems",
    "minItems",
    "maxItems",
    // Object
    "properties",
    "additionalProperties",
    "required",
    // String
    "minLength",
    "maxLength",
    "pattern",
    "format",
    // Number
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
];

// Keywords that are used for metadata or annotations, not directly driving validation.
// Note that some keywords like $id and $schema affect the behavior of other keywords, but
// they can safely be ignored if other keywords aren't present
const META_AND_ANNOTATIONS: [&str; 15] = [
    "$anchor",
    "$defs",
    "definitions",
    "$schema",
    "$id",
    "id",
    "$comment",
    "title",
    "description",
    "default",
    "readOnly",
    "writeOnly",
    "examples",
    "contentMediaType",
    "contentEncoding",
];

fn limited_str(node: &Value) -> String {
    let s = node.to_string();
    if s.len() > 100 {
        format!("{}...", &s[..100])
    } else {
        s
    }
}

#[derive(Debug, Clone)]
pub enum Schema {
    Any,
    Unsatisfiable {
        reason: String,
    },
    Null,
    Boolean,
    Number {
        minimum: Option<f64>,
        maximum: Option<f64>,
        exclusive_minimum: Option<f64>,
        exclusive_maximum: Option<f64>,
        integer: bool,
    },
    String {
        min_length: u64,
        max_length: Option<u64>,
        regex: Option<RegexAst>,
    },
    Array {
        min_items: u64,
        max_items: Option<u64>,
        prefix_items: Vec<Schema>,
        items: Option<Box<Schema>>,
    },
    Object {
        properties: IndexMap<String, Schema>,
        additional_properties: Option<Box<Schema>>,
        required: IndexSet<String>,
    },
    LiteralBool {
        value: bool,
    },
    AnyOf {
        options: Vec<Schema>,
    },
    OneOf {
        options: Vec<Schema>,
    },
    Ref {
        uri: String,
    },
}

impl Schema {
    pub fn false_schema() -> Schema {
        Schema::Unsatisfiable {
            reason: "schema is false".to_string(),
        }
    }

    /// Shallowly normalize the schema, removing any unnecessary nesting or empty options.
    fn normalize(self) -> Schema {
        match self {
            Schema::AnyOf { options } => {
                let mut unsats = Vec::new();
                let mut valid = Vec::new();
                for option in options.into_iter() {
                    match option {
                        Schema::Any => {
                            return Schema::Any;
                        }
                        Schema::Unsatisfiable { reason } => {
                            unsats.push(Schema::Unsatisfiable { reason })
                        }
                        Schema::AnyOf { options: nested } => valid.extend(nested),
                        other => valid.push(other),
                    }
                }
                if valid.is_empty() {
                    // Return the first unsatisfiable schema for debug-ability
                    if let Some(unsat) = unsats.into_iter().next() {
                        return unsat;
                    }
                    // We must not have had any schemas to begin with
                    return Schema::Unsatisfiable {
                        reason: "anyOf is empty".to_string(),
                    };
                }
                if valid.len() == 1 {
                    // Unwrap singleton
                    return valid.swap_remove(0);
                }
                Schema::AnyOf { options: valid }
            }
            Schema::OneOf { options } => {
                let mut unsats = Vec::new();
                let mut valid = Vec::new();
                for option in options.into_iter() {
                    match option {
                        Schema::Unsatisfiable { reason } => {
                            unsats.push(Schema::Unsatisfiable { reason })
                        }
                        // Flatten nested oneOfs: (A⊕B)⊕(C⊕D) = A⊕B⊕C⊕D
                        Schema::OneOf { options: nested } => valid.extend(nested),
                        other => valid.push(other),
                    }
                }
                if valid.is_empty() {
                    // Return the first unsatisfiable schema for debug-ability
                    if let Some(unsat) = unsats.into_iter().next() {
                        return unsat;
                    }
                    // We must not have had any schemas to begin with
                    return Schema::Unsatisfiable {
                        reason: "oneOf is empty".to_string(),
                    };
                }
                if valid.len() == 1 {
                    // Unwrap singleton
                    return valid.swap_remove(0);
                }
                if valid.iter().enumerate().all(|(i, x)| {
                    valid
                        .iter()
                        .skip(i + 1) // "upper diagonal"
                        .all(|y| x.is_verifiably_disjoint_from(y))
                }) {
                    Schema::AnyOf { options: valid }
                } else {
                    Schema::OneOf { options: valid }
                }
            }
            other_schema => other_schema,
        }
    }

    /// Intersect two schemas, returning a new (normalized) schema that represents the intersection of the two.
    fn intersect(self, other: Schema, ctx: &Context) -> Result<Schema> {
        ctx.increment()?;

        let merged = match (self, other) {
            (Schema::Any, schema1) => schema1,
            (schema0, Schema::Any) => schema0,
            (Schema::Unsatisfiable { reason }, _) => Schema::Unsatisfiable { reason },
            (_, Schema::Unsatisfiable { reason }) => Schema::Unsatisfiable { reason },
            (Schema::Ref { uri }, schema1) => intersect_ref(ctx, &uri, schema1, true)?,
            (schema0, Schema::Ref { uri }) => intersect_ref(ctx, &uri, schema0, false)?,
            (Schema::OneOf { options }, schema1) => Schema::OneOf {
                options: options
                    .into_iter()
                    .map(|opt| opt.intersect(schema1.clone(), ctx))
                    .collect::<Result<Vec<_>>>()?,
            },
            (schema0, Schema::OneOf { options }) => Schema::OneOf {
                options: options
                    .into_iter()
                    .map(|opt| schema0.clone().intersect(opt, ctx))
                    .collect::<Result<Vec<_>>>()?,
            },
            (Schema::AnyOf { options }, schema1) => Schema::AnyOf {
                options: options
                    .into_iter()
                    .map(|opt| opt.intersect(schema1.clone(), ctx))
                    .collect::<Result<Vec<_>>>()?,
            },
            (schema0, Schema::AnyOf { options }) => Schema::AnyOf {
                options: options
                    .into_iter()
                    .map(|opt| schema0.clone().intersect(opt, ctx))
                    .collect::<Result<Vec<_>>>()?,
            },
            (Schema::Null, Schema::Null) => Schema::Null,
            (Schema::Boolean, Schema::Boolean) => Schema::Boolean,
            (Schema::Boolean, Schema::LiteralBool { value }) => Schema::LiteralBool { value },
            (Schema::LiteralBool { value }, Schema::Boolean) => Schema::LiteralBool { value },
            (Schema::LiteralBool { value: value1 }, Schema::LiteralBool { value: value2 }) => {
                if value1 == value2 {
                    Schema::LiteralBool { value: value1 }
                } else {
                    Schema::Unsatisfiable {
                        reason: "incompatible boolean values".to_string(),
                    }
                }
            }
            (
                Schema::Number {
                    minimum: min1,
                    maximum: max1,
                    exclusive_minimum: emin1,
                    exclusive_maximum: emax1,
                    integer: int1,
                },
                Schema::Number {
                    minimum: min2,
                    maximum: max2,
                    exclusive_minimum: emin2,
                    exclusive_maximum: emax2,
                    integer: int2,
                },
            ) => Schema::Number {
                minimum: opt_max(min1, min2),
                maximum: opt_min(max1, max2),
                exclusive_minimum: opt_max(emin1, emin2),
                exclusive_maximum: opt_min(emax1, emax2),
                integer: int1 || int2,
            },
            (
                Schema::String {
                    min_length: min1,
                    max_length: max1,
                    regex: r1,
                },
                Schema::String {
                    min_length: min2,
                    max_length: max2,
                    regex: r2,
                },
            ) => Schema::String {
                min_length: min1.max(min2),
                max_length: opt_min(max1, max2),
                regex: match (r1, r2) {
                    (None, None) => None,
                    (None, Some(r)) => Some(r),
                    (Some(r), None) => Some(r),
                    (Some(r1), Some(r2)) => Some(RegexAst::And(vec![r1, r2])),
                },
            },
            (
                Schema::Array {
                    min_items: min1,
                    max_items: max1,
                    prefix_items: mut prefix1,
                    items: items1,
                },
                Schema::Array {
                    min_items: min2,
                    max_items: max2,
                    prefix_items: mut prefix2,
                    items: items2,
                },
            ) => Schema::Array {
                min_items: min1.max(min2),
                max_items: opt_min(max1, max2),
                prefix_items: {
                    let len = prefix1.len().max(prefix2.len());
                    prefix1.resize_with(len, || items1.as_deref().cloned().unwrap_or(Schema::Any));
                    prefix2.resize_with(len, || items2.as_deref().cloned().unwrap_or(Schema::Any));
                    prefix1
                        .into_iter()
                        .zip(prefix2.into_iter())
                        .map(|(item1, item2)| item1.intersect(item2, ctx))
                        .collect::<Result<Vec<_>>>()?
                },
                items: match (items1, items2) {
                    (None, None) => None,
                    (None, Some(item)) => Some(item),
                    (Some(item), None) => Some(item),
                    (Some(item1), Some(item2)) => Some(Box::new((*item1).intersect(*item2, ctx)?)),
                },
            },
            (
                Schema::Object {
                    properties: props1,
                    additional_properties: add1,
                    required: req1,
                },
                Schema::Object {
                    properties: mut props2,
                    additional_properties: add2,
                    required: req2,
                },
            ) => {
                let mut new_props = IndexMap::new();
                for (key, prop1) in props1.into_iter() {
                    let prop2 = props2
                        .shift_remove(&key)
                        .or_else(|| add2.as_deref().cloned())
                        .unwrap_or(Schema::Any);
                    new_props.insert(key, prop1.intersect(prop2, ctx)?);
                }
                for (key, prop2) in props2.into_iter() {
                    let prop1 = add1.as_deref().cloned().unwrap_or(Schema::Any);
                    new_props.insert(key, prop1.intersect(prop2, ctx)?);
                }
                let mut required = req1;
                required.extend(req2);
                Schema::Object {
                    properties: new_props,
                    additional_properties: match (add1, add2) {
                        (None, None) => None,
                        (None, Some(add2)) => Some(add2),
                        (Some(add1), None) => Some(add1),
                        (Some(add1), Some(add2)) => Some(Box::new((*add1).intersect(*add2, ctx)?)),
                    },
                    required,
                }
            }
            //TODO: get types for error message
            _ => Schema::Unsatisfiable {
                reason: "incompatible types".to_string(),
            },
        };
        Ok(merged.normalize())
    }

    fn is_verifiably_disjoint_from(&self, other: &Schema) -> bool {
        match (self, other) {
            (Schema::Unsatisfiable { .. }, _) => true,
            (_, Schema::Unsatisfiable { .. }) => true,
            (Schema::Any, _) => false,
            (_, Schema::Any) => false,
            (Schema::Boolean, Schema::LiteralBool { .. }) => false,
            (Schema::LiteralBool { .. }, Schema::Boolean) => false,
            (Schema::Ref { .. }, _) => false, // TODO: could resolve
            (_, Schema::Ref { .. }) => false, // TODO: could resolve
            (Schema::LiteralBool { value: value1 }, Schema::LiteralBool { value: value2 }) => {
                value1 != value2
            }
            (Schema::AnyOf { options }, _) => options
                .iter()
                .all(|opt| opt.is_verifiably_disjoint_from(other)),
            (_, Schema::AnyOf { options }) => options
                .iter()
                .all(|opt| self.is_verifiably_disjoint_from(opt)),
            (Schema::OneOf { options }, _) => options
                .iter()
                .all(|opt| opt.is_verifiably_disjoint_from(other)),
            (_, Schema::OneOf { options }) => options
                .iter()
                .all(|opt| self.is_verifiably_disjoint_from(opt)),
            // TODO: could actually compile the regexes and check for overlap
            (
                Schema::String {
                    regex: Some(RegexAst::Literal(lit1)),
                    ..
                },
                Schema::String {
                    regex: Some(RegexAst::Literal(lit2)),
                    ..
                },
            ) => lit1 != lit2,
            (
                Schema::Object {
                    properties: props1,
                    required: req1,
                    additional_properties: add1,
                },
                Schema::Object {
                    properties: props2,
                    required: req2,
                    additional_properties: add2,
                },
            ) => req1.intersection(req2).any(|key| {
                let prop1 = props1
                    .get(key)
                    .unwrap_or(add1.as_deref().unwrap_or(&Schema::Any));
                let prop2 = props2
                    .get(key)
                    .unwrap_or(add2.as_deref().unwrap_or(&Schema::Any));
                prop1.is_verifiably_disjoint_from(prop2)
            }),
            _ => {
                // Except for in the cases above, it should suffice to check that the types are different
                mem::discriminant(self) != mem::discriminant(other)
            }
        }
    }
}

#[derive(Clone)]
struct SchemaBuilderOptions {
    max_size: usize,
}

impl Default for SchemaBuilderOptions {
    fn default() -> Self {
        SchemaBuilderOptions { max_size: 50_000 }
    }
}

struct SharedContext {
    defs: HashMap<String, Schema>,
    seen: HashSet<String>,
    n_compiled: usize,
}

impl SharedContext {
    fn new() -> Self {
        SharedContext {
            defs: HashMap::new(),
            seen: HashSet::new(),
            n_compiled: 0,
        }
    }
}

struct Context<'a> {
    resolver: Resolver<'a>,
    draft: Draft,
    shared: Rc<RefCell<SharedContext>>,
    options: SchemaBuilderOptions,
}

impl<'a> Context<'a> {
    fn in_subresource(&'a self, resource: ResourceRef) -> Result<Context<'a>> {
        let resolver = self.resolver.in_subresource(resource)?;
        Ok(Context {
            resolver: resolver,
            draft: resource.draft(),
            shared: Rc::clone(&self.shared),
            options: self.options.clone(),
        })
    }

    fn as_resource_ref<'r>(&'a self, contents: &'r Value) -> ResourceRef<'r> {
        self.draft
            .detect(contents)
            .unwrap_or(DEFAULT_DRAFT)
            .create_resource_ref(contents)
    }

    fn normalize_ref(&self, reference: &str) -> Result<String> {
        Ok(self
            .resolver
            .resolve_against(&self.resolver.base_uri().borrow(), reference)?
            .normalize()
            .into_string())
    }

    fn lookup_resource(&'a self, reference: &str) -> Result<ResourceRef<'a>> {
        let resolved = self.resolver.lookup(reference)?;
        Ok(self.as_resource_ref(&resolved.contents()))
    }

    fn insert_ref(&self, uri: &str, schema: Schema) {
        self.shared
            .borrow_mut()
            .defs
            .insert(uri.to_string(), schema);
    }

    fn get_ref_cloned(&self, uri: &str) -> Option<Schema> {
        self.shared.borrow().defs.get(uri).cloned()
    }

    fn mark_seen(&self, uri: &str) {
        self.shared.borrow_mut().seen.insert(uri.to_string());
    }

    fn been_seen(&self, uri: &str) -> bool {
        self.shared.borrow().seen.contains(uri)
    }

    fn is_valid_keyword(&self, keyword: &str) -> bool {
        if !self.draft.is_known_keyword(keyword)
            || IMPLEMENTED.contains(&keyword)
            || META_AND_ANNOTATIONS.contains(&keyword)
        {
            return true;
        }
        return false;
    }

    fn increment(&self) -> Result<()> {
        let mut shared = self.shared.borrow_mut();
        shared.n_compiled += 1;
        if shared.n_compiled > self.options.max_size {
            bail!("schema too large");
        }
        Ok(())
    }
}

fn draft_for(value: &Value) -> Draft {
    DEFAULT_DRAFT.detect(value).unwrap_or(DEFAULT_DRAFT)
}

pub fn build_schema(contents: Value) -> Result<(Schema, HashMap<String, Schema>)> {
    if let Some(b) = contents.as_bool() {
        if b {
            return Ok((Schema::Any, HashMap::new()));
        } else {
            return Ok((Schema::false_schema(), HashMap::new()));
        }
    }

    let draft = draft_for(&contents);
    let resource = draft.create_resource(contents);
    let base_uri = resource.id().unwrap_or(DEFAULT_ROOT_URI).to_string();

    let registry = Registry::try_new(&base_uri, resource)?;

    let resolver = registry.try_resolver(&base_uri)?;
    let ctx = Context {
        resolver: resolver,
        draft: draft,
        shared: Rc::new(RefCell::new(SharedContext::new())),
        options: SchemaBuilderOptions::default(),
    };

    let root_resource = ctx.lookup_resource(&base_uri)?;
    let schema = compile_resource(&ctx, root_resource)?;
    let defs = std::mem::take(&mut ctx.shared.borrow_mut().defs);
    Ok((schema, defs))
}

fn compile_resource(ctx: &Context, resource: ResourceRef) -> Result<Schema> {
    let ctx = ctx.in_subresource(resource)?;
    compile_contents(&ctx, resource.contents())
}

fn compile_contents(ctx: &Context, contents: &Value) -> Result<Schema> {
    compile_contents_inner(ctx, contents).map(|schema| schema.normalize())
}

fn compile_contents_inner(ctx: &Context, contents: &Value) -> Result<Schema> {
    if let Some(b) = contents.as_bool() {
        if b {
            return Ok(Schema::Any);
        } else {
            return Ok(Schema::false_schema());
        }
    }

    // Get the schema as an object
    // TODO: validate against metaschema & check for unimplemented keys
    let schemadict = contents
        .as_object()
        .ok_or_else(|| anyhow!("schema must be an object or boolean"))?;

    // Make a mutable copy of the schema so we can modify it
    let schemadict = schemadict
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<HashMap<_, _>>();

    compile_contents_map(ctx, schemadict)
}

fn only_meta_and_annotations(schemadict: &HashMap<&str, &Value>) -> bool {
    schemadict.keys().all(|k| META_AND_ANNOTATIONS.contains(k))
}

fn compile_contents_map(ctx: &Context, mut schemadict: HashMap<&str, &Value>) -> Result<Schema> {
    ctx.increment()?;

    // We don't need to compile the schema if it's just meta and annotations
    if only_meta_and_annotations(&schemadict) {
        return Ok(Schema::Any);
    }

    // Check for unimplemented keys and bail if any are found
    let unimplemented_keys = schemadict
        .keys()
        .filter(|k| !ctx.is_valid_keyword(k))
        .collect::<Vec<_>>();
    if unimplemented_keys.len() > 0 {
        bail!("Unimplemented keys: {:?}", unimplemented_keys);
    }

    if let Some(instance) = schemadict.remove("const") {
        let const_schema = compile_const(instance)?;
        let siblings = compile_contents_map(ctx, schemadict)?;
        return const_schema.intersect(siblings, ctx);
    }

    if let Some(instances) = schemadict.remove("enum") {
        let instances = instances
            .as_array()
            .ok_or_else(|| anyhow!("enum must be an array"))?;
        let siblings = compile_contents_map(ctx, schemadict)?;
        // Short-circuit if schema is already unsatisfiable
        if matches!(siblings, Schema::Unsatisfiable { .. }) {
            return Ok(siblings);
        }
        let options = instances
            .into_iter()
            .map(|instance| compile_const(instance))
            .map(|res| res.and_then(|schema| schema.intersect(siblings.clone(), ctx)))
            .collect::<Result<Vec<_>>>()?;
        return Ok(Schema::AnyOf { options });
    }

    if let Some(all_of) = schemadict.remove("allOf") {
        let all_of = all_of
            .as_array()
            .ok_or_else(|| anyhow!("allOf must be an array"))?;
        let siblings = compile_contents_map(ctx, schemadict)?;
        // Short-circuit if schema is already unsatisfiable
        if matches!(siblings, Schema::Unsatisfiable { .. }) {
            return Ok(siblings);
        }
        let options = all_of
            .iter()
            .map(|value| compile_resource(&ctx, ctx.as_resource_ref(value)))
            .collect::<Result<Vec<_>>>()?;
        let merged = intersect(
            ctx,
            vec![siblings]
                .into_iter()
                .chain(options.into_iter())
                .collect(),
        )?;
        return Ok(merged);
    }

    if let Some(any_of) = schemadict.remove("anyOf") {
        let any_of = any_of
            .as_array()
            .ok_or_else(|| anyhow!("anyOf must be an array"))?;
        let siblings = compile_contents_map(ctx, schemadict)?;
        // Short-circuit if schema is already unsatisfiable
        if matches!(siblings, Schema::Unsatisfiable { .. }) {
            return Ok(siblings);
        }
        let options = any_of
            .into_iter()
            .map(|value| compile_resource(&ctx, ctx.as_resource_ref(value)))
            .map(|res| res.and_then(|schema| siblings.clone().intersect(schema, ctx)))
            .collect::<Result<Vec<_>>>()?;
        return Ok(Schema::AnyOf { options });
    }

    // TODO: refactor to share code with anyOf
    if let Some(one_of) = schemadict.remove("oneOf") {
        let one_of = one_of
            .as_array()
            .ok_or_else(|| anyhow!("oneOf must be an array"))?;
        let siblings = compile_contents_map(ctx, schemadict)?;
        // Short-circuit if schema is already unsatisfiable
        if matches!(siblings, Schema::Unsatisfiable { .. }) {
            return Ok(siblings);
        }
        let options = one_of
            .into_iter()
            .map(|value| compile_resource(&ctx, ctx.as_resource_ref(value)))
            .map(|res| res.and_then(|schema| siblings.clone().intersect(schema, ctx)))
            .collect::<Result<Vec<_>>>()?;
        return Ok(Schema::OneOf { options }.normalize());
    }

    if let Some(reference) = schemadict.remove("$ref") {
        let reference = reference
            .as_str()
            .ok_or_else(|| anyhow!("$ref must be a string, got {}", limited_str(&reference)))?
            .to_string();

        let uri: String = ctx.normalize_ref(&reference)?;
        let siblings = compile_contents_map(ctx, schemadict)?;
        if matches!(siblings, Schema::Any) {
            define_ref(ctx, &uri)?;
            return Ok(Schema::Ref { uri });
        } else {
            return intersect_ref(ctx, &uri, siblings, false);
        }
    }

    let types = match schemadict.remove("type") {
        Some(Value::String(tp)) => {
            return compile_type(ctx, &tp, &schemadict);
        }
        Some(Value::Array(types)) => types
            .iter()
            .map(|tp| match tp.as_str() {
                Some(tp) => Ok(tp.to_string()),
                None => bail!("type must be a string"),
            })
            .collect::<Result<Vec<String>>>()?,
        Some(_) => bail!("type must be a string or array"),
        None => TYPES.iter().map(|s| s.to_string()).collect(),
    };

    // Shouldn't need siblings here since we've already handled allOf, anyOf, oneOf, and $ref
    let options = types
        .iter()
        .map(|tp| compile_type(ctx, &tp, &schemadict))
        .collect::<Result<Vec<Schema>>>()?;
    Ok(Schema::AnyOf { options })
}

fn define_ref(ctx: &Context, ref_uri: &str) -> Result<()> {
    if !ctx.been_seen(ref_uri) {
        ctx.mark_seen(ref_uri);
        let resource = ctx.lookup_resource(ref_uri)?;
        let resolved_schema = compile_resource(ctx, resource)?;
        ctx.insert_ref(&ref_uri, resolved_schema);
    }
    Ok(())
}

fn intersect_ref(ctx: &Context, ref_uri: &str, schema: Schema, ref_first: bool) -> Result<Schema> {
    define_ref(ctx, ref_uri)?;
    let resolved_schema = ctx
        .get_ref_cloned(ref_uri)
        // The ref might not have been defined if we're in a recursive loop and every ref in the loop
        // has a sibling key.
        // TODO: add an extra layer of indirection by defining a URI for the current location (e.g. by hashing the serialized sibling schema)
        // and returning a ref to that URI here to break the loop.
        .ok_or_else(|| {
            anyhow!(
                "circular references with sibling keys are not supported: {}",
                ref_uri
            )
        })?;
    if ref_first {
        resolved_schema.intersect(schema, ctx)
    } else {
        schema.intersect(resolved_schema, ctx)
    }
}

fn compile_const(instance: &Value) -> Result<Schema> {
    match instance {
        Value::Null => Ok(Schema::Null),
        Value::Bool(b) => Ok(Schema::LiteralBool { value: *b }),
        Value::Number(n) => {
            let value = n.as_f64().ok_or_else(|| {
                anyhow!(
                    "Expected f64 for numeric const, got {}",
                    limited_str(instance)
                )
            })?;
            Ok(Schema::Number {
                minimum: Some(value),
                maximum: Some(value),
                exclusive_minimum: None,
                exclusive_maximum: None,
                integer: n.is_i64(),
            })
        }
        Value::String(s) => Ok(Schema::String {
            min_length: 0,
            max_length: None,
            regex: Some(RegexAst::Literal(s.to_string())),
        }),
        Value::Array(items) => {
            let prefix_items = items
                .iter()
                .map(|item| compile_const(item))
                .collect::<Result<Vec<Schema>>>()?;
            Ok(Schema::Array {
                min_items: prefix_items.len() as u64,
                max_items: Some(prefix_items.len() as u64),
                prefix_items,
                items: Some(Box::new(Schema::false_schema())),
            })
        }
        Value::Object(mapping) => {
            let properties = mapping
                .iter()
                .map(|(k, v)| Ok((k.clone(), compile_const(v)?)))
                .collect::<Result<IndexMap<String, Schema>>>()?;
            let required = properties.keys().cloned().collect();
            Ok(Schema::Object {
                properties,
                additional_properties: Some(Box::new(Schema::false_schema())),
                required,
            })
        }
    }
}

fn compile_type(ctx: &Context, tp: &str, schema: &HashMap<&str, &Value>) -> Result<Schema> {
    ctx.increment()?;

    let get = |key: &str| schema.get(key).map(|v| *v);

    match tp {
        "null" => Ok(Schema::Null),
        "boolean" => Ok(Schema::Boolean),
        "number" | "integer" => compile_numeric(
            get("minimum"),
            get("maximum"),
            get("exclusiveMinimum"),
            get("exclusiveMaximum"),
            tp == "integer",
        ),
        "string" => compile_string(
            get("minLength"),
            get("maxLength"),
            get("pattern"),
            get("format"),
        ),
        "array" => compile_array(
            ctx,
            get("minItems"),
            get("maxItems"),
            get("prefixItems"),
            get("items"),
            get("additionalItems"),
        ),
        "object" => compile_object(
            ctx,
            get("properties"),
            get("additionalProperties"),
            get("required"),
        ),
        _ => bail!("Invalid type: {}", tp),
    }
}

fn compile_numeric(
    minimum: Option<&Value>,
    maximum: Option<&Value>,
    exclusive_minimum: Option<&Value>,
    exclusive_maximum: Option<&Value>,
    integer: bool,
) -> Result<Schema> {
    let minimum = match minimum {
        None => None,
        Some(val) => Some(
            val.as_f64()
                .ok_or_else(|| anyhow!("Expected f64 for 'minimum', got {}", limited_str(val)))?,
        ),
    };
    let maximum = match maximum {
        None => None,
        Some(val) => Some(
            val.as_f64()
                .ok_or_else(|| anyhow!("Expected f64 for 'maximum', got {}", limited_str(val)))?,
        ),
    };
    // TODO: actually use ctx.draft to determine which style of exclusiveMinimum/Maximum to use
    let exclusive_minimum = match exclusive_minimum {
        // Draft4-style boolean values
        None | Some(Value::Bool(false)) => None,
        Some(Value::Bool(true)) => minimum,
        // Draft2020-12-style numeric values
        Some(value) => Some(value.as_f64().ok_or_else(|| {
            anyhow!(
                "Expected f64 for 'exclusiveMinimum', got {}",
                limited_str(value)
            )
        })?),
    };
    let exclusive_maximum = match exclusive_maximum {
        // Draft4-style boolean values
        None | Some(Value::Bool(false)) => None,
        Some(Value::Bool(true)) => maximum,
        // Draft2020-12-style numeric values
        Some(value) => Some(value.as_f64().ok_or_else(|| {
            anyhow!(
                "Expected f64 for 'exclusiveMaximum', got {}",
                limited_str(value)
            )
        })?),
    };
    Ok(Schema::Number {
        minimum,
        maximum,
        exclusive_minimum,
        exclusive_maximum,
        integer: integer,
    })
}

fn pattern_to_regex(pattern: &str) -> RegexAst {
    let left_anchored = pattern.starts_with('^');
    let right_anchored = pattern.ends_with('$');
    let trimmed = pattern.trim_start_matches('^').trim_end_matches('$');
    let mut result = String::new();
    if !left_anchored {
        result.push_str(".*");
    }
    // without parens, for a|b we would get .*a|b.* which is (.*a)|(b.*)
    result.push_str("(");
    result.push_str(trimmed);
    result.push_str(")");
    if !right_anchored {
        result.push_str(".*");
    }
    RegexAst::Regex(result)
}

fn compile_string(
    min_length: Option<&Value>,
    max_length: Option<&Value>,
    pattern: Option<&Value>,
    format: Option<&Value>,
) -> Result<Schema> {
    let min_length = match min_length {
        None => 0,
        Some(val) => val
            .as_u64()
            .ok_or_else(|| anyhow!("Expected u64 for 'minLength', got {}", limited_str(val)))?,
    };
    let max_length = match max_length {
        None => None,
        Some(val) => Some(
            val.as_u64()
                .ok_or_else(|| anyhow!("Expected u64 for 'maxLength', got {}", limited_str(val)))?,
        ),
    };
    let pattern_rx = match pattern {
        None => None,
        Some(val) => Some({
            let s = val
                .as_str()
                .ok_or_else(|| anyhow!("Expected string for 'pattern', got {}", limited_str(val)))?
                .to_string();
            pattern_to_regex(&s)
        }),
    };
    let format_rx = match format {
        None => None,
        Some(val) => Some({
            let key = val
                .as_str()
                .ok_or_else(|| anyhow!("Expected string for 'format', got {}", limited_str(val)))?
                .to_string();
            let fmt = lookup_format(&key).ok_or_else(|| anyhow!("Unknown format: {}", key))?;
            pattern_to_regex(&fmt)
        }),
    };
    let regex = match (pattern_rx, format_rx) {
        (None, None) => None,
        (None, Some(fmt)) => Some(fmt),
        (Some(pat), None) => Some(pat),
        (Some(pat), Some(fmt)) => Some(RegexAst::And(vec![pat, fmt])),
    };
    Ok(Schema::String {
        min_length,
        max_length,
        regex: regex,
    })
}

fn compile_array(
    ctx: &Context,
    min_items: Option<&Value>,
    max_items: Option<&Value>,
    prefix_items: Option<&Value>,
    items: Option<&Value>,
    additional_items: Option<&Value>,
) -> Result<Schema> {
    let (prefix_items, items) = {
        // Note that draft detection falls back to Draft202012 if the draft is unknown, so let's relax the draft constraint a bit
        // and assume we're in an old draft if additionalItems is present or items is an array
        if ctx.draft <= Draft::Draft201909
            || additional_items.is_some()
            || matches!(items, Some(Value::Array(..)))
        {
            match (items, additional_items) {
                // Treat array items as prefixItems and additionalItems as items in draft 2019-09 and earlier
                (Some(Value::Array(..)), _) => (items, additional_items),
                // items is treated as items, and additionalItems is ignored if items is not an array (or is missing)
                _ => (None, items),
            }
        } else {
            (prefix_items, items)
        }
    };
    let min_items = match min_items {
        None => 0,
        Some(val) => val
            .as_u64()
            .ok_or_else(|| anyhow!("Expected u64 for 'minItems', got {}", limited_str(val)))?,
    };
    let max_items = match max_items {
        None => None,
        Some(val) => Some(
            val.as_u64()
                .ok_or_else(|| anyhow!("Expected u64 for 'maxItems', got {}", limited_str(val)))?,
        ),
    };
    let prefix_items = match prefix_items {
        None => vec![],
        Some(val) => val
            .as_array()
            .ok_or_else(|| anyhow!("Expected array for 'prefixItems', got {}", limited_str(val)))?
            .iter()
            .map(|item| compile_resource(&ctx, ctx.as_resource_ref(item)))
            .collect::<Result<Vec<Schema>>>()?,
    };
    let items = match items {
        None => None,
        Some(val) => Some(Box::new(compile_resource(&ctx, ctx.as_resource_ref(val))?)),
    };
    Ok(Schema::Array {
        min_items,
        max_items,
        prefix_items,
        items,
    })
}

fn compile_object(
    ctx: &Context,
    properties: Option<&Value>,
    additional_properties: Option<&Value>,
    required: Option<&Value>,
) -> Result<Schema> {
    let properties = match properties {
        None => IndexMap::new(),
        Some(val) => val
            .as_object()
            .ok_or_else(|| anyhow!("Expected object for 'properties', got {}", limited_str(val)))?
            .iter()
            .map(|(k, v)| compile_resource(&ctx, ctx.as_resource_ref(v)).map(|v| (k.clone(), v)))
            .collect::<Result<IndexMap<String, Schema>>>()?,
    };
    let additional_properties = match additional_properties {
        None => None,
        Some(val) => Some(Box::new(compile_resource(&ctx, ctx.as_resource_ref(val))?)),
    };
    let required = match required {
        None => IndexSet::new(),
        Some(val) => val
            .as_array()
            .ok_or_else(|| anyhow!("Expected array for 'required', got {}", limited_str(val)))?
            .iter()
            .map(|item| {
                item.as_str()
                    .ok_or_else(|| {
                        anyhow!(
                            "Expected string for 'required' item, got {}",
                            limited_str(item)
                        )
                    })
                    .map(|s| s.to_string())
            })
            .collect::<Result<IndexSet<String>>>()?,
    };
    Ok(Schema::Object {
        properties,
        additional_properties,
        required,
    })
}

fn intersect(ctx: &Context, schemas: Vec<Schema>) -> Result<Schema> {
    let (schemas, unsatisfiable) = schemas
        .into_iter()
        // "Any" schemas can be ignored
        .filter(|schema| !matches!(schema, Schema::Any))
        // Split into unsatisfiable and satisfiable schemas
        .partition::<Vec<_>, _>(|schema| !matches!(schema, Schema::Unsatisfiable { .. }));

    if let Some(schema) = unsatisfiable.into_iter().next() {
        return Ok(schema);
    }

    let mut merged = Schema::Any;
    for subschema in schemas.into_iter() {
        merged = merged.intersect(subschema, ctx)?;
        if matches!(merged, Schema::Unsatisfiable { .. }) {
            // Early exit if the schema is already unsatisfiable
            break;
        }
    }
    Ok(merged)
}

fn opt_max<T: PartialOrd>(a: Option<T>, b: Option<T>) -> Option<T> {
    match (a, b) {
        (Some(a), Some(b)) => {
            if a >= b {
                Some(a)
            } else {
                Some(b)
            }
        }
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn opt_min<T: PartialOrd>(a: Option<T>, b: Option<T>) -> Option<T> {
    match (a, b) {
        (Some(a), Some(b)) => {
            if a <= b {
                Some(a)
            } else {
                Some(b)
            }
        }
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}
