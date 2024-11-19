use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use anyhow::{anyhow, bail, Result};
use indexmap::{IndexMap, IndexSet};
use jsonschema::{validator_for, Uri, Validator};
use referencing::{Draft, Registry, Resolver, ResourceRef};
use serde_json::{Map, Value};

const DEFAULT_ROOT_URI: &str = "json-schema:///";
const DEFAULT_DRAFT: Draft = Draft::Draft202012;
const TYPES: [&str; 6] = ["null", "boolean", "number", "string", "array", "object"];

// Keywords that are implemented in this module
const IMPLEMENTED: [&str; 22] = [
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

#[derive(Debug, PartialEq, Clone)]
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
        pattern: Option<String>,
        format: Option<String>,
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
    Const {
        value: Value,
    },
    Enum {
        options: Vec<Value>,
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
            reason: "Schema is false".to_string(),
        }
    }

    fn normalize(self) -> Result<Schema> {
        match self {
            Schema::Any => Ok(self),
            Schema::Unsatisfiable { .. } => Ok(self),
            Schema::Null => Ok(self),
            Schema::Boolean => Ok(self),
            Schema::Number { .. } => {
                // TODO: validation logic, maybe returning Schema::Unsatisfiable
                Ok(self)
            }
            Schema::String { .. } => {
                // TODO: validation logic, maybe returning Schema::Unsatisfiable
                Ok(self)
            }
            Schema::Array {
                min_items,
                max_items,
                prefix_items,
                items,
            } => {
                // TODO: validation logic, maybe returning Schema::Unsatisfiable
                Ok(Schema::Array {
                    min_items,
                    max_items,
                    prefix_items: prefix_items
                        .into_iter()
                        .map(|v| v.normalize())
                        .collect::<Result<Vec<_>>>()?,
                    items: items.map(|v| v.normalize().map(Box::new)).transpose()?,
                })
            }
            Schema::Object {
                properties,
                additional_properties,
                required,
            } => {
                // TODO: validation logic, maybe returning Schema::Unsatisfiable
                Ok(Schema::Object {
                    properties: properties
                        .into_iter()
                        .map(|(k, v)| v.normalize().map(|v| (k, v)))
                        .collect::<Result<IndexMap<_, _>>>()?,
                    additional_properties: additional_properties
                        .map(|v| v.normalize().map(Box::new))
                        .transpose()?,
                    required: required,
                })
            }
            Schema::Const { .. } => Ok(self),
            Schema::Enum { options } => {
                if options.is_empty() {
                    return Ok(Schema::Unsatisfiable {
                        reason: "enum is empty".to_string(),
                    });
                }
                Ok(Schema::Enum { options })
            }
            Schema::AnyOf { options } => {
                if options.is_empty() {
                    return Ok(Schema::Unsatisfiable {
                        reason: "anyOf is empty".to_string(),
                    });
                }
                let mut unsats = Vec::new();
                let mut valid = Vec::new();
                for option in options {
                    let normed = option.normalize()?;
                    match normed {
                        Schema::Unsatisfiable { .. } => unsats.push(normed),
                        // Flatten nested anyOfs
                        Schema::AnyOf { options: nested } => valid.extend(nested),
                        _ => valid.push(normed),
                    }
                }
                if valid.is_empty() {
                    // Return the first unsatisfiable schema for debug-ability
                    return Ok(unsats.swap_remove(0));
                }
                if valid.len() == 1 {
                    return Ok(valid.swap_remove(0));
                }
                Ok(Schema::AnyOf { options: valid })
            }
            Schema::OneOf { options } => {
                if options.is_empty() {
                    return Ok(Schema::Unsatisfiable {
                        reason: "oneOf is empty".to_string(),
                    });
                }
                let mut unsats = Vec::new();
                let mut valid = Vec::new();
                for option in options {
                    let normed = option.normalize()?;
                    match normed {
                        Schema::Unsatisfiable { .. } => unsats.push(normed),
                        // Flatten nested oneOfs: (A⊕B)⊕(C⊕D) = A⊕B⊕C⊕D
                        Schema::OneOf { options: nested } => valid.extend(nested),
                        _ => valid.push(normed),
                    }
                }
                if valid.is_empty() {
                    // Return the first unsatisfiable schema for debug-ability
                    return Ok(unsats.swap_remove(0));
                }
                if valid.len() == 1 {
                    return Ok(valid.swap_remove(0));
                }
                Ok(Schema::OneOf { options: valid })
            }
            // TODO: ?
            Schema::Ref { .. } => Ok(self),
        }
    }
}

struct Context<'a> {
    resolver: Resolver<'a>,
    // TODO: actually use this to handle draft-specific behavior
    draft: Draft,
    defs: Rc<RefCell<HashMap<String, Schema>>>,
    seen: Rc<RefCell<HashSet<String>>>,
}

impl<'a> Context<'a> {
    fn in_subresource(&'a self, resource: ResourceRef) -> Result<Context<'a>> {
        let resolver = self.resolver.in_subresource(resource)?;
        Ok(Context {
            resolver: resolver,
            draft: resource.draft(),
            defs: Rc::clone(&self.defs),
            seen: Rc::clone(&self.seen),
        })
    }

    fn as_resource_ref<'r>(&'a self, contents: &'r Value) -> ResourceRef<'r> {
        self.draft
            .detect(contents)
            .unwrap_or(DEFAULT_DRAFT)
            .create_resource_ref(contents)
    }

    fn normalize_ref(&self, reference: &str) -> Result<Uri<String>> {
        Ok(self
            .resolver
            .resolve_against(&self.resolver.base_uri().borrow(), reference)?
            .normalize())
    }

    fn lookup_resource(&'a self, reference: &str) -> Result<ResourceRef> {
        let resolved = self.resolver.lookup(reference)?;
        Ok(self.as_resource_ref(&resolved.contents()))
    }

    fn insert_ref(&self, uri: &str, schema: Schema) {
        self.defs.borrow_mut().insert(uri.to_string(), schema);
    }

    fn mark_seen(&self, uri: &str) {
        self.seen.borrow_mut().insert(uri.to_string());
    }

    fn been_seen(&self, uri: &str) -> bool {
        self.seen.borrow().contains(uri)
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
}

fn draft_for(value: &Value) -> Draft {
    DEFAULT_DRAFT.detect(value).unwrap_or(DEFAULT_DRAFT)
}

fn instance_if_valid<'a>(instance: &'a Value, validator: &'a Validator) -> Option<&'a Value> {
    if validator.is_valid(instance) {
        Some(instance)
    } else {
        None
    }
}

pub fn build_schema(contents: &Value) -> Result<(Schema, HashMap<String, Schema>)> {
    if let Some(b) = contents.as_bool() {
        if b {
            return Ok((Schema::Any, HashMap::new()));
        } else {
            return Ok((Schema::false_schema(), HashMap::new()));
        }
    }

    let draft = draft_for(contents);
    let resource_ref = draft.create_resource_ref(contents);
    let resource = draft.create_resource(contents.clone());
    let base_uri = resource.id().unwrap_or(DEFAULT_ROOT_URI).to_string();

    let registry = Registry::try_new(&base_uri, resource)?;

    let resolver = registry.try_resolver(&base_uri)?;
    let ctx = Context {
        resolver: resolver,
        draft: draft,
        defs: Rc::new(RefCell::new(HashMap::new())),
        seen: Rc::new(RefCell::new(HashSet::new())),
    };

    let schema = compile_resource(&ctx, resource_ref)?;
    let definitions = ctx.defs.take();
    Ok((schema, definitions))
}

fn compile_resource(ctx: &Context, resource: ResourceRef) -> Result<Schema> {
    let ctx = ctx.in_subresource(resource)?;
    compile_contents(&ctx, resource.contents())
}

fn compile_contents(ctx: &Context, contents: &Value) -> Result<Schema> {
    compile_contents_inner(ctx, contents)?.normalize()
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

    // We don't need to compile the schema if it's just meta and annotations
    if schemadict
        .keys()
        .all(|k| META_AND_ANNOTATIONS.contains(&k.as_str()))
    {
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

    // Short-circuit for const -- don't need to compile the rest of the schema
    if let Some(instance) = schemadict.get("const") {
        let validator = validator_for(contents)?;
        if let Some(instance) = instance_if_valid(instance, &validator) {
            return Ok(Schema::Const {
                value: instance.clone(),
            });
        }
        return Ok(Schema::Unsatisfiable {
            reason: format!(
                "const instance is invalid against parent schema: {:?}",
                instance
            ),
        });
    }

    // Short-circuit for enum -- don't need to compile the rest of the schema
    if let Some(instances) = schemadict.get("enum") {
        let validator: Validator = validator_for(contents)?;
        let instances = instances
            .as_array()
            .ok_or_else(|| anyhow!("enum must be an array"))?;
        let valid_instances = instances
            .iter()
            .filter_map(|instance| instance_if_valid(instance, &validator))
            .cloned()
            .collect::<Vec<_>>();
        if valid_instances.is_empty() {
            return Ok(Schema::Unsatisfiable {
                reason: format!(
                    "enum instances all invalid against parent schema: {:?}",
                    instances
                ),
            });
        }
        return Ok(Schema::Enum {
            options: valid_instances,
        });
    }

    // Make a mutable copy of the schema so we can modify it
    let mut schemadict = schemadict.clone();

    if let Some(all_of) = schemadict.remove("allOf") {
        let all_of = all_of
            .as_array()
            .ok_or_else(|| anyhow!("allOf must be an array"))?;
        let siblings = compile_contents(ctx, &Value::Object(schemadict))?;
        // Short-circuit if schema is already unsatisfiable
        if matches!(siblings, Schema::Unsatisfiable { .. }) {
            return Ok(siblings);
        }
        let options = all_of
            .iter()
            .map(|value| compile_resource(&ctx, ctx.as_resource_ref(value)))
            .collect::<Result<Vec<_>>>()?;
        let merged = intersect(options.into_iter().chain(vec![siblings]).collect())?;
        return Ok(merged);
    }

    if let Some(any_of) = schemadict.remove("anyOf") {
        let any_of = any_of
            .as_array()
            .ok_or_else(|| anyhow!("anyOf must be an array"))?;
        let siblings = compile_contents(ctx, &Value::Object(schemadict))?;
        // Short-circuit if schema is already unsatisfiable
        if matches!(siblings, Schema::Unsatisfiable { .. }) {
            return Ok(siblings);
        }
        let options = any_of
            .into_iter()
            .map(|value| compile_resource(&ctx, ctx.as_resource_ref(value)))
            .map(|res| res.and_then(|schema| intersect_two(schema, siblings.clone())))
            .collect::<Result<Vec<_>>>()?;
        return Ok(Schema::AnyOf { options });
    }

    // TODO: refactor to share code with anyOf
    if let Some(one_of) = schemadict.remove("oneOf") {
        let one_of = one_of
            .as_array()
            .ok_or_else(|| anyhow!("oneOf must be an array"))?;
        let siblings = compile_contents(ctx, &Value::Object(schemadict))?;
        // Short-circuit if schema is already unsatisfiable
        if matches!(siblings, Schema::Unsatisfiable { .. }) {
            return Ok(siblings);
        }
        let options = one_of
            .into_iter()
            .map(|value| compile_resource(&ctx, ctx.as_resource_ref(value)))
            .map(|res| res.and_then(|schema| intersect_two(schema, siblings.clone())))
            .collect::<Result<Vec<_>>>()?;
        return Ok(Schema::OneOf { options });
    }

    if let Some(reference) = schemadict.remove("$ref") {
        let reference = reference
            .as_str()
            .ok_or_else(|| anyhow!("$ref must be a string, got {}", limited_str(&reference)))?
            .to_string();

        let uri: String = ctx.normalize_ref(&reference)?.into_string();
        let siblings = compile_contents(ctx, &Value::Object(schemadict))?;
        if siblings == Schema::Any {
            if !ctx.been_seen(&uri) {
                ctx.mark_seen(&uri);
                let resource = ctx.lookup_resource(&reference)?;
                let resolved_schema = compile_resource(ctx, resource)?;
                ctx.insert_ref(&uri, resolved_schema);
            }
            return Ok(Schema::Ref { uri });
        } else {
            if ctx.been_seen(&uri) {
                bail!("Recursive $refs with sibling keys not implemented")
            }
            let resource = ctx.lookup_resource(&reference)?;
            let resolved_schema = compile_resource(ctx, resource)?;
            return Ok(intersect_two(siblings, resolved_schema)?);
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

fn compile_type(ctx: &Context, tp: &str, schema: &Map<String, Value>) -> Result<Schema> {
    match tp {
        "null" => Ok(Schema::Null),
        "boolean" => Ok(Schema::Boolean),
        "number" | "integer" => compile_numeric(
            schema.get("minimum"),
            schema.get("maximum"),
            schema.get("exclusiveMinimum"),
            schema.get("exclusiveMaximum"),
            tp == "integer",
        ),
        "string" => compile_string(
            schema.get("minLength"),
            schema.get("maxLength"),
            schema.get("pattern"),
            schema.get("format"),
        ),
        "array" => compile_array(
            ctx,
            schema.get("minItems"),
            schema.get("maxItems"),
            schema.get("prefixItems"),
            schema.get("items"),
        ),
        "object" => compile_object(
            ctx,
            schema.get("properties"),
            schema.get("additionalProperties"),
            schema.get("required"),
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
    let pattern = match pattern {
        None => None,
        Some(val) => Some(
            val.as_str()
                .ok_or_else(|| anyhow!("Expected string for 'pattern', got {}", limited_str(val)))?
                .to_string(),
        ),
    };
    let format = match format {
        None => None,
        Some(val) => Some(
            val.as_str()
                .ok_or_else(|| anyhow!("Expected string for 'format', got {}", limited_str(val)))?
                .to_string(),
        ),
    };
    Ok(Schema::String {
        min_length,
        max_length,
        pattern,
        format,
    })
}

fn compile_array(
    ctx: &Context,
    min_items: Option<&Value>,
    max_items: Option<&Value>,
    prefix_items: Option<&Value>,
    items: Option<&Value>,
) -> Result<Schema> {
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

fn intersect(schemas: Vec<Schema>) -> Result<Schema> {
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
        merged = intersect_two(merged, subschema)?;
        if matches!(merged, Schema::Unsatisfiable { .. }) {
            // Early exit if the schema is already unsatisfiable
            break;
        }
    }
    Ok(merged)
}

fn intersect_two(schema0: Schema, schema1: Schema) -> Result<Schema> {
    match (schema0, schema1) {
        (Schema::Any, schema1) => Ok(schema1),
        (schema0, Schema::Any) => Ok(schema0),
        (Schema::Unsatisfiable { reason }, _) => Ok(Schema::Unsatisfiable { reason }),
        (_, Schema::Unsatisfiable { reason }) => Ok(Schema::Unsatisfiable { reason }),
        (Schema::Ref { .. }, _) => Err(anyhow!("$ref with siblings not implemented")),
        (_, Schema::Ref { .. }) => Err(anyhow!("$ref with siblings not implemented")),
        (Schema::OneOf { options }, schema1) => Ok(Schema::OneOf {
            options: options
                .into_iter()
                .map(|opt| intersect_two(opt, schema1.clone()))
                .collect::<Result<Vec<_>>>()?,
        }),
        (schema0, Schema::OneOf { options }) => Ok(Schema::OneOf {
            options: options
                .into_iter()
                .map(|opt| intersect_two(schema0.clone(), opt))
                .collect::<Result<Vec<_>>>()?,
        }),
        (Schema::AnyOf { options }, schema1) => Ok(Schema::AnyOf {
            options: options
                .into_iter()
                .map(|opt| intersect_two(opt, schema1.clone()))
                .collect::<Result<Vec<_>>>()?,
        }),
        (schema0, Schema::AnyOf { options }) => Ok(Schema::AnyOf {
            options: options
                .into_iter()
                .map(|opt| intersect_two(schema0.clone(), opt))
                .collect::<Result<Vec<_>>>()?,
        }),
        (Schema::Null, Schema::Null) => Ok(Schema::Null),
        (Schema::Boolean, Schema::Boolean) => Ok(Schema::Boolean),
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
        ) => Ok(Schema::Number {
            minimum: opt_max(min1, min2),
            maximum: opt_min(max1, max2),
            exclusive_minimum: opt_max(emin1, emin2),
            exclusive_maximum: opt_min(emax1, emax2),
            integer: int1 || int2,
        }),
        (
            Schema::String {
                min_length: min1,
                max_length: max1,
                pattern: pattern1,
                format: format1,
            },
            Schema::String {
                min_length: min2,
                max_length: max2,
                pattern: pattern2,
                format: format2,
            },
        ) => Ok(Schema::String {
            min_length: min1.max(min2),
            max_length: opt_min(max1, max2),
            pattern: match (pattern1, pattern2) {
                (None, None) => None,
                (None, Some(r)) => Some(r),
                (Some(r), None) => Some(r),
                (Some(r1), Some(r2)) => {
                    if r1 == r2 {
                        Some(r1)
                    } else {
                        bail!("intersection of patterns not implemented")
                    }
                }
            },
            format: match (format1, format2) {
                (None, None) => None,
                (None, Some(fmt)) => Some(fmt),
                (Some(fmt), None) => Some(fmt),
                (Some(fmt1), Some(fmt2)) => {
                    if fmt1 == fmt2 {
                        Some(fmt1)
                    } else {
                        bail!("intersection of formats not implemented")
                    }
                }
            },
        }),
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
        ) => Ok(Schema::Array {
            min_items: min1.max(min2),
            max_items: opt_min(max1, max2),
            prefix_items: {
                let len = prefix1.len().max(prefix2.len());
                prefix1.resize(len, items2.as_deref().cloned().unwrap_or(Schema::Any));
                prefix2.resize(len, items1.as_deref().cloned().unwrap_or(Schema::Any));
                prefix1
                    .into_iter()
                    .zip(prefix2.into_iter())
                    .map(|(item1, item2)| intersect_two(item1, item2))
                    .collect::<Result<Vec<_>>>()?
            },
            items: match (items1, items2) {
                (None, None) => None,
                (None, Some(item)) => Some(item),
                (Some(item), None) => Some(item),
                (Some(item1), Some(item2)) => Some(Box::new(intersect_two(*item1, *item2)?)),
            },
        }),
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
                new_props.insert(key, intersect_two(prop1, prop2)?);
            }
            for (key, prop2) in props2.into_iter() {
                let prop1 = add1.as_deref().cloned().unwrap_or(Schema::Any);
                new_props.insert(key, intersect_two(prop1, prop2)?);
            }
            let mut required = req1;
            required.extend(req2);
            Ok(Schema::Object {
                properties: new_props,
                additional_properties: match (add1, add2) {
                    (None, None) => None,
                    (None, Some(add2)) => Some(add2),
                    (Some(add1), None) => Some(add1),
                    (Some(add1), Some(add2)) => Some(Box::new(intersect_two(*add1, *add2)?)),
                },
                required,
            })
        }
        //TODO: get types for error message
        _ => Ok(Schema::Unsatisfiable {
            reason: "incompatible types".to_string(),
        }),
    }
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

mod test {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_linked_list_simple() {
        let schema = json!({
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                },
                "next": {
                    "$ref": "#",
                },
            },
            "additionalProperties": false,
            "required": ["value"],
        });
        let schema = build_schema(&schema).unwrap();
        println!("{:?}", schema);
    }

    #[test]
    fn test_normalize_3() {
        let schema = json!({
            "anyOf": [
                {
                    "type": "string",
                    "minLength": 1,
                },
                {
                    "type": "number",
                    "minimum": 10,
                },
            ],
            "oneOf": [
                {
                    "type": ["string", "integer"],
                    "maxLength": 1,
                },
                {
                    "type": "number",
                    "maximum": 11,
                },
            ],
            "pattern": "^[a-z]+$",
        });
        let schema = build_schema(&schema).unwrap();
        println!("{:?}", schema);
    }

    #[test]
    fn test_ref() {
        let schema = json!({
            "type": "object",
            "properties": {
                "foo": {
                    "$ref": "#ooo",
                },
                "bar": {
                    "$id": "baz.com",
                    "type": "array",
                    "prefixItems": [
                        {
                            "$ref": "#/prefixItems/1"
                        },
                        {
                            "type": "number",
                        },
                    ],
                },
                "baz": {
                    "$anchor": "ooo",
                    "type": "number",
                },
            },
        });
        let schema = build_schema(&schema).unwrap();
        println!("{:?}", schema);
    }
}
