use std::{cell::RefCell, collections::HashMap, rc::Rc};

use anyhow::{anyhow, bail, Result};
use indexmap::{IndexMap, IndexSet};
use jsonschema::Uri;
use referencing::{Draft, Registry, Resolver, ResourceRef};
use serde_json::{Map, Value};

const DEFAULT_ROOT_URI: &str = "json-schema:///";
const DEFAULT_DRAFT: Draft = Draft::Draft202012;
const TYPES: [&str; 6] = ["null", "boolean", "number", "string", "array", "object"];

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

impl From<bool> for Schema {
    fn from(b: bool) -> Schema {
        if b {
            Schema::Any
        } else {
            Schema::Unsatisfiable {
                reason: "Schema is false".to_string(),
            }
        }
    }
}

impl Schema {
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
}

impl<'a> Context<'a> {
    fn in_subresource(&'a self, resource: ResourceRef) -> Result<Context<'a>> {
        let resolver = self.resolver.in_subresource(resource)?;
        Ok(Context {
            resolver: resolver,
            draft: resource.draft(),
            defs: Rc::clone(&self.defs),
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

    fn contains_ref(&self, uri: &str) -> bool {
        self.defs.borrow().contains_key(uri)
    }
}

fn draft_for(value: &Value) -> Draft {
    DEFAULT_DRAFT.detect(value).unwrap_or(DEFAULT_DRAFT)
}

pub fn build_schema(contents: &Value) -> Result<(Schema, HashMap<String, Schema>)> {
    if let Some(b) = contents.as_bool() {
        return Ok((b.into(), HashMap::new()));
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
        return Ok(b.into());
    }

    // Get the schema as an object
    // TODO: validate against metaschema & check for unimplemented keys
    let schemadict = contents
        .as_object()
        .ok_or_else(|| anyhow!("schema must be an object or boolean"))?;

    if schemadict.is_empty() {
        // TODO: should be ok to have ignored keys here
        return Ok(Schema::Any);
    }

    if let Some(instance) = schemadict.get("const") {
        // TODO: validate the instance against the schema, maybe returning Schema::Unsatisfiable
        return Ok(Schema::Const {
            value: instance.clone(),
        });
    }

    if let Some(instances) = schemadict.get("enum") {
        let instances = instances
            .as_array()
            .ok_or_else(|| anyhow!("enum must be an array"))?;
        // TODO: validate the instances against the schema, maybe returning Schema::Unsatisfiable
        return Ok(Schema::Enum {
            options: instances.clone(),
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
        let merged = merge(options.iter().chain(vec![&siblings]).collect())?;
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
            .iter()
            .map(|value| compile_resource(&ctx, ctx.as_resource_ref(value)))
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
            .iter()
            .map(|value| compile_resource(&ctx, ctx.as_resource_ref(value)))
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
            let placeholder = Schema::Ref { uri: uri.clone() };
            if ctx.contains_ref(uri.as_str()) {
                // We've already built this ref, so we can just return it
                return Ok(placeholder);
            }
            ctx.insert_ref(uri.as_str(), placeholder.clone());
            let resource = ctx.lookup_resource(&reference)?;
            let resolved_schema = compile_resource(ctx, resource)?;
            ctx.insert_ref(uri.as_str(), resolved_schema.clone());
            return Ok(placeholder);
        } else {
            if ctx.contains_ref(uri.as_str()) {
                bail!("Recursive $refs with sibling keys not implemented")
            }
            let resource = ctx.lookup_resource(&reference)?;
            let resolved_schema = compile_resource(ctx, resource)?;
            return Ok(merge_two(&siblings, &resolved_schema)?);
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

fn merge(schemas: Vec<&Schema>) -> Result<Schema> {
    if schemas.is_empty() {
        bail!("merge called with empty list")
    }
    if schemas.iter().all(|schema| matches!(schema, Schema::Any)) {
        return Ok(Schema::Any);
    }
    if let Some(unsat) = schemas
        .iter()
        .find(|schema| matches!(schema, Schema::Unsatisfiable { .. }))
    {
        // Return the first unsatisfiable schema for debug-ability
        return Ok((*unsat).to_owned());
    }
    // TODO: can we avoid cloning here?
    let mut merged = schemas[0].to_owned();
    for subschema in &schemas[1..] {
        merged = merge_two(&merged, subschema)?;
        if matches!(merged, Schema::Unsatisfiable { .. }) {
            // Early exit if the schema is already unsatisfiable
            break;
        }
    }
    Ok(merged)
}

fn merge_two(schema0: &Schema, schema1: &Schema) -> Result<Schema> {
    match (schema0, schema1) {
        (Schema::Any, _) => Ok(schema1.to_owned()),
        (_, Schema::Any) => Ok(schema0.to_owned()),
        (Schema::Unsatisfiable { .. }, _) => Ok(schema0.to_owned()),
        (_, Schema::Unsatisfiable { .. }) => Ok(schema1.to_owned()),
        (Schema::Ref { .. }, _) => Err(anyhow!("$ref with siblings not implemented")),
        (_, Schema::Ref { .. }) => Err(anyhow!("$ref with siblings not implemented")),
        (Schema::OneOf { options }, _) => Ok(Schema::OneOf {
            options: options
                .iter()
                .map(|opt| merge_two(opt, schema1))
                .collect::<Result<Vec<_>>>()?,
        }),
        (_, Schema::OneOf { options }) => Ok(Schema::OneOf {
            options: options
                .iter()
                .map(|opt| merge_two(schema0, opt))
                .collect::<Result<Vec<_>>>()?,
        }),
        (Schema::AnyOf { options }, _) => Ok(Schema::AnyOf {
            options: options
                .iter()
                .map(|opt| merge_two(opt, schema1))
                .collect::<Result<Vec<_>>>()?,
        }),
        (_, Schema::AnyOf { options }) => Ok(Schema::AnyOf {
            options: options
                .iter()
                .map(|opt| merge_two(schema0, opt))
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
            minimum: opt_max(*min1, *min2),
            maximum: opt_min(*max1, *max2),
            exclusive_minimum: opt_max(*emin1, *emin2),
            exclusive_maximum: opt_min(*emax1, *emax2),
            integer: *int1 || *int2,
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
            min_length: *min1.max(min2),
            max_length: opt_min(*max1, *max2),
            pattern: match (pattern1, pattern2) {
                (None, None) => None,
                (None, Some(r)) => Some(r.clone()),
                (Some(r), None) => Some(r.clone()),
                (Some(r1), Some(r2)) => {
                    if r1 == r2 {
                        Some(r1.clone())
                    } else {
                        bail!("intersection of patterns not implemented")
                    }
                }
            },
            format: match (format1, format2) {
                (None, None) => None,
                (None, Some(fmt)) => Some(fmt.clone()),
                (Some(fmt), None) => Some(fmt.clone()),
                (Some(fmt1), Some(fmt2)) => {
                    if fmt1 == fmt2 {
                        Some(fmt1.clone())
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
                prefix_items: prefix1,
                items: items1,
            },
            Schema::Array {
                min_items: min2,
                max_items: max2,
                prefix_items: prefix2,
                items: items2,
            },
        ) => Ok(Schema::Array {
            min_items: *min1.max(min2),
            max_items: opt_min(*max1, *max2),
            prefix_items: zip_default(
                prefix1,
                prefix2,
                items2.as_deref().unwrap_or(&Schema::Any),
                items1.as_deref().unwrap_or(&Schema::Any),
            )
            .iter()
            .map(|(item1, item2)| merge_two(item1, item2))
            .collect::<Result<Vec<Schema>>>()?,
            items: match (items1, items2) {
                (None, None) => None,
                (None, Some(item)) => Some(Box::new(*item.clone())),
                (Some(item), None) => Some(Box::new(*item.clone())),
                (Some(item1), Some(item2)) => Some(Box::new(merge_two(&item1, &item2)?)),
            },
        }),
        (
            Schema::Object {
                properties: props1,
                additional_properties: add1,
                required: req1,
            },
            Schema::Object {
                properties: props2,
                additional_properties: add2,
                required: req2,
            },
        ) => {
            let mut new_props = IndexMap::new();
            for key in props1.keys().chain(props2.keys()) {
                let new_schema = match (props1.get(key), props2.get(key), add1, add2) {
                    (Some(schema1), Some(schema2), _, _) => merge_two(schema1, schema2)?,
                    (Some(schema1), None, _, Some(add)) => merge_two(schema1, &add)?,
                    (None, Some(schema2), Some(add), _) => merge_two(&add, schema2)?,
                    (Some(schema1), None, _, None) => schema1.to_owned(),
                    (None, Some(schema2), None, _) => schema2.to_owned(),
                    (None, None, _, _) => bail!("should not happen"),
                };
                new_props.insert(key.clone(), new_schema);
            }
            Ok(Schema::Object {
                properties: new_props,
                additional_properties: match (add1, add2) {
                    (None, None) => None,
                    (None, Some(add)) => Some(Box::new(*add.clone())),
                    (Some(add), None) => Some(Box::new(*add.clone())),
                    (Some(add1), Some(add2)) => Some(Box::new(merge_two(&add1, &add2)?)),
                },
                required: req1.union(req2).cloned().collect(),
            })
        }
        //TODO: get types for error message
        _ => Ok(Schema::Unsatisfiable {
            reason: "incompatible types".to_string(),
        }),
    }
}

fn zip_default<'a, T>(
    arr1: &'a [T],
    arr2: &'a [T],
    default1: &'a T,
    default2: &'a T,
) -> Vec<(&'a T, &'a T)> {
    let iter1 = arr1.iter().chain(std::iter::repeat(default1));
    let iter2 = arr2.iter().chain(std::iter::repeat(default2));
    iter1.zip(iter2).take(arr1.len().max(arr2.len())).collect()
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
