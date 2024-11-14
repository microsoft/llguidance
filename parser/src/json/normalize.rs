use anyhow::{anyhow, bail, Result};
use indexmap::IndexMap;
use serde_json::{Map, Value};

const TYPES: [&str; 6] = ["null", "boolean", "number", "string", "array", "object"];

#[derive(Debug, PartialEq, Clone)]
enum Schema {
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
        required: Vec<String>,
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
    #[allow(dead_code)]
    Ref(Box<Schema>),
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
            Schema::Ref(..) => Ok(self),
        }
    }
}

impl TryFrom<Value> for Schema {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        schema_from_value_inner(value)?.normalize()
    }
}

fn schema_from_value_inner(value: Value) -> Result<Schema> {
    if let Some(b) = value.as_bool() {
        return Ok({
            if b {
                Schema::Any
            } else {
                Schema::Unsatisfiable {
                    reason: "schema is false".to_string(),
                }
            }
        });
    }

    // Get the schema as an object
    // TODO: validate against metaschema & check for unimplemented keys
    let schemadict = value
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
        let siblings = Schema::try_from(Value::Object(schemadict))?;
        // Short-circuit if schema is already unsatisfiable
        if matches!(siblings, Schema::Unsatisfiable { .. }) {
            return Ok(siblings);
        }
        let options = all_of
            .iter()
            .map(|value| Schema::try_from(value.to_owned()))
            .collect::<Result<Vec<_>>>()?;
        let merged = merge(options.iter().chain(vec![&siblings]).collect())?;
        return Ok(merged);
    }

    if let Some(any_of) = schemadict.remove("anyOf") {
        let any_of = any_of
            .as_array()
            .ok_or_else(|| anyhow!("anyOf must be an array"))?;
        let siblings = Schema::try_from(Value::Object(schemadict))?;
        // Short-circuit if schema is already unsatisfiable
        if matches!(siblings, Schema::Unsatisfiable { .. }) {
            return Ok(siblings);
        }
        let options = any_of
            .iter()
            .map(|value| {
                Schema::try_from(value.to_owned()).and_then(|schema| merge_two(&schema, &siblings))
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(Schema::AnyOf { options });
    }

    // TODO: refactor to share code with anyOf
    if let Some(one_of) = schemadict.remove("oneOf") {
        let one_of = one_of
            .as_array()
            .ok_or_else(|| anyhow!("oneOf must be an array"))?;
        let siblings = Schema::try_from(Value::Object(schemadict))?;
        // Short-circuit if schema is already unsatisfiable
        if matches!(siblings, Schema::Unsatisfiable { .. }) {
            return Ok(siblings);
        }
        let options = one_of
            .iter()
            .map(|value| {
                Schema::try_from(value.to_owned()).and_then(|schema| merge_two(&schema, &siblings))
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(Schema::OneOf { options });
    }

    if let Some(_) = schemadict.remove("$ref") {
        bail!("Ref not implemented")
    }

    let types = match schemadict.remove("type") {
        Some(Value::String(tp)) => {
            return try_type(&schemadict, &tp);
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

    let options = types
        .iter()
        .map(|tp| try_type(&schemadict, &tp))
        .collect::<Result<Vec<Schema>>>()?;
    Ok(Schema::AnyOf { options })
}

fn try_type(schema: &Map<String, Value>, tp: &str) -> Result<Schema> {
    match tp {
        "null" => Ok(Schema::Null),
        "boolean" => Ok(Schema::Boolean),
        "number" | "integer" => {
            let minimum = schema.get("minimum").and_then(|v| v.as_f64());
            let maximum = schema.get("maximum").and_then(|v| v.as_f64());
            let exclusive_minimum = schema.get("exclusiveMinimum").and_then(|v| v.as_f64());
            let exclusive_maximum = schema.get("exclusiveMaximum").and_then(|v| v.as_f64());
            Ok(Schema::Number {
                minimum,
                maximum,
                exclusive_minimum,
                exclusive_maximum,
                integer: tp == "integer",
            })
        }
        "string" => {
            let min_length = schema
                .get("minLength")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let max_length = schema.get("maxLength").and_then(|v| v.as_u64());
            let pattern = schema
                .get("pattern")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let format = schema
                .get("format")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            Ok(Schema::String {
                min_length,
                max_length,
                pattern,
                format,
            })
        }
        "array" => {
            let min_items = schema.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0);
            let max_items = schema.get("maxItems").and_then(|v| v.as_u64());
            let prefix_items = schema
                .get("prefixItems")
                .map(|v| {
                    v.as_array()
                        .ok_or_else(|| anyhow!("prefixItems must be an array"))
                })
                .transpose()?
                .map(|items| {
                    items
                        .iter()
                        .map(|item| Schema::try_from(item.to_owned()))
                        .collect::<Result<Vec<Schema>>>()
                })
                .transpose()?
                .unwrap_or_default();
            let items = schema
                .get("items")
                .map(|v| Schema::try_from(v.to_owned()))
                .transpose()?;
            Ok(Schema::Array {
                min_items,
                max_items,
                prefix_items,
                items: items.map(Box::new),
            })
        }
        "object" => {
            let properties = schema
                .get("properties")
                .map(|v| {
                    v.as_object()
                        .ok_or_else(|| anyhow!("properties must be an object"))
                })
                .transpose()?
                .map(|props| {
                    props
                        .iter()
                        .map(|(k, v)| {
                            Schema::try_from(v.to_owned()).map(|schema| (k.clone(), schema))
                        })
                        .collect::<Result<IndexMap<String, Schema>>>()
                })
                .transpose()?
                .unwrap_or_default();
            let additional_properties = schema
                .get("additionalProperties")
                .map(|v| Schema::try_from(v.to_owned()))
                .transpose()?;
            let required = schema
                .get("required")
                .map(|v| {
                    v.as_array()
                        .ok_or_else(|| anyhow!("required must be an array"))
                })
                .transpose()?
                .map(|items| {
                    items
                        .iter()
                        .map(|item| {
                            item.as_str()
                                .ok_or_else(|| anyhow!("required items must be strings"))
                                .map(|s| s.to_string())
                        })
                        .collect::<Result<Vec<String>>>()
                })
                .transpose()?
                .unwrap_or_default();
            Ok(Schema::Object {
                properties,
                additional_properties: additional_properties.map(Box::new),
                required,
            })
        }
        _ => bail!("Invalid type: {}", tp),
    }
}

fn merge(schemas: Vec<&Schema>) -> Result<Schema> {
    if schemas.is_empty() {
        bail!("merge called with empty list")
    }
    if schemas.iter().all(|schema| matches!(schema, Schema::Any)) {
        return Ok(Schema::Any);
    }
    if schemas
        .iter()
        .any(|schema| matches!(schema, Schema::Unsatisfiable { .. }))
    {
        // Return the first unsatisfiable schema for debug-ability
        // TODO: think through ownership and don't return a clone if possible
        return Ok(schemas[0].to_owned());
    }
    // TODO: can we avoid cloning here?
    let mut merged = schemas[0].to_owned();
    for subschema in &schemas[1..] {
        merged = merge_two(&merged, subschema)?;
    }
    Ok(merged.to_owned())
}

fn merge_two(schema0: &Schema, schema1: &Schema) -> Result<Schema> {
    match (schema0, schema1) {
        (Schema::Any, _) => Ok(schema1.to_owned()),
        (_, Schema::Any) => Ok(schema0.to_owned()),
        (Schema::Unsatisfiable { .. }, _) => Ok(schema0.to_owned()),
        (_, Schema::Unsatisfiable { .. }) => Ok(schema1.to_owned()),
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
                required: req1.iter().chain(req2.iter()).cloned().collect(),
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
    fn test_normalize() {
        let schema = json!({
            "minLength": 1,
            "maxLength": 10,
            "pattern": "^[a-z]+$",
            "format": "email",
            "minimum": 10,
        });
        let schema = Schema::try_from(schema).unwrap();
        println!("{:?}", schema);
    }

    #[test]
    fn test_normalize_2() {
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
            "allOf" : [
                {"type": "integer"}
            ],
            "pattern": "^[a-z]+$",
            "maximum": 11
        });
        let schema = Schema::try_from(schema).unwrap();
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
        let schema = Schema::try_from(schema).unwrap();
        println!("{:?}", schema);
    }
}
