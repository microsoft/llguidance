use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchemaStats {
    pub features: HashMap<String, usize>,
    pub full_size: usize,
    pub stripped_size: usize,
    pub strip_error: Option<String>,
    pub additional_features: bool,
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
        "properties" | "patternProperties" => true,
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

fn is_valid_schema_type(tp: &str) -> bool {
    match tp {
        "object" | "number" | "string" | "integer" | "boolean" | "null" | "array" => true,
        _ => false,
    }
}

fn looks_like_schema(schema: &Value) -> bool {
    if !schema.is_object() {
        return false;
    }

    if let Some(k) = schema["type"].as_str() {
        return is_valid_schema_type(k);
    }

    if schema["type"].is_array() {
        return true;
    }

    for p in [
        "additionalProperties",
        "properties",
        "$ref",
        "oneOf",
        "anyOf",
        "allOf",
        "if",
        "not",
        "pattern",
        "patternProperties",
        "enum",
        "minProperties",
        "maxProperties",
    ] {
        if !schema[p].is_null() {
            return true;
        }
    }

    false
}

impl SchemaStats {
    fn incr(&mut self, feature: &str) {
        self.features
            .entry(feature.to_string())
            .and_modify(|e| *e += 1)
            .or_insert(1);
    }

    fn has_type(&self, obj: &Value, t: &str) -> bool {
        if let Some(types) = obj["type"].as_array() {
            types.iter().any(|v| v == t)
        } else if let Some(t) = obj["type"].as_str() {
            t == t
        } else {
            false
        }
    }

    fn incr_feature(&mut self, kw: &str, val: &Value, obj: &Value) {
        match kw {
            "additionalItems" | "additionalProperties" => {
                if val.is_object() {
                    self.incr(&format!("{}:object", kw));
                }
            }
            "type" => {
                if let Some(types) = val.as_array() {
                    self.incr("type:[]");
                    for t in types {
                        if let Some(t) = t.as_str() {
                            self.incr(&format!("type:{}", t));
                        }
                    }
                } else if let Some(type_id) = val.as_str() {
                    self.incr(&format!("type:{}", type_id));
                }
            }
            "multipleOf" => {
                self.incr(&format!("multipleOf:{}", val));
            }
            "format" => {
                if let Some(format) = val.as_str() {
                    self.incr(&format!("format:{}", format));
                }
            }
            "allOf" | "anyOf" | "oneOf" => {
                if let Some(arr) = val.as_array() {
                    if arr.len() <= 1 {
                        // any of these with a single item is not interesting
                        self.incr(&format!("{}:trivial", kw));
                        return;
                    }
                }
            }
            "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" => {
                if self.has_type(obj, "integer") {
                    self.incr("@minmaxInteger");
                } else {
                    self.incr("@minmaxNumber");
                }
            }
            "minLength" | "maxLength" => {
                self.incr("@minmaxLength");
            }
            "minItems" | "maxItems" => {
                self.incr("@minmaxItems");
            }
            "minProperties" | "maxProperties" => {
                self.incr("@minmaxProperties");
            }
            "enum" if self.additional_features => {
                if let Some(arr) = val.as_array() {
                    let len = arr.len();
                    self.incr(&format!("enum:{}", len.next_power_of_two()));

                    if arr.iter().all(|v| v.is_string() || v.is_null()) {
                        self.incr("enum:string_or_null");
                    } else if arr.iter().all(|v| v.is_number() || v.is_null()) {
                        self.incr("enum:number_or_null");
                    } else if arr.iter().all(|v| v.is_boolean() || v.is_null()) {
                        self.incr("enum:boolean_or_null");
                    } else if arr
                        .iter()
                        .all(|v| v.is_boolean() || v.is_string() || v.is_number() || v.is_null())
                    {
                        self.incr("enum:primitive");
                    } else {
                        self.incr("enum:mixed");
                    }
                }
            }
            _ => {}
        }

        self.incr(kw);
    }

    fn map_one_subschema(&mut self, v: &Value) -> Option<Value> {
        if looks_like_schema(v) {
            match self.map_schema(v) {
                Ok(new_v) => Some(new_v),
                Err(_) => {
                    // just ignore it
                    None
                }
            }
        } else {
            self.map_subschemas(v)
        }
    }

    fn map_subschemas(&mut self, schema: &Value) -> Option<Value> {
        if let Some(obj) = schema.as_object() {
            let mut new_obj = serde_json::Map::new();
            for (k, v) in obj.iter() {
                if let Some(new_v) = self.map_one_subschema(v) {
                    new_obj.insert(k.clone(), new_v);
                }
            }
            if new_obj.is_empty() {
                None
            } else {
                Some(Value::Object(new_obj))
            }
        } else {
            None
        }
    }

    fn map_schema(&mut self, schema: &Value) -> Result<Value> {
        match schema {
            Value::Bool(_) => {
                self.incr("_boolSchema");
                Ok(schema.clone())
            }
            Value::Object(obj) => {
                let mut new_obj = serde_json::Map::new();
                let mut has_ref = 0;
                let mut has_x_of = 0;
                let mut has_normal_kw = 0;
                // if obj.contains_key("items") && obj.contains_key("additionalItems") {
                //     if schema["additionalItems"].as_bool() == Some(false) {
                //         self.incr("@itemsAndAdditionalItemsFalse");
                //     } else if schema["additionalItems"].as_bool() == Some(true) {
                //         self.incr("@itemsAndAdditionalItemsTrue");
                //     } else {
                //         self.incr("@itemsAndAdditionalItemsObj");
                //     }
                // }
                for (k, v) in obj.iter() {
                    if !is_schema_kw(k) {
                        match k.as_str() {
                            "definitions" | "$defs" => self.incr(k),
                            _ => {}
                        }

                        if let Some(new_v) = self.map_one_subschema(v) {
                            new_obj.insert(k.clone(), new_v);
                        }

                        continue;
                    }
                    match k.as_str() {
                        "$ref" => {
                            has_ref = 1;
                        }
                        "allOf" | "anyOf" | "oneOf" => {
                            has_x_of = 1;
                        }
                        _ => {
                            has_normal_kw = 1;
                        }
                    }
                    self.incr_feature(k, v, schema);
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
                    } else if k == "required" {
                        let arr = v
                            .as_array()
                            .ok_or_else(|| anyhow!("Expected array for required"))?;
                        if arr.is_empty() {
                            self.incr("_requiredEmpty");
                        }
                        v.clone()
                    } else {
                        v.clone()
                    };
                    new_obj.insert(k.clone(), new_v);
                }
                if has_ref + has_normal_kw + has_x_of > 1 {
                    self.incr("@siblingKeys");
                }
                Ok(Value::Object(new_obj))
            }
            _ => bail!(
                "Expected object or bool schema; got {}",
                serde_json::to_string(schema).unwrap()
            ),
        }
    }

    pub fn for_file(file_name: &str, schema: &Value, additional_features: bool) -> SchemaStats {
        let mut stats = SchemaStats::default();
        stats.additional_features = additional_features;
        stats.full_size = serde_json::to_string(schema).unwrap().len();

        match stats.map_schema(schema) {
            Ok(val) => {
                stats.stripped_size = serde_json::to_string(&val).unwrap().len();
            }
            Err(e) => {
                eprintln!("{} Error Stats: {}", file_name, e);
                stats.strip_error = Some(format!("{e}"));
            }
        }

        stats
    }
}
