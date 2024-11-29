# JSON schema -> llguidance converter

This sub-module converts JSON schema to llguidance grammar.

## Supported JSON schema features

Following JSON schema features are supported.

Core features:

- `anyOf`
- `oneOf` - not supported right now, use `anyOf` instead, [issue](https://github.com/microsoft/llguidance/issues/77)
- `allOf` - intersection of certain schemas is not supported right now
- `$ref` - within the document only
- `const`
- `enum`
- `type` - both single type and array of types

Array features:

- `items`
- `prefixItems`
- `minItems`
- `maxItems`

Object features:

- `properties` - order of properties is fixed to the order in schema
- `additionalProperties`
- `required`

String features:

- `minLength`
- `maxLength`
- `pattern` (though we always anchor them, [issue](https://github.com/microsoft/llguidance/issues/66))
- `format`, with the following formats: `date-time`, `time`, `date`, `duration`, `email`, `hostname`, `ipv4`, `ipv6`, `uuid`,

Number features (for both integer and number):

- `minimum`
- `maximum`
- `exclusiveMinimum`
- `exclusiveMaximum`
