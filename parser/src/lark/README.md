# Lark-like syntax

This module converts from [Lark-like](https://github.com/lark-parser/lark) syntax to llguidance.
It makes it easier to get started with a new grammar,
and provides a familiar syntax, however is not a drop-in replacement for Lark.

Following are the extensions to Lark syntax:

- when several grammars are passed in one request (`grammars` field),
  they ones using Lark can reference others using syntax like `@17` refering
  to grammar at index 17 in the `grammars` list, or `@my_grammar` refering to grammar
  with `"name": "my_grammar"`.
- you can specify temperature for subgrammar by referencing it via
  `my_temp_json[temperature=0.7]: @json` syntax
- special tokens can referenced via `<token_name>` syntax, for example `<|ENDOFTEXT|>`;
  they cannot be used inside of terminals, but can be used in regular rules;
  the exact syntax depends on the tokenizer
- `max_tokens`, `temperature` and `stop` can be specified on rules, but the rule body must be a token expression,
  for example: `mygen[stop="\n", max_tokens=10, temperature=0.7]: /.*/`
- if `stop` is specified (possibly as `""`) the rule is treated as `gen()` in Guidance;
  otherwise it is treated as `lexeme()`


Following are currently not supported:

- lookarounds in lexer regexes
- lazy modifier (`?`) in lexer regexes; in llguidance the whole lexeme is either lazy or greedy
- priorities
- templates
- imports (other than built-in `%import common`)
- regexes use Rust `regex` crate [syntax](https://docs.rs/regex/latest/regex/#syntax), not Python's `re` (though they are similar)
- certain string syntax, see [issue](https://github.com/microsoft/llguidance/issues/54)

Following features of llguidance are currently not exposed in Lark syntax:

- per-lexeme contextual and lazy flags

## Examples

### Llama JSON tool calling

Here, we restrict the output to either normal text response,
or a tool call to either Brave or Wolfram Alpha.

```lark
start: normal_text | brave | wolfram
normal_text: /(.|\n)*/
brave: <|python_tag|> "brave_search.call(query=" JSON_STRING ")" <|eom_id|>
wolfram: <|python_tag|> "wolfram_alpha.call(query=" JSON_STRING ")" <|eom_id|>
JSON_STRING_CHAR: /(\\([\"\\\/bfnrt]|u[a-fA-F0-9]{4})|[^\"\\\x00-\x1F\x7F])/
JSON_STRING: "\"" JSON_STRING_CHAR* "\""
```

Note that just as in lark uppercase identifiers define grammar lexemes
(also often called tokens) - they can't be recursive
(they are compiled to regular expressions).
This has performance implications, in particular you should **avoid short lexemes**.
If the grammar used `json_string` not `JSON_STRING`,
then each `json_string` would consists of lexeme `"`, followed
by any number of single-character lexemes, followed by lexeme `"`.
Such grammar would be very slow to run.
With upper-case `JSON_STRING`, the whole string is a lexeme.

BTW, in this case you may want to replace the JSON string
with Python string, depending on how the model was trained.

You can also use Lark-like syntax to combine JSON schemas with regular output.
In that case, you pass the JSON schemas as additional grammars, with
the lark grammar being the top-level one.

```lark
start: normal_text | fun_call
// @fun0, @fun1 refer to other sub-grammars, see below
fun_call: <|python_tag|> ( @fun0 | @fun1 ) <|eom_id|>
normal_text: /(.|\n)*/
```

```json
{
  "grammars": [
    {
      "lark_grammar": "...the lark above...",
    },
    {"name": "fun0", "json_schema": { ... }},
    {"name": "fun1", "json_schema": { ... }}
  ]
}
```
