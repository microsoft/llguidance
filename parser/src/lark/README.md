# Lark-like syntax

This module converts from [Lark-like](https://github.com/lark-parser/lark) syntax to llguidance.
It makes it easier to get started with a new grammar,
and provides a familiar syntax, however is not a drop-in replacement for Lark.

Following are the extensions to Lark syntax:

- when several grammars are passed in one request (`grammars` field),
  they ones using Lark can reference others using syntax like `@17` refering
  to grammar at index 17 in the `grammars` list, or `@my_grammar` refering to grammar
  with `"name": "my_grammar"`.
- special tokens can referenced via `<token_name>` syntax, for example `<|ENDOFTEXT|>`;
  they cannot be used inside of terminals, but can be used in regular rules;
  the exact syntax depends on the tokenizer

Following are currently not supported:

- lookarounds in lexer regexes
- lazy modifier (`?`) in lexer regexes; in llguidance the whole lexeme is either lazy or greedy
- priorities
- templates
- imports (other than built-in `%import common`)
- regexes use Rust `regex` crate [syntax](https://docs.rs/regex/latest/regex/#syntax), not Python's `re` (though they are similar)

Following features of llguidance are currently not exposed in Lark syntax:

- `max_tokens` limits
- hiding of `stop=...`
- per-lexeme contextual and lazy flags
