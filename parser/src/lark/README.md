# Lark-like syntax

This module converts from [Lark-like](https://github.com/lark-parser/lark) syntax to llguidance.
It makes it easier to get started with a new grammar,
and provides a familiar syntax, however is not a drop-in replacement for Lark.

Following are currently not supported:

- lookarounds in lexer regexes
- lazy modifier (`?`) in lexer regexes; in llguidance the whole lexeme is either lazy or greedy
- priorities
- templates
- imports (other than built-in `%import common`)
- regexes use Rust `regex` crate [syntax](https://docs.rs/regex/latest/regex/#syntax), not Python's `re` (though they are similar)

Following features of llguidance are currently not exposed in Lark syntax:

- composite/nested grammars
- `max_tokens` limits
- hiding of `stop=...`
- per-lexeme contextual and lazy flags
