# Support for special tokens

Tokenizers typically include special tokens, such as 
`<|end_of_text|>`, `<|eot_id|>`, `<|python_tag|>`, `<|start_header_id|>`, etc.
This library is tasked with translating between the byte sequences
and tokens.
If you see bytes `<|eot_id|>` in the input, you may or may not want to treat them
as a special token.

The library assumes that by default you want ot treat them as bytes
(so they would be tokenized as `<|`, `eot`, `_`, `id`, `|>` or similar).
To indicate that you want to treat them as a special token, you need to
prefix them with "marker" byte 0xFF (255) (`TokTrie::SPECIAL_TOKEN_MARKER`).

Byte FF is chosen as a marker because it is not a valid UTF-8 byte, so it should not normally
occur in regular inputs.
In Rust, you cannot have byte FF in `&str`, only in `&[u8]`.
In Python note the difference between `b"\xFF"` and `"\xFF".encode("utf-8")`
(or equivalently `"\u00FF".encode("utf-8")`), which is `b"\xC3\xBF"`.

If you're constructing the token array for `TokTrie` constructor manually, 
it should include the special tokens prefixed with the marker byte FF.

The llguidance library does not expose the FF bytes externally
(except for special `tokenize_bytes_marker` methods), so you
generally don't need to worry about them, except when building the `TokTrie`.
