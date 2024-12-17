# What makes llg go fast?

The main operation when computing a token mask is walking the tokenizer trie.
The trie is laid flat in memory, and just walking it is highly optimized,
with minimal branch mis-predictions.
When doing simple execution of regular expression automaton on the trie,
on AMD EPYC visiting one trie node takes about 13 cycles.
This is likely [close to optimal](https://github.com/guidance-ai/llguidance/blob/main/docs/toktrie.md).

For a tokenizer with `n_vocab` tokens, the trie typically has around `2 * n_vocab` nodes.
Thus, for 128k llama3 tokenizer, for EPYC running at 2 GHz,
we get around 1.5ms to compute the token mask for a simple regular expression.
In other words, the baseline is relatively fast.

## CFGs and lexer/parser split

To make this work for more complex grammars, we use the following:

- [derivre](https://github.com/microsoft/derivre), a derivative-based regular expression engine,
  which can construct automaton lazily, with very low startup cost
- a highly optimized
  [Earley parser](https://en.wikipedia.org/wiki/Earley_parser)
  for context-free grammars (CFGs)
  on top of the lexer defined with regular expressions

As for the lexer/parser split:
Back in the 1970s when computers were slow, people figured out that one can
first deal with words
(also called tokens (not be confused with LLM tokens) or lexemes)
and only then one deals with syntax.
This is because splitting text into words is cheaper than parsing it.
And so regular expressions were used for "lexing"
(splitting into words or lexemes) and context-free grammars were
used for the higher-level parsing.
Now, this is theoretically unnecessary, since regular languages are
subset of context-free languages.
It's just that doing lexing first and parsing on top of larger items just
happens to be quite a bit faster.
While computers are much faster now, the token masking is this specific problem where one has to do lots of parsing in a very short time.

Also, virtually all programming language definitions (including JSON)
have this lexer/parser separation.

Typically the LLM tokens are somewhat aligned with lexemes,
meaning that when walking the trie,
the parser needs to be involved in processing under 0.5% of trie nodes,
leaving the rest to the lexer.

As a consequence, walking the trie with a CFG is almost as fast as walking it with a regular expression.

## Earley parser optimizations

- CFG rules are stored in a flat array
- Earley items are indices into this array (dot position), and into Earley row array
- after an Earley row is computed, we determine which lexemes (terminals) are
  allowed in the current state; then we setup the lexer to only recognize these lexemes;
  thus the lexer only processes lexemes that are relevant in a given state
- when walking down the token trie, rows are added to the parser state (pushed)
  when a lexeme is scanned,
  and when coming back up, rows are popped;
  we do not actually pop the rows, but just move a pointer, and if we're
  about to scan a lexeme, we check if it is the same as previously pushed
  lexeme - in that case the row can be reused and doesn't have to re-computed;
  this happens very often

## Slicer optimization

Generally, computing almost empty token masks is cheap.
This is because if the lexer or parser don't allow a given byte
in the trie, the entire sub-tree can be skipped.
Thus, for example, a token mask resulting from a regular expression defining
integers is quite quick to compute (as it has only number tokens in it).

However, large masks are slower to compute.
They typically happen inside of a relatively unconstrained context in the grammar.
For example, inside of JSON string, or a comment in a programming language.

We thus define a series _slices_, under-approximation of such unconstrained contexts.
The slices are defined by regular expressions typically of the form `[...]{1,N}`
(that is a character class repeated up to `N` times).

For example, a good slice for JSON schemas is `[^"\\\x00-\x1F\x7F]{1,30}` -
it excludes `"`, `\`, and ASCII control characters, all of which have to
be escaped in JSON strings.
We put a length limit of `30`, since it covers a vast majority of the
tokenizer, but allows for matching in context when the length of the string
is limited to more than 30.

We go through each slice in the definition order,
and for each claim all tokens that match the regular expression of the slice,
and build a token trie for them.
The final slice is implicitly defined as the remainder of the tokens.
Thus, each token is only present in one slice (and thus one token trie and one corresponding mask).

When computing the mask,
we check if the slice is completely contained in any of the currently allowed lexemes.
That is, we check if the lexer will allow all tokens in the slice.
If that is the case, we add the corresponding mask, and skip walking the trie of the slice.
Otherwise, we walk the trie as usual.

For example, at some position in a JSON scheme,
it may allow `"foo"`, `"bar"` and `"C*"` where C is defined as
`([^\"\\\x00-\x1F\x7F]|\\([\"\\\/bfnrt]|u[a-fA-F0-9]{4}))`.
Here, our JSON slice `[^"\\\x00-\x1F\x7F]{1,30}` is not contained
in any of the allowed lexemes (because of the initial quote).
After scanning token corresponding to the opening quote `"`,
the lexer will allow `foo"`, `bar"` and `C*"`.
Now, the JSON slice is contained in `C*"`,
and thus we can skip walking the trie for the slice.

Similarly, if the lexer allows `C{0,50}"` (because there is a `"string"`
with `"maxLength": 50` in the schema), the JSON slice is contained in this lexeme.
OTOH, if the lexer allows `C{0,20}"`, than the JSON slice is not contained in this lexeme.

This optimization make the mask computation about 10x faster for JSON schemas.
