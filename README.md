# Low-level Guidance (llguidance)

This controller implements a context-free grammar parser with Earley's algorithm
on top of a lexer which uses [derivatives of regular expressions](../derivre/README.md).

It's to be used by next-generation [Guidance](https://github.com/guidance-ai/guidance) grammars.
See how it works in [plan.md](./plan.md).

Guidance branch: https://github.com/hudson-ai/guidance/tree/lazy_grammars

## Guidance implementation notes

- `gen()` now generates a new node, `Gen`
- grammar is serialized to JSON, see `ll_serialize()`

## TODO

- [ ] `substring()` in Guidance needs to be re-implemented (translate to RegexAst)
- [ ] `to_regex_vec()` in lexerspec.rs - non-contextual keywords
- [ ] allow byte sequence to fast-forward through grammar at start (grammar derivative)
- [ ] check if env allows for backtracking (if not, don't use it)
- [ ] return `{when_sampled:[EOS],ff:[]}` as slice when EOS ends gen()
- [ ] check for relevance of intersection and negation in `derivre`

## Lexeme-splitting

See https://github.com/hudson-ai/guidance/issues/5

```python
    g = select(["a", "abq", "c"]) + optional("bQ")
    check_grammar(g, ["", "a‧b‧q‧≺EOS≻"]) # fails 'q' is forced
    check_grammar(g, ["", "a‧b‧Q"]) # doesn't match at all
```

## Only valid tokens

See https://github.com/hudson-ai/guidance/issues/13

- [ ] implement `.forced_byte()` method in `derivre`
- [ ] use this for cheap `.forced_byte()` impl in `llguidance`
- [ ] while walking token trie, remember all forced paths (there shouldn't be too many of them)

In toktrie walk, if we encounter a forced byte, we go into forced mode
where we just chase all forced bytes.
The first token we find on this path we put on some list.
We do not add any of these tokens to the allow set.

Then, after token trie walk, for every token on this list we re-create
the forced byte string, tokenize, chop excessive tokens, and add the first
token from tokenization to allow set and remaining tokens (if any) as conditional
splice.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
