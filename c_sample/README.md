# llguidance C++ sample

This is a simple example of how to use the llguidance library in C++.

It reads a Guidance grammar from a JSON file as well as the text that we
pretend the LLM has generated and then makes sure the text conforms to the
grammar.

## Building

- [install rust](https://www.rust-lang.org/tools/install); 1.75 or later
- clone the repository
- `cd c_sample`
- `make`

## Making it real

For a real integration:

- replace `bogus_tokenize()` with a real tokenizer for your LLM
- make sure you pass the list of tokens to `create_tokenizer()`
- for an incoming request, create a constraint based on data in the
  request; make sure to handle errors returned by `llg_get_error()`
- while computing logits, run `llg_compute_mask()`
- sample with the returned mask
- pass the sampled token to `llg_commit_token()`

## TODO

- [ ] extend to read JSON schema
- [ ] extend to allow simple regex as constraint
