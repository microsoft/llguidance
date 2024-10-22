#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>

#include "llguidance.h"

// Create an LlgTokenizer; tokens[token_id] is a byte sequence corresponding to
// given token_id; see below for tokenize_fn
LlgTokenizer *create_tokenizer(std::vector<std::vector<uint8_t>> &tokens,
                               uint32_t tok_eos, LlgTokenizeFn tokenize_fn,
                               const void *tokenize_user_data) {
  auto token_lens = new uint32_t[tokens.size()];
  size_t total_size = 0;
  for (size_t i = 0; i < tokens.size(); i++) {
    token_lens[i] = tokens[i].size();
    total_size += token_lens[i];
  }
  auto token_bytes = new uint8_t[total_size];
  size_t offset = 0;
  for (size_t i = 0; i < tokens.size(); i++) {
    memcpy(token_bytes + offset, tokens[i].data(), token_lens[i]);
    offset += token_lens[i];
  }
  LlgTokenizerInit tok_init = {
      .vocab_size = (uint32_t)tokens.size(),
      .tok_eos = tok_eos,
      .token_lens = token_lens,
      .token_bytes = token_bytes,
      .tokenize_assumes_string = false,
      .tokenize_user_data = tokenize_user_data,
      .tokenize_fn = tokenize_fn,
  };
  return llg_new_tokenizer(&tok_init);
}

// This function assumes that each byte is a single token.
// You want to replace this. This has to be thread-safe!
std::vector<uint32_t> bogus_tokenize(const uint8_t *bytes_ptr, size_t nbytes) {
  std::vector<uint32_t> token_ids;
  for (size_t i = 0; i < nbytes; i++) {
    token_ids.push_back(bytes_ptr[i]);
  }
  return token_ids;
}

// This wraps a C++-style "bogus_tokenize()" in a way llg wants it.
size_t tokenize_callback(const void *user_data, const uint8_t *bytes,
                         size_t bytes_len, uint32_t *output_tokens,
                         size_t output_tokens_len) {
  (void)user_data;
  auto tokens = bogus_tokenize(bytes, bytes_len);
  if (output_tokens_len > 0) {
    memcpy(output_tokens, tokens.data(),
           std::min(output_tokens_len, tokens.size()) * sizeof(uint32_t));
  }
  return tokens.size();
}

// This creates a tokenizer that treats each byte as a token.
LlgTokenizer *create_byte_tokenizer(void) {
  std::vector<std::vector<uint8_t>> tokens;
  // every byte is a token
  for (size_t i = 0; i < 256; i++) {
    tokens.push_back({(uint8_t)i});
  }
  const char *eos = "<EOS>";
  tokens.push_back(std::vector<uint8_t>(eos, eos + strlen(eos)));
  return create_tokenizer(tokens, tokens.size() - 1, tokenize_callback,
                          nullptr);
}

std::string read_file(const std::string &filePath) {
  std::ifstream file(filePath);
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

void fail_constraint(LlgConstraint *c) {
  printf("Error: %s\n", llg_get_error(c));
  llg_free_constraint(c);
  exit(1);
}

int main(int argc, const char *argv[]) {
  // the tokenizer can (and should) be shared between constraints
  LlgTokenizer *tokenizer = create_byte_tokenizer();

  if (argc != 3) {
    printf("Usage: %s <schema.ll.json> <sample.json>\n", argv[0]);
    return 1;
  }

  auto schema_json = read_file(argv[1]);
  auto sample_json = read_file(argv[2]);

  LlgConstraintInit init;
  llg_constraint_init_set_defaults(&init, tokenizer);
  init.log_stderr_level = 0; // default to 1 (warnings only)

  LlgConstraint *c = llg_new_constraint(&init, schema_json.c_str());
  // this is a very common place where errors can happen - for example the
  // schema was invalid
  if (llg_get_error(c)) {
    fail_constraint(c);
  }

  // we assume our "LLM" will generate these tokens
  auto tokens =
      bogus_tokenize((const uint8_t *)sample_json.c_str(), sample_json.size());

  LlgMaskResult mask_res;
  for (size_t i = 0; i < tokens.size(); i++) {
    // compute mask - this can be done with parallel with logit generation
    if (llg_compute_mask(c, &mask_res) != 0) {
      fail_constraint(c);
    }

    // here, we would normally sample constrained to mask_res.sample_mask
    // using mask_res.temperature
    uint32_t token = tokens[i];

    // make sure token is in the mask
    assert(mask_res.sample_mask[token / 32] & (1 << (token % 32)));

    // here we commit the token
    // if "ff_tokens" are enabled, this can return more than one token
    // to fast-forward
    LlgCommitResult commit_res;
    if (llg_commit_token(c, tokens[i], &commit_res) != 0) {
      fail_constraint(c);
    }

    // we didn't enable ff_tokens, so the exact token that we passed should be
    // returned
    assert(commit_res.n_tokens == 1);
    assert(commit_res.tokens[0] == token);
  }

  if (llg_compute_mask(c, &mask_res) != 0) {
    fail_constraint(c);
  }
  // we assume the constraint will force EOS at the end of the input
  assert(mask_res.is_stop);

  printf("OK!\n");
  return 0;
}
