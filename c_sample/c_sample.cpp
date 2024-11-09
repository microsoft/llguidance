#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <cstring>

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
  LlgTokenizerInit tok_init = {};
  tok_init.vocab_size = (uint32_t)tokens.size();
  tok_init.tok_eos = tok_eos;
  tok_init.token_lens = token_lens;
  tok_init.token_bytes = token_bytes;
  tok_init.tokenize_assumes_string = false;
  tok_init.tokenize_user_data = tokenize_user_data;
  tok_init.tokenize_fn = tokenize_fn;

  char error_buf[128];
  auto tok = llg_new_tokenizer(&tok_init, error_buf, sizeof(error_buf));

  if (tok == nullptr) {
    printf("Error: %s\n", error_buf);
    exit(1);
  }

  return tok;
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

LlgTokenizer *create_hf_tokenizer(std::string tokenizer_json,
                                  uint32_t tok_eos) {
  LlgTokenizerInit tok_init = {};

  tok_init.tok_eos = tok_eos;
  tok_init.use_approximate_greedy_tokenize_fn = true;
  tok_init.tokenizer_json = tokenizer_json.c_str();

  char error_buf[128];
  auto tok = llg_new_tokenizer(&tok_init, error_buf, sizeof(error_buf));

  if (tok == nullptr) {
    printf("Error: %s\n", error_buf);
    exit(1);
  }

  return tok;
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

std::vector<uint32_t> do_llg_tokenize(const LlgTokenizer *tok, std::string s) {
  std::vector<uint32_t> tokens;
  size_t n_tokens =
      llg_tokenize_bytes(tok, (const uint8_t *)s.c_str(), s.size(), nullptr, 0);
  tokens.resize(n_tokens);
  llg_tokenize_bytes(tok, (const uint8_t *)s.c_str(), s.size(), tokens.data(),
                     n_tokens);
  return tokens;
}

std::string do_llg_stringify_tokens(const LlgTokenizer *tok,
                                    std::vector<uint32_t> tokens) {
  char buffer[1024];
  size_t n_bytes = llg_stringify_tokens(tok, tokens.data(), tokens.size(),
                                        buffer, sizeof(buffer));
  if (n_bytes >= sizeof(buffer)) {
    char *new_buffer = new char[n_bytes + 1];
    llg_stringify_tokens(tok, tokens.data(), tokens.size(), new_buffer,
                         n_bytes + 1);
    auto r = std::string(new_buffer);
    delete[] new_buffer;
    return r;
  } else {
    return std::string(buffer);
  }
}

int main(int argc, const char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <schema.ll.json> <sample.json> [tokenizer.json]\n",
           argv[0]);
    return 1;
  }

  // the tokenizer can (and should) be shared between constraints
  LlgTokenizer *tokenizer = argc > 3
                                ? create_hf_tokenizer(read_file(argv[3]), 2)
                                : create_byte_tokenizer();

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

  // for debugging the tokenizer:
  // for (int i = 0; i < 320; ++i) {
  //   std::vector<uint32_t> tokens;
  //   tokens.push_back(i);
  //   std::string s = do_llg_stringify_tokens(tokenizer, tokens);
  //   printf("Token %d: %s\n", i, s.c_str());
  // }

  // we assume our "LLM" will generate these tokens
  auto tokens = do_llg_tokenize(tokenizer, sample_json);

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
