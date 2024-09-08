use std::{
    ffi::{c_char, CStr},
    sync::Arc,
};

use anyhow::Result;
use toktrie::{InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenizerEnv};

use crate::{
    api::{ParserLimits, TopLevelGrammar},
    Constraint, Logger, TokenParser,
};

struct CTokenizerInner {
    trie: TokTrie,
    tokenize_fn: TokenizeFn,
    tokenize_assumes_string: bool,
}

impl CTokenizerInner {
    fn raw_tokenize(&self, s: &[u8]) -> Vec<toktrie::TokenId> {
        let mut res_toks = vec![0; s.len() / 4 + 5];
        let n_toks = (self.tokenize_fn)(s.as_ptr(), s.len(), res_toks.as_mut_ptr(), res_toks.len());

        if n_toks > res_toks.len() {
            res_toks.resize(n_toks, 0);
            (self.tokenize_fn)(s.as_ptr(), s.len(), res_toks.as_mut_ptr(), res_toks.len());
        }

        res_toks.truncate(n_toks);
        res_toks
    }
}

impl TokenizerEnv for CTokenizerInner {
    fn stop(&self) -> ! {
        panic!("stop() called on CTokenizerInner")
    }

    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<toktrie::TokenId> {
        if self.tokenize_assumes_string {
            self.trie
                .tokenize_with_greedy_fallback(s, |s| self.raw_tokenize(s.as_bytes()))
        } else {
            self.raw_tokenize(s)
        }
    }
}

pub struct CTokenizer {
    token_env: TokEnv,
}

impl CTokenizer {
    fn from_init(init: &TokenizerInit) -> Self {
        let token_lens =
            unsafe { std::slice::from_raw_parts(init.token_lens, init.vocab_size as usize) };
        let total_len = token_lens.iter().sum::<u32>();
        let token_bytes =
            unsafe { std::slice::from_raw_parts(init.token_bytes, total_len as usize) };

        let mut tokens = vec![];
        let mut ptr = 0;
        for len in token_lens {
            let token = &token_bytes[ptr..ptr + *len as usize];
            tokens.push(token.to_vec());
            ptr += *len as usize;
        }
        let trie = TokTrie::from(&TokRxInfo::new(init.vocab_size, init.tok_eos), &tokens);

        CTokenizer {
            token_env: Arc::new(CTokenizerInner {
                trie,
                tokenize_assumes_string: init.tokenize_assumes_string,
                tokenize_fn: init.tokenize_fn,
            }),
        }
    }

    fn to_env(&self) -> TokEnv {
        self.token_env.clone()
    }
}

pub type TokenId = u32;

/// Tokenization function
/// Will not write more than output_tokens_len tokens (which can be 0)
/// Returns the total number of tokens (which can be more than output_tokens_len)
pub type TokenizeFn = extern "C" fn(
    bytes: *const u8,
    bytes_len: usize,
    output_tokens: *mut u32,
    output_tokens_len: usize,
) -> usize;

#[repr(C)]
pub struct TokenizerInit {
    /// The number of tokens in the vocabulary
    pub vocab_size: u32,

    /// The token ID for the end of sentence token
    /// For chat mode, set it to end-of-turn token
    pub tok_eos: TokenId,

    /// An array of the lengths of the token strings (vocab_size elements)
    pub token_lens: *const u32,

    /// A pointer to the token strings
    /// The length of this the sum of all token_lens
    pub token_bytes: *const u8,

    /// Set to true to enable hack that works around the tokenize_fn only
    /// accepting valid UTF-8 strings and possibly adding <BOS> etc.
    /// TODO: the <BOS> bit not implemented yet
    pub tokenize_assumes_string: bool,

    /// Tokenization function, see TokenizeFn docs.
    /// It should only tokenize the bytes and not add
    /// any <BOS> etc. It should also work on any byte sequence, including
    /// invalid UTF-8. If this is not the case, set tokenize_assumes_string to true.
    /// Either way, this function has to be thread-safe!
    pub tokenize_fn: TokenizeFn,
}

#[repr(C)]
pub struct ConstraintInit {
    /// The tokenizer to use, created with llg_new_tokenizer()
    pub tokenizer: *const CTokenizer,
    /// The log level for the buffer that is kept inside of the constraint
    /// 0 - no logging, 1 - warnings only, 2 - info
    pub log_buffer_level: u32,
    /// The log level for writing to stderr
    pub log_stderr_level: u32,
    /// Does the engine support fast-forward tokens?
    /// (Appending more than one token to output at once)
    pub ff_tokens_ok: bool,
    /// Does the engine support backtracking?
    /// (Removing tokens from the output)
    pub backtrack_ok: bool,
    /// The resource limits for the parser
    /// Default values will be used for all fields that are 0
    pub limits: ParserLimits,
}

pub struct CConstraint {
    local_error: Option<String>,
    constraint: Option<Constraint>,
}

#[repr(C)]
pub struct CMaskResult {
    /// One bit per vocab token
    /// This is valid until any call to llg_*() on the current constraint
    pub sample_mask: *const u32,
    /// Temperature to use for sampling
    pub temperature: f32,
    /// Should the sequence stop?
    pub is_stop: bool,
}

/// Represents result from llg_commit_token()
#[repr(C)]
pub struct CCommitResult {
    /// The tokens to append to the output if any
    /// This is valid until any call to llg_*() on the current constraint
    pub tokens: *const u32,
    /// The number of tokens in the tokens array (can be 0)
    pub n_tokens: u32,
    /// Should the sequence stop?
    pub is_stop: bool,
}

fn new_constraint(init: &ConstraintInit, grammar_json: *const c_char) -> Result<Constraint> {
    let grammar_json = unsafe { CStr::from_ptr(grammar_json) }
        .to_str()
        .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in grammar_json"))?;
    let grammar: TopLevelGrammar = serde_json::from_str(grammar_json)
        .map_err(|e| anyhow::anyhow!("Invalid JSON in grammar_json: {e}"))?;

    let mut limits = ParserLimits::default();
    macro_rules! set_limit {
        ($field:ident) => {
            if init.limits.$field != 0 {
                limits.$field = init.limits.$field;
            }
        };
    }
    set_limit!(max_items_in_row);
    set_limit!(initial_lexer_fuel);
    set_limit!(step_lexer_fuel);
    set_limit!(max_lexer_states);
    set_limit!(max_grammar_size);

    let tok_env = unsafe { (&*init.tokenizer).to_env() };
    let tok_parser = TokenParser::from_llguidance_json(
        tok_env,
        grammar,
        Logger::new(init.log_buffer_level, init.log_stderr_level),
        InferenceCapabilities {
            ff_tokens: init.ff_tokens_ok,
            backtrack: init.backtrack_ok,
            conditional_ff_tokens: false,
            fork: false,
        },
        limits,
        vec![],
    )?;

    Ok(Constraint::new(tok_parser))
}

impl CConstraint {
    fn get_error(&self) -> *const c_char {
        match &self.local_error {
            Some(e) => e.as_ptr() as *const c_char,
            None => std::ptr::null(),
        }
    }

    fn set_error(&mut self, e: &str) {
        self.constraint = None;
        self.local_error = Some(format!("{e}\0"));
    }
}

/// Create a new constraint from a grammar JSON string
/// Always returns a non-null value. Call llg_get_error() on the result to check for errors.
#[no_mangle]
pub extern "C" fn llg_new_constraint(
    init: &ConstraintInit,
    grammar_json: *const c_char,
) -> *mut CConstraint {
    let mut res = CConstraint {
        local_error: None,
        constraint: None,
    };

    match new_constraint(init, grammar_json) {
        Ok(constraint) => res.constraint = Some(constraint),
        Err(e) => res.set_error(&e.to_string()),
    };

    Box::into_raw(Box::new(res))
}

/// Get the error message from the constraint or null if there is no error
#[no_mangle]
pub extern "C" fn llg_get_error(cc: &CConstraint) -> *const c_char {
    cc.get_error()
}

/// Compute mask for the next token sampling
/// It typically takes up to a millisecond for a 100k tokenizer, so should be called in background.
/// Returns null on success, or an error message on failure (use llg_get_error() to get it again if needed).
/// When null is returned, the result is written to *res_p.
#[no_mangle]
pub extern "C" fn llg_compute_mask(cc: &mut CConstraint, res_p: *mut CMaskResult) -> *const c_char {
    if let Some(constraint) = &mut cc.constraint {
        match constraint.compute_mask() {
            Ok(r) => {
                let r = CMaskResult {
                    sample_mask: r
                        .sample_mask
                        .as_ref()
                        .map_or(std::ptr::null(), |m| unsafe { m.as_ptr() }),
                    is_stop: r.is_stop(),
                    temperature: constraint.temperature,
                };
                unsafe {
                    *res_p = r;
                }
                return std::ptr::null();
            }
            Err(e) => cc.set_error(&e.to_string()),
        }
    }
    cc.get_error()
}

/// Commit the token sampled with the mask returned from llg_compute_mask().
/// Can be run on the critical path of sampling (is fast).
/// Returns null on success, or an error message on failure (use llg_get_error() to get it again if needed).
/// When null is returned, the result is written to *res_p.
#[no_mangle]
pub extern "C" fn llg_commit_token(
    cc: &mut CConstraint,
    token: TokenId,
    res_p: *mut CCommitResult,
) -> *const c_char {
    if let Some(constraint) = &mut cc.constraint {
        let trie = constraint.parser.token_env.tok_trie();
        let token = if token < trie.vocab_size() as TokenId {
            Some(token)
        } else {
            None
        };
        match constraint.commit_token(token) {
            Ok(r) => {
                let res = if let Some(s) = r.unconditional_splice() {
                    CCommitResult {
                        tokens: s.ff_tokens.as_ptr(),
                        n_tokens: s.ff_tokens.len() as u32,
                        is_stop: r.is_stop(),
                    }
                } else {
                    CCommitResult {
                        tokens: std::ptr::null(),
                        n_tokens: 0,
                        is_stop: r.is_stop(),
                    }
                };
                unsafe { *res_p = res }
                return std::ptr::null();
            }
            Err(e) => cc.set_error(&e.to_string()),
        }
    }
    cc.get_error()
}

/// Construct a new tokenizer from the given TokenizerInit
#[no_mangle]
pub extern "C" fn llg_new_tokenizer(tok_init: &TokenizerInit) -> *mut CTokenizer {
    let tok = CTokenizer::from_init(tok_init);
    Box::into_raw(Box::new(tok))
}

/// Free the tokenizer. Should *NOT* be called while there are still constraints using it.
#[no_mangle]
pub extern "C" fn llg_free_tokenizer(tok: *mut CTokenizer) {
    unsafe {
        drop(Box::from_raw(tok));
    }
}

/// Free the constraint
#[no_mangle]
pub extern "C" fn llg_free_constraint(cc: *mut CConstraint) {
    unsafe {
        drop(Box::from_raw(cc));
    }
}
