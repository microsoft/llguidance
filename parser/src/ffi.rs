use std::{
    ffi::{c_char, c_void, CStr},
    sync::Arc,
};

use anyhow::{bail, ensure, Result};
use toktrie::{InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenizerEnv};

use crate::{
    api::{ParserLimits, RegexNode, TopLevelGrammar},
    lark_to_llguidance, CommitResult, Constraint, JsonCompileOptions, Logger, TokenParser,
};

struct CTokenizerInner {
    trie: TokTrie,
    tokenize_fn: LlgTokenizeFn,
    tokenize_user_data: *const c_void,
    tokenize_assumes_string: bool,
}
unsafe impl Send for CTokenizerInner {}
unsafe impl Sync for CTokenizerInner {}

impl CTokenizerInner {
    fn raw_tokenize(&self, s: &[u8]) -> Vec<toktrie::TokenId> {
        if let Some(tokenize_fn) = self.tokenize_fn {
            let mut res_toks = vec![0; s.len() / 4 + 5];
            let n_toks = tokenize_fn(
                self.tokenize_user_data,
                s.as_ptr(),
                s.len(),
                res_toks.as_mut_ptr(),
                res_toks.len(),
            );

            if n_toks > res_toks.len() {
                res_toks.resize(n_toks, 0);
                tokenize_fn(
                    self.tokenize_user_data,
                    s.as_ptr(),
                    s.len(),
                    res_toks.as_mut_ptr(),
                    res_toks.len(),
                );
            }

            res_toks.truncate(n_toks);
            res_toks
        } else {
            self.trie.greedy_tokenize(s)
        }
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

pub struct LlgTokenizer {
    token_env: TokEnv,
}

impl LlgTokenizer {
    fn from_init(init: &LlgTokenizerInit) -> Result<Self> {
        ensure!(
            init.tokenize_fn.is_some() || init.use_approximate_greedy_tokenize_fn,
            "Either tokenize_fn or use_approximate_greedy_tokenize_fn must be set"
        );
        let tokens = if init.tokenizer_json.is_null() {
            ensure!(
                !init.token_lens.is_null() && !init.token_bytes.is_null(),
                "token_lens and token_bytes must be set"
            );
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
            tokens
        } else {
            let tokenizer_json = unsafe { CStr::from_ptr(init.tokenizer_json) }
                .to_str()
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in tokenizer_json"))?;
            let tokenizer_json = serde_json::from_str(tokenizer_json)
                .map_err(|e| anyhow::anyhow!("Invalid JSON in tokenizer_json: {e}"))?;
            let mut token_bytes =
                crate::tokenizer_json::token_bytes_from_tokenizer_json(&tokenizer_json)?;

            let sz = init.vocab_size as usize;
            if token_bytes.len() < sz {
                token_bytes.resize(sz, vec![]);
            }

            token_bytes
        };

        let trie = TokTrie::from(&TokRxInfo::new(tokens.len() as u32, init.tok_eos), &tokens);

        Ok(LlgTokenizer {
            token_env: Arc::new(CTokenizerInner {
                trie,
                tokenize_assumes_string: init.tokenize_assumes_string && init.tokenize_fn.is_some(),
                tokenize_fn: init.tokenize_fn,
                tokenize_user_data: init.tokenize_user_data,
            }),
        })
    }

    fn to_env(&self) -> TokEnv {
        self.token_env.clone()
    }
}

pub type LlgToken = u32;

/// Tokenization function
/// Will not write more than output_tokens_len tokens (which can be 0)
/// Returns the total number of tokens (which can be more than output_tokens_len)
/// This function has to be thread-safe!
pub type LlgTokenizeFn = Option<
    extern "C" fn(
        user_data: *const c_void,
        bytes: *const u8,
        bytes_len: usize,
        output_tokens: *mut u32,
        output_tokens_len: usize,
    ) -> usize,
>;

/// Function which llg calls when an operation is done.
pub type LlgCallback = Option<extern "C" fn(user_data: *const c_void)>;

#[repr(C)]
pub struct LlgTokenizerInit {
    /// The number of tokens in the vocabulary
    pub vocab_size: u32,

    /// The token ID for the end of sentence token
    /// For chat mode, set it to end-of-turn token
    pub tok_eos: LlgToken,

    /// An array of the lengths of the token strings (vocab_size elements)
    pub token_lens: *const u32,

    /// A pointer to the token strings
    /// The length of this the sum of all token_lens
    pub token_bytes: *const u8,

    /// Instead of passing token_lens and token_bytes, this can be set to
    /// the contents of HF tokenizer.json file.
    pub tokenizer_json: *const c_char,

    /// Set to true to enable hack that works around the tokenize_fn only
    /// accepting valid UTF-8 strings and possibly adding <BOS> etc.
    /// TODO: the <BOS> bit not implemented yet
    pub tokenize_assumes_string: bool,

    /// Tokenization function, see LlgTokenizeFn docs.
    /// It should only tokenize the bytes and not add
    /// any <BOS> etc. It should also work on any byte sequence, including
    /// invalid UTF-8. If this is not the case, set tokenize_assumes_string to true.
    /// Either way, this function has to be thread-safe!
    pub tokenize_fn: LlgTokenizeFn,

    /// Set to true to not use tokenize_fn and instead tokenize greedily,
    /// which is often incorrect and may reduce accuracy.
    pub use_approximate_greedy_tokenize_fn: bool,

    /// User data to pass to the tokenize_fn
    pub tokenize_user_data: *const c_void,
}

#[repr(C)]
pub struct LlgConstraintInit {
    /// The tokenizer to use, created with llg_new_tokenizer()
    pub tokenizer: *const LlgTokenizer,
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

#[derive(Clone)]
pub struct LlgConstraintStep {
    /// The constraint to compute mask for.
    pub constraint: *mut LlgConstraint,
    /// Pointer to memory where the mask should be written.
    pub mask_dest: *mut u32,
    /// The length of the mask_dest array in bytes (not elements).
    pub mask_byte_len: usize,
}

unsafe impl Send for LlgConstraintStep {}

pub struct LlgConstraint {
    local_error: Option<String>,
    last_logs: String,
    pub(crate) constraint: Option<Constraint>,
    last_commit_result: CommitResult,
}

impl Clone for LlgConstraint {
    fn clone(&self) -> Self {
        LlgConstraint {
            local_error: self.local_error.clone(),
            last_logs: self.last_logs.clone(),
            constraint: self.constraint.clone(),
            last_commit_result: self.last_commit_result.clone(),
        }
    }
}

impl Default for LlgConstraint {
    fn default() -> Self {
        LlgConstraint {
            local_error: None,
            last_logs: "\x00".to_string(),
            constraint: None,
            last_commit_result: CommitResult::default(),
        }
    }
}

#[repr(C)]
pub struct LlgMaskResult {
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
pub struct LlgCommitResult {
    /// The tokens to append to the output if any
    /// This is valid until any call to llg_*() on the current constraint
    pub tokens: *const u32,
    /// The number of tokens in the tokens array (can be 0)
    pub n_tokens: u32,
    /// Should the sequence stop?
    pub is_stop: bool,
}

impl LlgCommitResult {
    pub fn from_commit_result(r: &CommitResult) -> Self {
        let len = r.ff_tokens.len() as u32;
        LlgCommitResult {
            tokens: if len == 0 {
                std::ptr::null()
            } else {
                r.ff_tokens.as_ptr()
            },
            n_tokens: len,
            is_stop: r.stop,
        }
    }
}

fn new_constraint_regex(init: &LlgConstraintInit, regex: *const c_char) -> Result<Constraint> {
    let regex = unsafe { CStr::from_ptr(regex) }
        .to_str()
        .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in regex"))?;
    let grammar = TopLevelGrammar::from_regex(RegexNode::Regex(regex.to_string()));
    new_constraint_core(init, grammar)
}

fn new_constraint_lark(init: &LlgConstraintInit, lark: *const c_char) -> Result<Constraint> {
    let lark = unsafe { CStr::from_ptr(lark) }
        .to_str()
        .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in lark"))?;
    let grammar = lark_to_llguidance(lark)?;
    new_constraint_core(init, grammar)
}

fn new_constraint_json(init: &LlgConstraintInit, json_schema: *const c_char) -> Result<Constraint> {
    let json_schema = unsafe { CStr::from_ptr(json_schema) }
        .to_str()
        .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in json_schema"))?;
    let json_schema = serde_json::from_str(json_schema)
        .map_err(|e| anyhow::anyhow!("Invalid JSON in json_schema: {e}"))?;
    let opts = JsonCompileOptions { compact: false };
    let grammar = opts
        .json_to_llg(&json_schema)
        .map_err(|e| anyhow::anyhow!("Error compiling JSON schema to LLG: {e}"))?;
    new_constraint_core(init, grammar)
}

fn new_constraint(init: &LlgConstraintInit, grammar_json: *const c_char) -> Result<Constraint> {
    let grammar_json = unsafe { CStr::from_ptr(grammar_json) }
        .to_str()
        .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in grammar_json"))?;
    let grammar: TopLevelGrammar = serde_json::from_str(grammar_json)
        .map_err(|e| anyhow::anyhow!("Invalid JSON in grammar_json: {e}"))?;
    new_constraint_core(init, grammar)
}

fn new_constraint_any(
    init: &LlgConstraintInit,
    constraint_type: *const c_char,
    data: *const c_char,
) -> Result<Constraint> {
    let tp = unsafe { CStr::from_ptr(constraint_type) }
        .to_str()
        .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in constraint_type"))?;
    match tp {
        "regex" => new_constraint_regex(init, data),
        "json" | "json_schema" => new_constraint_json(init, data),
        "lark" => new_constraint_lark(init, data),
        "llguidance" | "guidance" => new_constraint(init, data),
        _ => bail!("unknown constraint type: {tp}"),
    }
}

fn new_constraint_core(init: &LlgConstraintInit, grammar: TopLevelGrammar) -> Result<Constraint> {
    if init.tokenizer.is_null() {
        bail!("Tokenizer is null");
    }

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
        init.limits.clone(),
        vec![],
    )?;

    Ok(Constraint::new(tok_parser))
}

impl LlgConstraint {
    fn get_error(&self) -> *const c_char {
        match &self.local_error {
            Some(e) => e.as_ptr() as *const c_char,
            None => std::ptr::null(),
        }
    }

    fn get_error_code(&self) -> i32 {
        if self.local_error.is_some() {
            -1
        } else {
            0
        }
    }

    pub(crate) fn set_error(&mut self, e: &str) {
        self.constraint = None;
        self.local_error = Some(format!("{e}\0"));
    }
}

/// Set the default values for the ConstraintInit
/// Disables ff_tokens and backtracking, enables warnings on stderr
/// and all logging to the buffer (get with llg_flush_logs()).
/// You need to set the tokenizer field manually.
#[no_mangle]
pub extern "C" fn llg_constraint_init_set_defaults(
    init: &mut LlgConstraintInit,
    tokenizer: *const LlgTokenizer,
) {
    *init = LlgConstraintInit {
        tokenizer,
        log_buffer_level: 0,
        log_stderr_level: 1,
        ff_tokens_ok: false,
        backtrack_ok: false,
        limits: ParserLimits::default(),
    };
}

fn return_constraint(c: Result<Constraint>) -> *mut LlgConstraint {
    let mut res = LlgConstraint::default();

    match c {
        Ok(constraint) => res.constraint = Some(constraint),
        Err(e) => res.set_error(&e.to_string()),
    };

    Box::into_raw(Box::new(res))
}

/// Create a new constraint from a grammar JSON string
/// Always returns a non-null value. Call llg_get_error() on the result to check for errors.
#[no_mangle]
pub extern "C" fn llg_new_constraint(
    init: &LlgConstraintInit,
    grammar_json: *const c_char,
) -> *mut LlgConstraint {
    return_constraint(new_constraint(init, grammar_json))
}

/// Create a new constraint from a given regular expression
/// Always returns a non-null value. Call llg_get_error() on the result to check for errors.
#[no_mangle]
pub extern "C" fn llg_new_constraint_regex(
    init: &LlgConstraintInit,
    regex: *const c_char,
) -> *mut LlgConstraint {
    return_constraint(new_constraint_regex(init, regex))
}

/// Create a new constraint from a given JSON schema
/// Always returns a non-null value. Call llg_get_error() on the result to check for errors.
#[no_mangle]
pub extern "C" fn llg_new_constraint_json(
    init: &LlgConstraintInit,
    json_schema: *const c_char,
) -> *mut LlgConstraint {
    return_constraint(new_constraint_json(init, json_schema))
}

/// Create a new constraint from a given lark grammar
/// Always returns a non-null value. Call llg_get_error() on the result to check for errors.
#[no_mangle]
pub extern "C" fn llg_new_constraint_lark(
    init: &LlgConstraintInit,
    lark: *const c_char,
) -> *mut LlgConstraint {
    return_constraint(new_constraint_lark(init, lark))
}

/// Create a new constraint with specified type
/// Type can be one of "regex", "json_schema" (or "json"), "lark", "llguidance" (or "guidance")
/// Always returns a non-null value. Call llg_get_error() on the result to check for errors.
#[no_mangle]
pub extern "C" fn llg_new_constraint_any(
    init: &LlgConstraintInit,
    constraint_type: *const c_char,
    data: *const c_char,
) -> *mut LlgConstraint {
    return_constraint(new_constraint_any(init, constraint_type, data))
}

/// Get the error message from the constraint or null if there is no error.
/// After it returns a non-null value, it will always return it until the constraint is freed
/// using llg_free_constraint() (at which point the pointer will be invalid).
#[no_mangle]
pub extern "C" fn llg_get_error(cc: &LlgConstraint) -> *const c_char {
    cc.get_error()
}

/// Get the current temperature of the constraint.
/// It is updated by mask computation.
#[no_mangle]
pub extern "C" fn llg_get_temperature(cc: &LlgConstraint) -> f32 {
    cc.constraint.as_ref().map_or(0.0, |c| c.temperature)
}

/// Check if constraint is stopped (cannot be extended further).
#[no_mangle]
pub extern "C" fn llg_is_stopped(cc: &LlgConstraint) -> bool {
    cc.constraint
        .as_ref()
        .map_or(true, |c| c.step_result().is_stop())
}

/// Compute mask for the next token sampling
/// It typically takes up to a millisecond for a 100k tokenizer, so should be called in background.
/// Returns 0 on success and -1 on error (use llg_get_error() to get the exact error).
/// When 0 is returned, the result is written to *res_p.
#[no_mangle]
pub extern "C" fn llg_compute_mask(cc: &mut LlgConstraint, res_p: *mut LlgMaskResult) -> i32 {
    if let Some(constraint) = &mut cc.constraint {
        match constraint.compute_mask() {
            Ok(r) => {
                let r = LlgMaskResult {
                    sample_mask: r
                        .sample_mask
                        .as_ref()
                        .map_or(std::ptr::null(), |m| unsafe { m.as_ptr() }),
                    is_stop: r.is_stop(),
                    temperature: constraint.temperature,
                };
                unsafe { *res_p = r };
            }
            Err(e) => cc.set_error(&e.to_string()),
        }
    }
    cc.get_error_code()
}

/// Commit the token sampled with the mask returned from llg_compute_mask().
/// Can be run on the critical path of sampling (is fast).
/// Returns 0 on success and -1 on error (use llg_get_error() to get the exact error).
/// When 0 is returned, the result is written to *res_p.
#[no_mangle]
pub extern "C" fn llg_commit_token(
    cc: &mut LlgConstraint,
    token: LlgToken,
    res_p: *mut LlgCommitResult,
) -> i32 {
    if let Some(constraint) = &mut cc.constraint {
        let trie = constraint.parser.token_env.tok_trie();
        let token = if token < trie.vocab_size() as LlgToken {
            Some(token)
        } else {
            None
        };
        match constraint.commit_token(token) {
            Ok(r) => {
                // store it, so it survives until the next call to llg_*()
                cc.last_commit_result = r;
                let res = LlgCommitResult::from_commit_result(&cc.last_commit_result);
                unsafe { *res_p = res };
            }
            Err(e) => cc.set_error(&e.to_string()),
        }
    }
    cc.get_error_code()
}

/// Compute mask for several constraints in parallel.
#[no_mangle]
pub extern "C" fn llg_par_compute_mask(
    steps: *const LlgConstraintStep,
    n_steps: usize,
    user_data: *const c_void,
    done_cb: LlgCallback,
) {
    if steps.is_null() {
        panic!("llg_par_compute_mask: steps is null");
    }

    #[cfg(feature = "rayon")]
    {
        let steps = unsafe { std::slice::from_raw_parts(steps, n_steps).to_vec() };
        crate::ffi_par::par_compute_mask(steps, user_data, done_cb);
    }

    #[cfg(not(feature = "rayon"))]
    {
        let _ = (steps, n_steps, user_data, done_cb);
        panic!("llg_par_compute_mask: rayon feature is not enabled");
    }
}

/// Clone the constraint
#[no_mangle]
pub extern "C" fn llg_clone_constraint(cc: &LlgConstraint) -> *mut LlgConstraint {
    Box::into_raw(Box::new(cc.clone()))
}

/// Construct a new tokenizer from the given TokenizerInit
#[no_mangle]
pub extern "C" fn llg_new_tokenizer(
    tok_init: &LlgTokenizerInit,
    error_string: *mut c_char,
    error_string_len: usize,
) -> *mut LlgTokenizer {
    match LlgTokenizer::from_init(tok_init) {
        Ok(tok) => Box::into_raw(Box::new(tok)),
        Err(e) => {
            if error_string_len > 0 {
                let e = e.to_string();
                let e = e.as_bytes();
                let len = std::cmp::min(e.len(), error_string_len - 1);
                unsafe {
                    std::ptr::copy_nonoverlapping(e.as_ptr(), error_string as *mut u8, len);
                    *error_string.add(len) = 0;
                }
            }
            std::ptr::null_mut()
        }
    }
}

/// Clone a tokenizer.
/// This increments a reference count and does a small allocation.
#[no_mangle]
pub extern "C" fn llg_clone_tokenizer(tok: &LlgTokenizer) -> *mut LlgTokenizer {
    Box::into_raw(Box::new(LlgTokenizer {
        token_env: tok.token_env.clone(),
    }))
}

/// Tokenize the given bytes and return the tokens.
/// Always returns the number of tokens that would be written to output_tokens
/// if output_tokens_len was large enough.
#[no_mangle]
pub extern "C" fn llg_tokenize_bytes(
    tok: &LlgTokenizer,
    bytes: *const u8,
    bytes_len: usize,
    output_tokens: *mut u32,
    output_tokens_len: usize,
) -> usize {
    let tokens = tok
        .token_env
        .tokenize_bytes(unsafe { std::slice::from_raw_parts(bytes, bytes_len) });
    let n_toks = tokens.len();
    let to_copy = std::cmp::min(n_toks, output_tokens_len);
    unsafe {
        std::ptr::copy_nonoverlapping(tokens.as_ptr(), output_tokens, to_copy);
    }
    n_toks
}

/// Return a string representation of the tokens, useful for debugging.
/// The output is null-terminated.
/// Returns the number of bytes that would be written to output if output_len was large enough.
#[no_mangle]
pub extern "C" fn llg_stringify_tokens(
    tok: &LlgTokenizer,
    tokens: *const u32,
    n_tokens: usize,
    output: *mut c_char,
    output_len: usize,
) -> usize {
    let trie = tok.token_env.tok_trie();
    let tokens = unsafe { std::slice::from_raw_parts(tokens, n_tokens) };
    let s = trie.tokens_dbg(tokens);
    let s = s.as_bytes();
    let len = std::cmp::min(s.len(), output_len - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), output as *mut u8, len);
        *output.add(len) = 0;
    }
    s.len() + 1
}

/// Free the tokenizer. Should *NOT* be called while there are still constraints using it.
#[no_mangle]
pub extern "C" fn llg_free_tokenizer(tok: *mut LlgTokenizer) {
    unsafe {
        drop(Box::from_raw(tok));
    }
}

/// Free the constraint
#[no_mangle]
pub extern "C" fn llg_free_constraint(cc: *mut LlgConstraint) {
    unsafe {
        drop(Box::from_raw(cc));
    }
}

/// Get the logs from the constraint, since last call to this function.
/// The logs are null-terminated.
/// The logs are kept in the constraint until the next call to this function
/// or until the constraint is freed.
#[no_mangle]
pub extern "C" fn llg_flush_logs(cc: &mut LlgConstraint) -> *const c_char {
    if let Some(constraint) = &mut cc.constraint {
        let s = constraint.flush_logs();
        if s.contains('\0') {
            cc.last_logs = s.replace('\0', "\\0");
        } else {
            cc.last_logs = s;
        }
        cc.last_logs.push('\0');
    }
    cc.last_logs.as_ptr() as *const c_char
}
