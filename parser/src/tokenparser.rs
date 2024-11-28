use std::{hint::black_box, panic::AssertUnwindSafe, sync::Arc, time::Duration};

use crate::{
    api::{ParserLimits, StopReason, TopLevelGrammar},
    earley::{
        grammars_from_json, BiasComputer, DefaultBiasComputer, Parser, ParserError, ParserStats,
    },
    infoln, warn, Logger,
};
use anyhow::{ensure, Result};
use serde_json::json;
use toktrie::{InferenceCapabilities, SimpleVob, TokEnv, TokenId};

#[derive(Clone)]
pub struct TokenParser {
    pub token_env: TokEnv,
    pub parser: Parser,
    pub compute_mask_start_time: instant::Instant,
    pub last_bias_time: Duration,
    pub inference_caps: InferenceCapabilities,
    pub logger: Logger,
    pub limits: ParserLimits,
    pub bias_computer: Arc<dyn BiasComputer>,
    last_step_stats: ParserStats,
    max_step_stats: ParserStats,
    test_trace: bool,
    eos_token: TokenId,

    is_accepting_cache: Option<bool>,
    stop_reason: StopReason,
    error_message: Option<String>,

    max_tokens_total: usize,

    // tokens currently in KV cache
    llm_tokens: Vec<TokenId>,
    llm_bytes: Vec<u8>,
    grm_prefix: Vec<u8>,
    is_fresh: bool,
}

impl TokenParser {
    pub fn from_llguidance_json(
        token_env: TokEnv,
        top_grammar: TopLevelGrammar,
        logger: Logger,
        inference_caps: InferenceCapabilities,
        limits: ParserLimits,
        extra_lexemes: Vec<String>,
    ) -> Result<Self> {
        crate::panics::catch_unwind(AssertUnwindSafe(|| {
            Self::from_llguidance_json_inner(
                token_env,
                top_grammar,
                logger,
                inference_caps,
                limits,
                extra_lexemes,
            )
        }))
    }

    fn from_llguidance_json_inner(
        token_env: TokEnv,
        top_grammar: TopLevelGrammar,
        mut logger: Logger,
        inference_caps: InferenceCapabilities,
        limits: ParserLimits,
        extra_lexemes: Vec<String>,
    ) -> Result<Self> {
        ensure!(
            token_env.tokenize_is_canonical() || !inference_caps.ff_tokens,
            "ff_tokens requires canonical tokenization"
        );
        ensure!(
            !inference_caps.backtrack || inference_caps.ff_tokens,
            "backtrack requires ff_tokens"
        );

        let compute_mask_start_time = instant::Instant::now();
        let test_trace = top_grammar.test_trace;
        let max_tokens = top_grammar.max_tokens.unwrap_or(usize::MAX);
        let compiled_grammar = grammars_from_json(
            top_grammar,
            &token_env,
            &mut logger,
            limits.clone(),
            extra_lexemes,
        )?;
        let parser = Parser::new(compiled_grammar, limits.clone())?;
        let eos_token = token_env.tok_trie().eos_token();

        Ok(TokenParser {
            bias_computer: Arc::new(DefaultBiasComputer::new(token_env.clone())),
            logger,
            test_trace,
            token_env,
            inference_caps,
            limits,
            max_step_stats: ParserStats::default(),
            last_step_stats: ParserStats::default(),
            compute_mask_start_time,
            is_accepting_cache: None,
            stop_reason: StopReason::NotStopped,
            error_message: None,
            parser,
            eos_token,
            llm_tokens: Vec::new(),
            llm_bytes: Vec::new(),
            grm_prefix: Vec::new(),
            max_tokens_total: max_tokens,
            last_bias_time: Duration::from_secs(0),
            is_fresh: true,
        })
    }

    // regular .clone() uses a shared lexer state
    pub fn deep_clone(&self) -> Self {
        let mut copy = self.clone();
        copy.parser = self.parser.deep_clone();
        copy
    }

    pub fn stop_reason(&self) -> StopReason {
        self.stop_reason
    }

    pub fn is_fresh(&self) -> bool {
        self.is_fresh
    }

    pub fn parser_stats(&self) -> &ParserStats {
        self.parser.stats()
    }

    pub fn last_step_stats(&self) -> &ParserStats {
        &self.last_step_stats
    }

    pub fn max_step_stats(&self) -> &ParserStats {
        &self.max_step_stats
    }

    pub fn num_tokens(&self) -> usize {
        self.llm_tokens.len()
    }

    pub fn final_bytes(&self) -> &[u8] {
        &self.llm_bytes[self.grm_prefix.len()..]
    }

    pub fn is_accepting(&mut self) -> bool {
        if let Some(acc) = self.is_accepting_cache {
            acc
        } else {
            let r = !self.has_ff_bytes() && self.parser.is_accepting();
            self.is_accepting_cache = Some(r);
            r
        }
    }

    pub fn bytes_since(&self, mut idx: usize) -> &[u8] {
        idx += self.grm_prefix.len();
        let endp = std::cmp::min(self.llm_bytes.len(), self.parser.hidden_start());
        if idx >= self.llm_bytes.len() || idx >= endp {
            return &[];
        }
        &self.llm_bytes[idx..endp]
    }

    pub fn start_without_prompt(&mut self) {
        infoln!(
            self,
            "initial lexer cost: {} (no prompt)",
            self.parser.lexer_stats()
        );

        assert!(self.is_fresh);
        self.is_fresh = false;
    }

    pub fn process_prompt(&mut self, prompt: Vec<TokenId>) -> Vec<TokenId> {
        infoln!(self, "initial lexer cost: {}", self.parser.lexer_stats());

        assert!(self.token_env.tokenize_is_canonical());
        assert!(self.is_fresh);
        self.is_fresh = false;

        assert!(self.llm_tokens.is_empty());

        let trie = self.token_env.tok_trie();
        infoln!(self, "prompt: {}", trie.tokens_dbg(&prompt));
        let mut prompt_bytes = trie.decode_raw(&prompt);
        self.parser.force_bytes();
        let grm_bytes = self.parser.get_bytes().to_vec();
        prompt_bytes.extend_from_slice(&grm_bytes);
        let tokens = self.token_env.tokenize_bytes_prefix(&prompt_bytes);
        infoln!(self, "prompt+grm: {}", trie.tokens_dbg(&tokens));
        let (chop_tokens, chop_bytes) = self
            .parser
            .with_recognizer(|r| trie.chop_tokens(r, &tokens));
        let res_prompt = tokens[..tokens.len() - chop_tokens].to_vec();

        // if we moved a bunch of grammar to the prompt, update llm_tokens to reflect that
        if chop_bytes <= grm_bytes.len() {
            self.llm_bytes = grm_bytes[0..grm_bytes.len() - chop_bytes].to_vec();
            self.llm_tokens = self.token_env.tokenize_bytes_prefix(&self.llm_bytes);
            self.parser.apply_forced(self.llm_bytes.len());
            let decoded = self.tok_trie().decode_raw(&self.llm_tokens);
            if self.llm_bytes.len() > 0
                && decoded.len() > 0
                && &decoded[1..] == &self.llm_bytes
                && decoded[0] == b' '
            {
                infoln!(self, "applying <s>space hack");
                self.grm_prefix = decoded[0..1].to_vec();
                self.llm_bytes = decoded;
            }
            infoln!(self, "ini_tokens: {}", trie.tokens_dbg(&self.llm_tokens));
        } else {
            // pretend the final bit of prompt was the prefix of the grammar
            self.grm_prefix = prompt_bytes
                [prompt_bytes.len() - chop_bytes..prompt_bytes.len() - grm_bytes.len()]
                .to_vec();
            infoln!(
                self,
                "force_prefix: {:?}",
                String::from_utf8_lossy(&self.grm_prefix)
            );
        }

        infoln!(self, "res_prompt: {}", trie.tokens_dbg(&res_prompt));
        if self.test_trace {
            self.test_trace_json(&json!({
                "prompt": trie.test_trace_tokens(&prompt),
                "res_prompt": trie.test_trace_tokens(&res_prompt),
            }));
        }
        res_prompt
    }

    fn test_trace_json(&mut self, j: &serde_json::Value) {
        if self.test_trace {
            infoln!(self, "TEST: {}", serde_json::to_string(j).unwrap());
        }
    }

    fn stop(&mut self, warn: &str, reason: StopReason) -> anyhow::Error {
        if warn.len() > 0 {
            self.error_message = Some(warn.to_string());
            warn!(self, "{}; stopping", warn);
        }
        self.stop_reason = reason;
        self.anyhow_error()
    }

    fn tok_trie(&self) -> &toktrie::TokTrie {
        self.token_env.tok_trie()
    }

    pub fn error_message(&self) -> Option<String> {
        self.error_message.clone()
    }

    fn check_initialized(&self, lbl: &str) -> Result<()> {
        ensure!(!self.is_fresh, "process_prompt() not called in {}", lbl);
        ensure!(
            self.stop_reason == StopReason::NotStopped,
            "parser stopped in {}; {}",
            lbl,
            self.error_message()
                .unwrap_or("no error message".to_string())
        );
        Ok(())
    }

    pub fn validate_token(&mut self, token: TokenId) -> Result<bool> {
        self.check_initialized("validate_token")?;
        let bytes = self.tok_trie().decode_raw(&[token]);
        let n_valid = self.parser.validate_bytes(&bytes);
        assert!(n_valid <= bytes.len());
        Ok(n_valid == bytes.len())
    }

    /// Returns how many of the passed tokens can be accepted by the parser.
    /// It does not tokenize forced bytes, so will accept non-canonical tokenizations.
    /// If called with more than one token, it may ignore max_tokens constraints.
    pub fn validate_tokens_raw(&mut self, tokens: &[TokenId]) -> Result<usize> {
        self.check_initialized("validate_tokens_raw")?;

        if tokens.is_empty() {
            return Ok(0);
        }

        if tokens.len() == 1 {
            return if self.validate_token(tokens[0])? {
                Ok(1)
            } else {
                Ok(0)
            };
        }

        let bytes = self.tok_trie().decode_raw(tokens);
        let n_valid = self.parser.validate_bytes(&bytes);
        assert!(n_valid <= bytes.len());

        // fast paths
        if n_valid == bytes.len() {
            return Ok(tokens.len());
        }
        if n_valid == 0 {
            return Ok(0);
        }

        let mut byte_ptr = 0;
        for (token_ptr, tok) in tokens.iter().enumerate() {
            byte_ptr += self.tok_trie().token(*tok).len();
            if byte_ptr > n_valid {
                return Ok(token_ptr);
            }
        }
        Ok(tokens.len())
    }

    fn anyhow_error(&self) -> anyhow::Error {
        anyhow::anyhow!(self
            .error_message
            .clone()
            .unwrap_or(self.stop_reason.to_string()))
    }

    // compute_mask() is a top-level method in this file.
    // compute_mask() is called by Constraint::compute_mask().
    pub fn compute_mask(&mut self) -> Result<SimpleVob> {
        self.compute_mask_start_time = instant::Instant::now();

        self.check_initialized("compute_mask")?;

        infoln!(self, "compute_mask");

        let mut prefix = self.compute_ff_bytes();

        // if ff_tokens is enabled, we assume the user has already called compute_ff_tokens()
        if !self.inference_caps.ff_tokens
            && !self.parser.grammar().lexer_spec().no_forcing
            && self.token_env.tokenize_is_canonical()
        {
            let (ff_tokens, token_prefix) = self.ff_bytes_to_tokens(prefix);
            if ff_tokens.len() > 0 {
                let t = ff_tokens[0];
                infoln!(self, "forcing ff_token by mask: {}", t);
                let mask = self.tok_trie().singleton_token_set(t);
                return Ok(mask);
            } else {
                // no tokens, so we got all our bytes back
                prefix = token_prefix;
            }
        }

        let mut allowed_tokens = self.compute_bias(&prefix);

        if self.is_accepting() {
            allowed_tokens.allow_token(self.eos_token);
        }

        self.log_final(&prefix, &allowed_tokens);

        if allowed_tokens.num_set() == 0 {
            infoln!(self, "no tokens allowed, stopping");
            return Err(self.stop("", StopReason::NoExtensionBias));
        }

        Ok(allowed_tokens)
    }

    fn stop_for_parser_error(&mut self, pref: &str, err: ParserError) -> anyhow::Error {
        self.stop(&format!("{}{}", pref, err.message()), err.stop_reason())
    }

    fn apply_token(&mut self, tok_id: TokenId) -> Result<usize> {
        let trie = self.token_env.tok_trie();
        self.is_accepting_cache = None;
        self.llm_tokens.push(tok_id);

        let tok_bytes = trie.token(tok_id);

        // first, check we're still in grm_prefix
        let prefix_len = self.grm_prefix.len().saturating_sub(self.llm_bytes.len());
        let tok_bytes = if prefix_len > 0 {
            let to_apply = &tok_bytes[0..std::cmp::min(tok_bytes.len(), prefix_len)];
            self.llm_bytes.extend_from_slice(to_apply);

            if self.grm_prefix[0..self.llm_bytes.len()] != self.llm_bytes {
                return Err(self.stop(
                    &format!(
                        "prefix mismatch: applying {:?}; {:?} vs {:?}",
                        String::from_utf8_lossy(to_apply),
                        String::from_utf8_lossy(&self.grm_prefix),
                        String::from_utf8_lossy(&self.llm_bytes)
                    ),
                    StopReason::InternalError,
                ));
            }

            if prefix_len < tok_bytes.len() {
                &tok_bytes[prefix_len..]
            } else {
                // still completely in prefix, nothing more to apply
                return Ok(0);
            }
        } else {
            tok_bytes
        };

        if let Some(err) = self.parser.get_error() {
            return Err(self.stop_for_parser_error("", err));
        }

        // now apply normally
        match self.parser.apply_token(tok_bytes) {
            Err(e) => {
                return Err(self.stop(
                    &format!("Parser Error: {}", e),
                    StopReason::ParserTooComplex, // TODO - there are other reasons
                ));
            }
            Ok(backtrack_bytes0) => {
                self.llm_bytes.extend_from_slice(tok_bytes);

                if backtrack_bytes0 != 0 {
                    let mut backtrack_bytes: isize = backtrack_bytes0.try_into().unwrap();
                    let mut backtrack_tokens = 0;
                    while backtrack_bytes > 0 {
                        let tok_off = self.llm_tokens.len() - backtrack_tokens;
                        if tok_off == 0 {
                            break; // we can't backtrack any further
                        }
                        let tok = self.llm_tokens[tok_off - 1];
                        backtrack_bytes -= trie.token(tok).len() as isize;
                        backtrack_tokens += 1;
                    }
                    assert!(backtrack_tokens > 0);

                    let byte_ptr = self.llm_bytes.len() - backtrack_bytes0;
                    infoln!(
                        self,
                        "backtrack: {} tokens / {} bytes (deletes: {:?})",
                        backtrack_tokens,
                        backtrack_bytes0,
                        String::from_utf8_lossy(&self.llm_bytes[byte_ptr..])
                    );
                    self.llm_bytes.truncate(byte_ptr);

                    let token_ptr = self.llm_tokens.len() - backtrack_tokens;
                    if !self.inference_caps.backtrack {
                        warn!(
                            self,
                            "can't backtrack over {}; this may confuse the model",
                            trie.tokens_dbg(&self.llm_tokens[token_ptr..])
                        );
                        // pretend there's no backtrack
                        backtrack_tokens = 0;
                    } else {
                        // make sure the parser know we actually don't have
                        // the non-backtracked bytes of backtracked token
                        self.parser
                            .additional_backtrack((-backtrack_bytes).try_into().unwrap());
                    }
                    self.llm_tokens.truncate(token_ptr);
                    return Ok(backtrack_tokens);
                }
            }
        }

        Ok(0)
    }

    fn pending_grm_prefix(&self) -> &[u8] {
        &self.grm_prefix[std::cmp::min(self.grm_prefix.len(), self.llm_bytes.len())..]
    }

    fn has_ff_bytes(&self) -> bool {
        self.pending_grm_prefix().len() > 0 || self.parser.currently_forced_bytes().len() > 0
    }

    fn compute_ff_bytes(&mut self) -> Vec<u8> {
        // PERF: in some cases, this may be long
        let mut new_forced = self.parser.force_bytes().to_vec();

        // handle grm_prefix we might have injected
        if self.llm_bytes.len() < self.grm_prefix.len() {
            let mut inject = self.grm_prefix[self.llm_bytes.len()..].to_vec();
            infoln!(
                self,
                "injecting prefix: {:?}",
                String::from_utf8_lossy(&inject)
            );
            inject.extend_from_slice(&new_forced);
            new_forced = inject;
        }

        new_forced
    }

    /// Converts forced bytes into tokens.
    /// Also returns any bytes that need to be prefix of the
    /// next sampled token (token healing).
    fn ff_bytes_to_tokens(&mut self, forced_bytes: Vec<u8>) -> (Vec<TokenId>, Vec<u8>) {
        let trie = self.token_env.tok_trie();

        let mut token_prefix = Vec::new();

        let do_force = forced_bytes.len() > 0 && self.token_env.tokenize_is_canonical();
        if do_force {
            let mut grm_tokens = self.token_env.tokenize_bytes_prefix(&forced_bytes);
            infoln!(
                self,
                "forced: {} bytes:{:?} tokens:{:?}",
                trie.tokens_dbg(&grm_tokens),
                forced_bytes,
                grm_tokens
            );
            let (chop_tokens, chop_bytes) = self
                .parser
                .with_recognizer(|r| trie.chop_tokens(r, &grm_tokens));
            infoln!(self, "chop: {} tokens, {} bytes", chop_tokens, chop_bytes);
            token_prefix = forced_bytes[forced_bytes.len() - chop_bytes..].to_vec();
            // here we remove a suffix from grm_tokens that could be possibly tokenized differently
            grm_tokens.truncate(grm_tokens.len() - chop_tokens);

            if grm_tokens.len() > 0 {
                infoln!(self, "fixed_tokens: {}", trie.tokens_dbg(&grm_tokens),);
                return (grm_tokens, token_prefix);
            } else {
                infoln!(self, "no fixed tokens");
            }
        } else if forced_bytes.len() > 0 {
            infoln!(self, "not-forcing {} bytes", forced_bytes.len());
            token_prefix = forced_bytes;
        }

        (Vec::new(), token_prefix)
    }

    fn compute_bias(&mut self, token_prefix: &Vec<u8>) -> SimpleVob {
        let pre_stats = self.parser.stats().clone();
        let set = self.parser.compute_bias(&*self.bias_computer, token_prefix);
        let p_stats = self.parser.stats().delta(&pre_stats);
        self.last_bias_time = Duration::from_micros(p_stats.compute_time_us);
        self.last_step_stats = p_stats.clone();
        self.max_step_stats = self.max_step_stats.max(&p_stats);
        set
    }

    fn log_final(&mut self, token_prefix: &Vec<u8>, allowed_tokens: &SimpleVob) {
        infoln!(
            self,
            "step-stats: {}us; {} lex fuel; {} items; {}",
            self.compute_mask_start_time.elapsed().as_micros(),
            self.last_step_stats.lexer_cost,
            self.last_step_stats.all_items,
            self.parser.lexer_stats(),
        );

        infoln!(
            self,
            "bias: (pref: {:?}; accpt: {}; temp: {:.3}) {}",
            String::from_utf8_lossy(&token_prefix),
            self.is_accepting_cache.unwrap(),
            self.parser.temperature().unwrap_or(0.0),
            self.token_env.tok_trie().token_set_dbg(&allowed_tokens)
        );
    }

    pub fn temperature(&self) -> Option<f32> {
        self.parser.temperature()
    }

    /// Extend the current state of the parser with given token.
    /// Returns number of tokens to backtrack if any.
    pub fn consume_token(&mut self, token: TokenId) -> Result<usize> {
        self.check_initialized("consume_token")?;

        if self.max_tokens_total == 0 {
            return Err(self.stop("max_tokens_total reached", StopReason::MaxTokensTotal));
        }
        self.max_tokens_total -= 1;

        if token == self.eos_token {
            if self.parser.scan_eos() {
                // it got scanned correctly, so we remove it
                infoln!(self, "scanned eos_token");
                // if self.inference_caps.backtrack {
                //     return Ok(1);
                // } else {
                //     warn!(self, "can't backtrack over eos_token");
                //     return Ok(0);
                // }
                // don't backtrack it for now, fails tests
                return Ok(0);
            } else {
                let accepting = self.is_accepting();
                infoln!(self, "didn't scan eos_token; accept={}", accepting);
                if accepting {
                    self.llm_tokens.push(token);
                    return Ok(0);
                }
            }
        }

        let apply_res = self.apply_token(token);
        self.parser.log_row_infos("post-apply");
        match apply_res {
            Err(_) => Err(self.anyhow_error()),
            Ok(n) => {
                self.parser.filter_max_tokens();
                Ok(n)
            }
        }
    }

    /// Check whether the current parser state forces the sequence to stop.
    /// If so, puts the parser in stop state and returns true.
    /// Otherwise, returns false.
    /// This generally should be called after consume_token().
    pub fn check_stop(&mut self) -> Result<bool> {
        let empty_token_prefix = !self.has_ff_bytes();
        let pending_eos = self.llm_tokens.last() == Some(&self.eos_token);
        let lexer_bytes = self.parser.has_pending_lexeme_bytes();
        let is_accepting = self.is_accepting();
        let can_advance = self.parser.can_advance();
        let parser_done = is_accepting && (!can_advance || pending_eos);
        infoln!(
            self,
            "parser_done: {parser_done}; lexer_bytes: {lexer_bytes}; \
                can_advance: {can_advance} (eos:{pending_eos}); \
                accept: {is_accepting}; \
                empty_token_prefix: {empty_token_prefix}"
        );
        assert!(!is_accepting || empty_token_prefix);

        if parser_done {
            infoln!(
                self,
                "only eos token allowed, stopping; accepting: {}",
                is_accepting
            );
            let reason = if pending_eos {
                StopReason::EndOfSentence
            } else {
                StopReason::NoExtension
            };
            self.stop("", reason);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if there are any tokens to fast-forward, forced by the current
    /// parser state.
    pub fn compute_ff_tokens(&mut self) -> Vec<TokenId> {
        // force after scanning tokens from LLM (this may walk the parser some more)
        let ff_bytes = self.compute_ff_bytes();
        let (ff_tokens, _token_prefix) = self.ff_bytes_to_tokens(ff_bytes);
        ff_tokens
    }

    /// Compute and then consume fast-forward tokens.
    pub fn consume_ff_tokens(&mut self) -> Result<Vec<TokenId>> {
        let ff_tokens = self.compute_ff_tokens();
        for &t in &ff_tokens {
            let num_backtrack = self.consume_token(t)?;
            if num_backtrack > 0 {
                return Err(self.stop(
                    &format!("backtrack required after ff_token: {}", t),
                    StopReason::InternalError,
                ));
            }
        }
        Ok(ff_tokens)
    }

    /// This function documents typical use of this interface.
    /// The `tokens` array simulates tokens being sampled.
    #[allow(dead_code)]
    fn typical_use(&mut self, prompt: Vec<TokenId>) -> Result<()> {
        // First, check if we need to token-heal the prompt,
        // and if there are some tokens forced by the beginning
        // of the grammar.
        let new_prompt = self.process_prompt(prompt);

        // pass new prompt to inference engine
        black_box(new_prompt);

        let mut tokens = vec![];

        loop {
            let temp = self.temperature();
            let mask = self.compute_mask()?;

            // model forward pass in parallel with compute_mask() goes here

            // simulate sampling a token with given mask
            black_box((temp, mask));
            let sampled_token = 42;

            let num_backtrack = self.consume_token(sampled_token)?;

            if num_backtrack == 0 {
                // normal situation - the token was accepted
                tokens.push(sampled_token);
            } else {
                // this will only happen if you enable backtrack
                assert!(self.inference_caps.backtrack);
                if num_backtrack == 1 {
                    // don't add the token to the list
                } else if num_backtrack > 1 {
                    // backtrack
                    tokens.truncate(tokens.len() - num_backtrack - 1);
                }
            }

            // This is optional; if you don't check, compute_mask() will
            // return an error when it cannot continue anymore.
            // If you check here, you can distinguish between normal stop
            // and an error.
            if self.check_stop()? {
                break;
            }

            // This is optional - call if you have the ability to append
            // several tokens at once.
            let forced = self.consume_ff_tokens()?;
            tokens.extend_from_slice(&forced);
        }

        Ok(())
    }
}
