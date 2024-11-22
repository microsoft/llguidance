use std::{sync::Arc, time::Duration};

use crate::{
    api::{GenGrammarOptions, ParserLimits, StopReason, TopLevelGrammar},
    earley::{
        grammars_from_json, BiasComputer, CGrammar, CSymIdx, DefaultBiasComputer, Parser,
        ParserError, ParserStats,
    },
    infoln, warn, Logger,
};
use anyhow::{ensure, Result};
use serde_json::json;
use toktrie::{InferenceCapabilities, SimpleVob, StepArg, StepResult, TokEnv, TokenId};

#[derive(Clone)]
pub struct TokenParser {
    pub token_env: TokEnv,
    pub parser: Parser,
    pub mid_process_start_time: instant::Instant,
    pub last_bias_time: Duration,
    pub inference_caps: InferenceCapabilities,
    pub logger: Logger,
    pub limits: ParserLimits,
    pub bias_computer: Arc<dyn BiasComputer>,
    last_step_stats: ParserStats,
    max_step_stats: ParserStats,
    // sampling any of these will pop the parser stack:
    pop_tokens: Option<SimpleVob>,
    test_trace: bool,
    parser_stack: Vec<ParserStackEntry>,
    parser_llm_tokens_offset: usize,
    // this is empty for top-level parser,
    // and the previous grm_bytes for sub-parsers
    previous_grm_bytes: Vec<u8>,

    mid_process_was_accepting: bool,
    stop_reason: StopReason,
    error_message: Option<String>,

    max_tokens_total: usize,
    max_tokens_parser: usize,
    compiled_grammars: Vec<Arc<CGrammar>>,

    // tokens currently in KV cache
    llm_tokens: Vec<TokenId>,
    llm_bytes: Vec<u8>,
    grm_prefix: Vec<u8>,
    is_fresh: bool,
}

#[derive(Clone)]
struct ParserStackEntry {
    parser: Parser,
    parser_llm_tokens_offset: usize,
    previous_grm_bytes_len: usize,
    symidx: CSymIdx,
    max_tokens_offset: usize,
    mask: Option<SimpleVob>,
    is_accepting: bool,
}

impl TokenParser {
    pub fn from_llguidance_json(
        token_env: TokEnv,
        top_grammar: TopLevelGrammar,
        mut logger: Logger,
        inference_caps: InferenceCapabilities,
        limits: ParserLimits,
        extra_lexemes: Vec<String>,
    ) -> Result<Self> {
        let mid_process_start_time = instant::Instant::now();
        let test_trace = top_grammar.test_trace;
        let max_tokens = top_grammar.max_tokens.unwrap_or(usize::MAX);
        let compiled_grammars = grammars_from_json(
            top_grammar,
            &token_env,
            &mut logger,
            limits.clone(),
            extra_lexemes,
        )?;
        let parser = Parser::new(
            Arc::clone(&compiled_grammars[0]),
            GenGrammarOptions::default(),
            limits.clone(),
        )?;

        Ok(TokenParser {
            bias_computer: Arc::new(DefaultBiasComputer::new(token_env.clone())),
            logger,
            test_trace,
            token_env,
            inference_caps,
            limits,
            max_step_stats: ParserStats::default(),
            last_step_stats: ParserStats::default(),
            mid_process_start_time,
            mid_process_was_accepting: false,
            stop_reason: StopReason::NotStopped,
            error_message: None,
            pop_tokens: None,
            parser,
            parser_llm_tokens_offset: 0,
            parser_stack: Vec::new(),
            previous_grm_bytes: Vec::new(),
            compiled_grammars,
            llm_tokens: Vec::new(),
            llm_bytes: Vec::new(),
            grm_prefix: Vec::new(),
            max_tokens_total: max_tokens,
            max_tokens_parser: max_tokens,
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

    pub fn mid_process_was_accepting(&self) -> bool {
        self.mid_process_was_accepting
    }

    pub fn bytes_since(&mut self, mut idx: usize) -> &[u8] {
        idx += self.grm_prefix.len();
        let endp = std::cmp::min(
            self.llm_bytes.len(),
            self.previous_grm_bytes
                .len()
                .saturating_add(self.parser.hidden_start()),
        );
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
            let decoded = self.token_env.tok_trie().decode_raw(&self.llm_tokens);
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

    fn stop(&mut self, warn: &str, reason: StopReason) -> StepResult {
        if warn.len() > 0 {
            self.error_message = Some(warn.to_string());
            warn!(self, "{}; stopping", warn);
        }
        self.stop_reason = reason;
        StepResult::stop()
    }

    pub fn error_message(&self) -> Option<String> {
        self.error_message.clone()
    }

    // advance_parser() is a top-level method in this file.
    // This advance_parser() is called by Constraint::commit_token().
    // It is accessible via the commit_token() method of
    // the LLInterpreter interface.
    //
    // The result here *never* includes a mask.
    // It's either stop or an unconditional splice (possibly noop).
    pub fn advance_parser(&mut self, token: TokenId) -> Result<StepResult> {
        ensure!(self.is_fresh == false, "process_prompt() not called");
        ensure!(self.inference_caps.ff_tokens, "ff_tokens required");
        ensure!(
            self.stop_reason == StopReason::NotStopped,
            "commit_token() on stopped parser"
        );

        self.mid_process_was_accepting = false;

        let tokens = &[token];
        infoln!(
            self,
            "commit_token: {}",
            self.token_env.tok_trie().tokens_dbg(tokens)
        );

        let r = match self.commit_tokens_inner(tokens) {
            Ok(_) => StepResult::noop(),
            Err(r) => r,
        };

        assert!(r.sample_mask.is_none());

        Ok(r)
    }

    // mid_process() is a top-level method in this file.
    // mid_process() is called by Constraint::commit_token().
    // It is also be called by TokenParser::commit_token()
    // within this file, in which case it is accessible
    // via the commit_token() method of the LLInterpreter interface.
    pub fn mid_process(&mut self, arg: StepArg) -> StepResult {
        assert!(self.is_fresh == false, "process_prompt() not called");

        self.mid_process_start_time = instant::Instant::now();
        if self.stop_reason != StopReason::NotStopped {
            let trie = self.token_env.tok_trie();
            infoln!(
                self,
                "stopped; post tokens: bt={} {}",
                arg.backtrack,
                trie.tokens_dbg(&arg.tokens)
            );

            if arg.backtrack == 0
                && (arg.tokens.len() == 0
                    || (arg.tokens.len() == 1 && arg.tokens[0] == trie.eos_token()))
            {
                // Don't warn in this case
                return StepResult::stop();
            }

            warn!(self, "stopped ({})", self.stop_reason.to_string());
            return StepResult::stop();
        }
        if self.max_tokens_total == 0 {
            return self.stop("max_tokens_total reached", StopReason::MaxTokensTotal);
        }
        self.max_tokens_total -= 1;
        self.max_tokens_parser = self.max_tokens_parser.saturating_sub(1);

        let trace = if self.test_trace {
            let tokens = self.token_env.tok_trie().test_trace_tokens(&arg.tokens);
            Some(json!({
                "backtrack": arg.backtrack,
                "tokens": tokens,
            }))
        } else {
            None
        };

        let r = self.mid_process_inner(&arg.tokens).unwrap_err();
        if self.test_trace {
            let res = if r.is_stop() {
                json!("stop")
            } else {
                let b = &r;
                json!({
                    "sample_mask": b.sample_mask.is_some(),
                    "temperature": b.temperature,
                    "splices": b.splices.iter().map(|s| {
                        json!({
                            "when_sampled": s.when_sampled,
                            "backtrack": s.backtrack,
                            "tokens": self.token_env.tok_trie().test_trace_tokens(&s.ff_tokens),
                        })
                    }).collect::<Vec<_>>(),
                })
            };
            self.test_trace_json(&json!({
                "arg": trace.unwrap(),
                "res": res,
            }));
        }

        if !self.inference_caps.ff_tokens && r.has_ff_tokens() {
            // PERF: avoid this for commit token
            let spl = r.unconditional_splice().unwrap();
            assert!(spl.backtrack == 0);
            if spl.ff_tokens.len() == 0 {
                return self.stop(
                    "ff_tokens not allowed, but got empty splice",
                    StopReason::InternalError,
                );
            }

            let t = spl.ff_tokens[0];
            infoln!(self, "forcing ff_token by mask: {}", t);
            let mask = self.token_env.tok_trie().singleton_token_set(t);
            return StepResult::sample(mask, None);
        }

        r
    }

    fn stop_for_parser_error(&mut self, pref: &str, err: ParserError) -> StepResult {
        self.stop(&format!("{}{}", pref, err.message()), err.stop_reason())
    }

    fn maybe_pop_parsers(&mut self, tokens: &[TokenId]) {
        if tokens.len() == 1 {
            let token = tokens[0];
            if let Some(pop) = &self.pop_tokens {
                if pop.is_allowed(token) {
                    let trie = self.token_env.tok_trie();
                    infoln!(self, "pop_tokens hit: {}", trie.token_set_dbg(pop));
                    loop {
                        let pentry = self.parser_stack.last().unwrap();
                        let top_allows_token = pentry.mask.as_ref().unwrap().is_allowed(token);
                        self.pop_parser();
                        if top_allows_token {
                            break;
                        } else {
                            // otherwise, we will keep popping parsers
                        }
                    }
                }
            }
        }
        self.pop_tokens = None;
    }

    fn maybe_scan_eos(&mut self, tokens: &[TokenId]) -> (bool, bool) {
        let mut pending_eos = false;
        let mut clear_tokens = false;
        if tokens.contains(&self.token_env.tok_trie().eos_token()) {
            assert!(tokens.len() == 1);
            if self.parser.scan_eos() {
                // it got scanned correctly, so we remove it
                infoln!(self, "scanned eos_token");
                clear_tokens = true;
            } else {
                infoln!(self, "didn't scan eos_token; saving");
                pending_eos = true;
            }
        }

        (pending_eos, clear_tokens)
    }

    // In this file, when using Result<T, StepResult>, the StepResult is just
    // an early exit, not necessarily an error. It often is an error (when calling self.stop()),
    // but sometimes it is used to return a StepResult::splice() value.
    fn apply_tokens(&mut self, tokens: &[TokenId]) -> Result<usize, StepResult> {
        let trie = self.token_env.tok_trie();

        self.llm_tokens.extend_from_slice(tokens);

        for (tidx, &tok_id) in tokens.iter().enumerate() {
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
                    continue;
                }
            } else {
                tok_bytes
            };

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
                        // we definitely need to backtrack the tokens we didn't get to yet
                        let mut backtrack_tokens = (tokens.len() - tidx) - 1;
                        if backtrack_tokens > 0 {
                            // normally, there should be none
                            warn!(self, "init backtrack_tokens: {}", backtrack_tokens);
                        }
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
        }

        Ok(0)
    }

    fn compute_forced_bytes(&mut self) -> Vec<u8> {
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

    fn maybe_force_tokens(
        &mut self,
        backtrack_tokens: usize,
        forced_bytes: Vec<u8>,
    ) -> Result<Vec<u8>, StepResult> {
        let trie = self.token_env.tok_trie();

        let mut token_prefix = Vec::new();

        let do_force = backtrack_tokens > 0
            || forced_bytes.len() > 0 && !self.parser.grammar().lexer_spec().no_forcing;
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

            if grm_tokens.len() > 0 || backtrack_tokens > 0 {
                infoln!(self, "fixed_tokens: {}", trie.tokens_dbg(&grm_tokens),);
                return Err(StepResult::splice(backtrack_tokens as u32, grm_tokens));
            } else {
                infoln!(self, "no fixed tokens");
            }
        } else if forced_bytes.len() > 0 {
            token_prefix.extend_from_slice(&forced_bytes);
            infoln!(self, "no-forced bytes:{:?}", forced_bytes);
        }

        Ok(token_prefix)
    }

    fn maybe_push_parser(&mut self) -> Result<()> {
        if let Some((msg, symidx, gen_grammar)) = self.parser.maybe_gen_grammar() {
            if msg.len() > 0 {
                warn!(self, "{}", msg);
            }
            let grm = Arc::clone(&self.compiled_grammars[gen_grammar.grammar.to_index().unwrap()]);
            let max_tokens = self.parser.grammar().sym_data(symidx).props.max_tokens;
            let parser = Parser::new(grm, gen_grammar, self.limits.clone())?;
            let mut old_parser = std::mem::replace(&mut self.parser, parser);
            self.parser.take_global_state_from(&mut old_parser);
            let entry = ParserStackEntry {
                parser: old_parser,
                parser_llm_tokens_offset: self.parser_llm_tokens_offset,
                previous_grm_bytes_len: self.previous_grm_bytes.len(),
                symidx,
                max_tokens_offset: self.max_tokens_total.saturating_sub(self.max_tokens_parser),
                mask: None,
                is_accepting: false, // computed with mask
            };
            self.max_tokens_parser = std::cmp::min(self.max_tokens_parser, max_tokens);
            self.parser_llm_tokens_offset = self.llm_tokens.len();
            self.previous_grm_bytes
                .extend_from_slice(&entry.parser.get_bytes());
            self.parser_stack.push(entry);
        }
        Ok(())
    }

    fn pop_parser(&mut self) {
        let inner_bytes = self.parser.get_bytes().to_vec();
        let entry = self.parser_stack.pop().unwrap();
        let mut prev_parser = std::mem::replace(&mut self.parser, entry.parser);
        self.parser.take_global_state_from(&mut prev_parser);
        self.parser_llm_tokens_offset = entry.parser_llm_tokens_offset;
        self.previous_grm_bytes
            .truncate(entry.previous_grm_bytes_len);
        infoln!(
            self,
            "pop_parser: {} tokens left; new {} - {} = {}",
            self.max_tokens_parser,
            self.max_tokens_total,
            entry.max_tokens_offset,
            self.max_tokens_total
                .saturating_sub(entry.max_tokens_offset)
        );
        self.max_tokens_parser = self
            .max_tokens_total
            .saturating_sub(entry.max_tokens_offset);
        self.parser.scan_gen_grammar(entry.symidx, inner_bytes);
    }

    fn maybe_accept_or_pop_parser(
        &mut self,
        pending_eos: bool,
        empty_token_prefix: bool,
    ) -> Result<bool, StepResult> {
        let (inner_done, inner_accepting) = {
            let lexer_bytes = self.parser.has_pending_lexeme_bytes();
            let is_accepting = self.parser.is_accepting();
            let can_advance = self.parser.can_advance();
            let inner_done = empty_token_prefix && is_accepting && (!can_advance || pending_eos);
            infoln!(
                self,
                "inner_done: {inner_done}; lexer_bytes: {lexer_bytes}; \
                can_advance: {can_advance} (eos:{pending_eos}); \
                accept: {is_accepting}; \
                empty_token_prefix: {empty_token_prefix}"
            );
            let inner_accepting = is_accepting && empty_token_prefix;
            (inner_done, inner_accepting)
        };

        if inner_done || self.max_tokens_parser == 0 {
            if self.parser_stack.is_empty() {
                self.mid_process_was_accepting = inner_accepting;
                infoln!(
                    self,
                    "only eos token allowed, stopping; accepting: {}",
                    inner_accepting
                );
                return Err(self.stop(
                    "",
                    if inner_done {
                        if pending_eos {
                            StopReason::EndOfSentence
                        } else {
                            StopReason::NoExtension
                        }
                    } else {
                        StopReason::MaxTokensParser
                    },
                ));
            } else {
                infoln!(self, "pop_parser; tokens left {}", self.max_tokens_parser);
                self.pop_parser();
                let trie = self.token_env.tok_trie();
                // re-start the whole process
                let tokens = if pending_eos {
                    &[trie.eos_token()]
                } else {
                    &[] as &[TokenId]
                };
                self.mid_process_was_accepting = false;
                return Err(self.mid_process_inner(tokens).unwrap_err());
            }
        }

        Ok(inner_accepting)
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

    // this only executes when the top-level parser is accepting
    fn setup_parser_popping(
        &mut self,
        token_prefix: &Vec<u8>,
        allowed_tokens: &mut SimpleVob,
    ) -> Result<(), StepResult> {
        let trie = self.token_env.tok_trie();
        let mut all_accepting = true;
        if self.parser_stack.len() > 0 {
            let mut pop_tokens = trie.alloc_token_set();
            for pentry in self.parser_stack.iter_mut() {
                if pentry.mask.is_none() {
                    assert!(token_prefix.is_empty());
                    let (is_accepting, mask) = pentry
                        .parser
                        .compute_bias_after_gen_grammar(&*self.bias_computer, pentry.symidx);
                    if let Some(err) = pentry.parser.get_error() {
                        return Err(self.stop_for_parser_error("inner parser: ", err));
                    }
                    infoln!(self, "bias for upper parser: {}", trie.token_set_dbg(&mask));
                    pentry.mask = Some(mask);
                    pentry.is_accepting = is_accepting;
                }
                let m = pentry.mask.as_ref().unwrap();
                // pop_tokens |= m & ~allowed_tokens
                pop_tokens.or_minus(m, &allowed_tokens);
                // allowed_tokens |= m
                allowed_tokens.or(m);
                if !pentry.is_accepting {
                    all_accepting = false;
                    break;
                }
            }
            infoln!(self, "pop_tokens: {}", trie.token_set_dbg(&pop_tokens));
            self.pop_tokens = Some(pop_tokens);
        }
        self.mid_process_was_accepting = all_accepting;
        if all_accepting {
            allowed_tokens.allow_token(trie.eos_token());
        }
        Ok(())
    }

    fn log_final(&mut self, token_prefix: &Vec<u8>, allowed_tokens: &mut SimpleVob) {
        infoln!(
            self,
            "step-stats: {}us; {} lex fuel; {} items; {}",
            self.mid_process_start_time.elapsed().as_micros(),
            self.last_step_stats.lexer_cost,
            self.last_step_stats.all_items,
            self.parser.lexer_stats(),
        );

        infoln!(
            self,
            "bias: (pref: {:?}; accpt: {}) {}",
            String::from_utf8_lossy(&token_prefix),
            self.mid_process_was_accepting,
            self.token_env.tok_trie().token_set_dbg(&allowed_tokens)
        );
    }

    fn commit_tokens_inner(&mut self, tokens: &[TokenId]) -> Result<(Vec<u8>, bool), StepResult> {
        self.maybe_pop_parsers(&tokens);

        let (pending_eos, clear_tokens) = self.maybe_scan_eos(tokens);
        let tokens = if clear_tokens {
            &[] as &[TokenId]
        } else {
            tokens
        };

        self.parser.log_row_infos("pre-apply");
        let apply_res = self.apply_tokens(&tokens);
        self.parser.log_row_infos("post-apply");
        let backtrack_tokens = apply_res?;

        self.parser.filter_max_tokens();

        // force after scanning tokens from LLM (this may walk the parser some more)

        let forced_bytes = self.compute_forced_bytes();
        let token_prefix = self.maybe_force_tokens(backtrack_tokens, forced_bytes)?;

        if token_prefix.is_empty() {
            if let Err(e) = self.maybe_push_parser() {
                return Err(self.stop(
                    &format!("Error creating nested parser: {}", e),
                    StopReason::InternalError,
                ));
            }
        }

        let inner_accepting =
            self.maybe_accept_or_pop_parser(pending_eos, token_prefix.is_empty())?;

        Ok((token_prefix, inner_accepting))
    }

    // This never returns Ok(()); we use Result<> as return type to be able to use '?' operator
    fn mid_process_inner(&mut self, tokens: &[TokenId]) -> Result<(), StepResult> {
        self.mid_process_was_accepting = false;

        let trie = self.token_env.tok_trie();
        infoln!(self, "compute_mask: {}", trie.tokens_dbg(tokens));

        let (token_prefix, inner_accepting) = self.commit_tokens_inner(tokens)?;

        let mut allowed_tokens = self.compute_bias(&token_prefix);

        if let Some(err) = self.parser.get_error() {
            return Err(self.stop_for_parser_error("", err));
        }

        if inner_accepting {
            self.setup_parser_popping(&token_prefix, &mut allowed_tokens)?;
        }

        self.log_final(&token_prefix, &mut allowed_tokens);

        if allowed_tokens.num_set() == 0 {
            infoln!(self, "no tokens allowed, stopping");
            return Err(self.stop("", StopReason::NoExtensionBias));
        }

        return Err(StepResult::sample(
            allowed_tokens,
            self.parser.temperature(),
        ));
    }
}
