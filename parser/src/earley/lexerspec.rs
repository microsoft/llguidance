use anyhow::Result;
use derivre::{ExprRef, JsonQuoteOptions, RegexAst, RegexBuilder};
use std::{fmt::Debug, hash::Hash};
use toktrie::{bytes::limit_str, SimpleVob, TokTrie};

use crate::api::ParserLimits;

use super::regexvec::RegexVec;

#[derive(Clone)]
pub struct LexerSpec {
    pub lexemes: Vec<LexemeSpec>,
    pub regex_builder: RegexBuilder,
    pub no_forcing: bool,
    pub allow_initial_skip: bool,
    pub num_extra_lexemes: usize,
    pub skip_by_class: Vec<LexemeIdx>,
    pub current_class: LexemeClass,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct LexemeClass(u8);

impl LexemeClass {
    pub const ROOT: LexemeClass = LexemeClass(0);

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
    pub fn new(class: usize) -> Self {
        LexemeClass(class.try_into().expect("class too large"))
    }
}

#[derive(Clone)]
pub struct LexemeSpec {
    pub(crate) idx: LexemeIdx,
    name: String,
    pub(crate) rx: RegexAst,
    class: LexemeClass,
    compiled_rx: ExprRef,
    ends_at_eos: bool,
    lazy: bool,
    contextual: bool,
    pub(crate) is_skip: bool,
    json_options: Option<JsonQuoteOptions>,
}

/// LexemeIdx is an index into the lexeme table.
/// It corresponds to a category like IDENTIFIER or STRING,
/// or to a very specific lexeme like WHILE or MULTIPLY.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct LexemeIdx(usize);

impl LexemeIdx {
    pub fn new(idx: usize) -> Self {
        LexemeIdx(idx)
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }

    pub fn as_u16(&self) -> u16 {
        self.0 as u16
    }
}

impl LexemeSpec {
    /// Check if the lexeme always matches bytes, and has at least one more byte to spare.
    pub fn has_forced_bytes(&self, bytes: &[u8]) -> bool {
        match &self.rx {
            RegexAst::Literal(s) if s.len() >= bytes.len() => {
                &s.as_bytes()[0..bytes.len()] == bytes
            }
            _ => false,
        }
    }

    pub fn class(&self) -> LexemeClass {
        self.class
    }
}

impl Debug for LexemeSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {} {:?}", self.idx.0, self.name, self.rx)?;
        if self.lazy {
            write!(f, " lazy")?;
        }
        if self.contextual {
            write!(f, " contextual")?;
        }
        Ok(())
    }
}

impl LexerSpec {
    pub fn new() -> Result<Self> {
        Ok(LexerSpec {
            lexemes: Vec::new(),
            regex_builder: RegexBuilder::new(),
            no_forcing: false,
            allow_initial_skip: false,
            num_extra_lexemes: 0,
            skip_by_class: Vec::new(),
            current_class: LexemeClass(0),
        })
    }

    pub fn new_lexeme_class(&mut self, skip: RegexAst) -> Result<LexemeClass> {
        let _ = self.regex_builder.mk(&skip)?; // validate first
        self.current_class = LexemeClass::new(self.skip_by_class.len());
        self.skip_by_class.push(LexemeIdx(0)); // avoid assert in empty_spec()
        let idx = self
            .add_lexeme_spec(LexemeSpec {
                name: format!("SKIP{}", self.current_class.as_usize()),
                rx: skip,
                is_skip: true,
                ..self.empty_spec()
            })
            .expect("already validated");
        self.skip_by_class.pop();
        self.skip_by_class.push(idx);
        Ok(self.current_class)
    }

    pub fn alloc_lexeme_set(&self) -> SimpleVob {
        SimpleVob::alloc(self.lexemes.len())
    }

    pub fn all_lexemes(&self) -> SimpleVob {
        let mut v = self.alloc_lexeme_set();
        self.lexemes[0..self.lexemes.len() - self.num_extra_lexemes]
            .iter()
            .enumerate()
            .for_each(|(idx, _)| v.set(idx, true));
        v
    }

    pub fn lazy_lexemes(&self) -> SimpleVob {
        let mut v = self.alloc_lexeme_set();
        for (idx, lex) in self.lexemes.iter().enumerate() {
            if lex.lazy {
                v.set(idx, true);
            }
        }
        v
    }

    pub fn eos_ending_lexemes(&self) -> SimpleVob {
        let mut v = self.alloc_lexeme_set();
        for (idx, lex) in self.lexemes.iter().enumerate() {
            if lex.ends_at_eos {
                v.set(idx, true);
            }
        }
        v
    }

    pub fn is_nullable(&self, idx: LexemeIdx) -> bool {
        self.regex_builder
            .is_nullable(self.lexemes[idx.0].compiled_rx)
    }

    pub fn to_regex_vec(&self, limits: &mut ParserLimits) -> Result<RegexVec> {
        // TODO
        // Find all non-contextual lexemes that are literals (we call them 'keywords')
        // This assumes that this is the only possible conflict in the lexer that we want to catch.
        // For every non literals lexeme, find all keywords that match it.
        // Replace the regex R for the lexeme with (R & ~(K1|K2|...)) where K1...
        // are the conflicting keywords.
        let rx_list: Vec<_> = self.lexemes.iter().map(|lex| lex.compiled_rx).collect();
        RegexVec::new_with_exprset(
            self.regex_builder.exprset(),
            &rx_list,
            Some(self.lazy_lexemes()),
            limits,
        )
    }

    fn add_lexeme_spec(&mut self, mut spec: LexemeSpec) -> Result<LexemeIdx> {
        let compiled = self.regex_builder.mk(&spec.rx)?;
        let compiled = if let Some(ref opts) = spec.json_options {
            self.regex_builder.json_quote(compiled, opts)?
        } else {
            compiled
        };
        if let Some(idx) = self
            .lexemes
            .iter()
            .position(|lex| lex.compiled_rx == compiled && lex.class == spec.class)
        {
            return Ok(LexemeIdx(idx));
        }
        let idx = LexemeIdx(self.lexemes.len());
        spec.idx = idx;
        spec.compiled_rx = compiled;
        self.lexemes.push(spec);
        Ok(idx)
    }

    fn empty_spec(&self) -> LexemeSpec {
        assert!(
            self.skip_by_class.len() > 0,
            "new_lexeme_class() not called"
        );
        LexemeSpec {
            idx: LexemeIdx(0),
            name: "".to_string(),
            rx: RegexAst::NoMatch,
            compiled_rx: ExprRef::INVALID,
            lazy: false,
            contextual: false,
            ends_at_eos: false,
            is_skip: false,
            json_options: None,
            class: self.current_class,
        }
    }

    pub fn add_rx_and_stop(
        &mut self,
        name: String,
        body_rx: RegexAst,
        stop_rx: RegexAst,
        lazy: bool,
    ) -> Result<LexemeIdx> {
        let rx = if !matches!(stop_rx, RegexAst::EmptyString) {
            RegexAst::Concat(vec![body_rx, RegexAst::LookAhead(Box::new(stop_rx))])
        } else {
            body_rx
        };
        self.add_lexeme_spec(LexemeSpec {
            name,
            rx,
            lazy,
            ends_at_eos: !lazy,
            ..self.empty_spec()
        })
    }

    pub fn add_simple_literal(
        &mut self,
        name: String,
        literal: &str,
        contextual: bool,
    ) -> Result<LexemeIdx> {
        self.add_lexeme_spec(LexemeSpec {
            name,
            rx: RegexAst::Literal(literal.to_string()),
            contextual,
            ..self.empty_spec()
        })
    }

    pub fn add_special_token(&mut self, name: String) -> Result<LexemeIdx> {
        let rx = RegexAst::Concat(vec![
            RegexAst::Byte(TokTrie::SPECIAL_TOKEN_PREFIX_BYTE),
            RegexAst::Literal(name.clone()),
        ]);
        self.add_lexeme_spec(LexemeSpec {
            name,
            rx,
            ..self.empty_spec()
        })
    }

    pub fn add_greedy_lexeme(
        &mut self,
        name: String,
        rx: RegexAst,
        contextual: bool,
        json_options: Option<JsonQuoteOptions>,
    ) -> Result<LexemeIdx> {
        self.add_lexeme_spec(LexemeSpec {
            name,
            rx,
            contextual,
            json_options,
            ..self.empty_spec()
        })
    }

    pub fn add_extra_lexemes(&mut self, extra_lexemes: &Vec<String>) {
        assert!(self.num_extra_lexemes == 0);
        self.num_extra_lexemes = extra_lexemes.len();
        for (idx, added) in extra_lexemes.iter().enumerate() {
            self.add_greedy_lexeme(
                format!("$extra_{}", idx),
                RegexAst::Regex(added.clone()),
                false,
                None,
            )
            .expect("adding lexeme");
        }
    }

    pub fn extra_lexeme(&self, idx: usize) -> LexemeIdx {
        assert!(idx < self.num_extra_lexemes);
        self.lexemes[self.lexemes.len() - self.num_extra_lexemes + idx].idx
    }

    pub fn dbg_lexeme(&self, lex: &Lexeme) -> String {
        let str = String::from_utf8_lossy(&lex.bytes).to_string();
        let info = &self.lexemes[lex.idx.0];
        if matches!(info.rx, RegexAst::Literal(_)) && lex.hidden_len == 0 {
            format!("[{}]", info.name)
        } else {
            format!(
                "[{}] match={:?} hidden={}",
                info.name,
                limit_str(&str, 32),
                lex.hidden_len
            )
        }
    }

    pub fn dbg_lexeme_set(&self, vob: &SimpleVob) -> String {
        format!(
            "Lexemes: [{}]",
            vob.iter()
                .map(|idx| self.lexemes[idx as usize].name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    pub fn lexeme_spec(&self, idx: LexemeIdx) -> &LexemeSpec {
        &self.lexemes[idx.0]
    }

    pub fn cost(&self) -> u64 {
        self.regex_builder.exprset().cost()
    }

    pub fn skip_id(&self, class: LexemeClass) -> LexemeIdx {
        self.skip_by_class[class.as_usize()]
    }
}

impl Debug for LexerSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LexerSpec {{ lexemes: [")?;
        for lex in &self.lexemes {
            writeln!(f, "  {:?}", lex)?;
        }
        write!(f, "] }}")
    }
}

#[derive(Clone)]
pub struct Lexeme {
    pub idx: LexemeIdx,
    bytes: Vec<u8>,
    hidden_len: usize,
}

impl Lexeme {
    pub fn new(idx: LexemeIdx, bytes: Vec<u8>, hidden_len: usize) -> Self {
        Lexeme {
            idx,
            bytes,
            hidden_len,
        }
    }

    pub fn just_idx(idx: LexemeIdx) -> Self {
        Lexeme {
            idx,
            hidden_len: 0,
            bytes: Vec::new(),
        }
    }

    pub fn bogus() -> Self {
        Lexeme::just_idx(LexemeIdx(0))
    }

    pub fn is_bogus(&self) -> bool {
        // TODO?
        self.idx.0 == 0 && self.bytes.is_empty()
    }

    pub fn num_hidden_bytes(&self) -> usize {
        self.hidden_len
    }

    pub fn num_visible_bytes(&self) -> usize {
        self.bytes.len() - self.hidden_len
    }

    pub fn visible_bytes(&self) -> &[u8] {
        &self.bytes[0..self.num_visible_bytes()]
    }

    pub fn hidden_bytes(&self) -> &[u8] {
        &self.bytes[self.num_visible_bytes()..]
    }
}
