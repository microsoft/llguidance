// In this file, "Kallmeyer 2018" refers to the
// slides for "Parsing: Earley parsing", Winter 2017/2018,
// Laura Kallmeyer, Heinrich Heine Universitaet, Dusseldorf,
// https://user.phil-fak.uni-duesseldorf.de/~kallmeyer/Parsing/earley.pdf
// (Retrieved 18 Sep 2024).

use std::{
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
    ops::Range,
    sync::{Arc, Mutex},
    vec,
};

use anyhow::{bail, ensure, Result};
use derivre::{RegexAst, StateID};
use instant::Instant;
use serde::{Deserialize, Serialize};
use toktrie::{Recognizer, SimpleVob, SpecialToken, TokEnv, TokTrie};

use crate::{
    api::{GenGrammarOptions, ParserLimits, StopReason},
    earley::lexer::Lexer,
};

use super::{
    grammar::{CGrammar, CSymIdx, CSymbol, RuleIdx},
    lexer::{LexerResult, PreLexeme},
    lexerspec::{Lexeme, LexemeIdx, LexerSpec},
};

const TRACE: bool = false;
const DEBUG: bool = true;

macro_rules! trace {
    ($($arg:tt)*) => {
        if cfg!(feature = "logging") && TRACE {
            eprintln!($($arg)*);
        }
    }
}

macro_rules! debug {
    ($($arg:tt)*) => {
        if cfg!(feature = "logging") && DEBUG {
            eprintln!($($arg)*);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Item {
    data: u64,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ParserStats {
    pub rows: usize,
    pub definitive_bytes: usize,
    pub lexer_ops: usize,
    pub num_lex_errors: usize,
    pub num_lexemes: usize,
    pub all_items: usize,
    pub lexer_cost: u64,
    pub compute_time_us: u64,
}

impl ParserStats {
    pub fn delta(&self, previous: &ParserStats) -> ParserStats {
        ParserStats {
            rows: self.rows - previous.rows,
            definitive_bytes: self.definitive_bytes - previous.definitive_bytes,
            lexer_ops: self.lexer_ops - previous.lexer_ops,
            num_lexemes: self.num_lexemes - previous.num_lexemes,
            num_lex_errors: self.num_lex_errors - previous.num_lex_errors,
            all_items: self.all_items - previous.all_items,
            lexer_cost: self.lexer_cost - previous.lexer_cost,
            compute_time_us: self.compute_time_us - previous.compute_time_us,
        }
    }

    pub fn max(&self, other: &ParserStats) -> ParserStats {
        ParserStats {
            rows: self.rows.max(other.rows),
            definitive_bytes: self.definitive_bytes.max(other.definitive_bytes),
            lexer_ops: self.lexer_ops.max(other.lexer_ops),
            num_lexemes: self.num_lexemes.max(other.num_lexemes),
            num_lex_errors: self.num_lex_errors.max(other.num_lex_errors),
            all_items: self.all_items.max(other.all_items),
            lexer_cost: self.lexer_cost.max(other.lexer_cost),
            compute_time_us: self.compute_time_us.max(other.compute_time_us),
        }
    }
}

// In this, code a "Row" is what is usually called an Earley set in the literature.
// The term "row" comes from Kallmeyer 2018, which uses a chart parsing algorithm
// in which the rows are Earley sets.
#[derive(Clone)]
struct Row {
    first_item: usize,
    last_item: usize,

    // The "allowed lexemes".  The allowed lexemes (aka acceptable
    // lexemes, aka relevant lexemes) are those which the recognizer
    // will accept in the next row.  They are all and only those lexemes
    // which can lead to a successful parse.
    allowed_lexemes: SimpleVob,
}

impl Row {
    fn item_indices(&self) -> Range<usize> {
        self.first_item..self.last_item
    }
}

// In this code, an "Item" is what is called in the literature, an
// "Earley item".
impl Item {
    #[allow(dead_code)]
    const NULL: Self = Item { data: 0 };

    fn new(rule: RuleIdx, start: usize) -> Self {
        Item {
            data: rule.as_index() as u64 | ((start as u64) << 32),
        }
    }

    fn rule_idx(&self) -> RuleIdx {
        RuleIdx::from_index(self.data as u32)
    }

    fn start_pos(&self) -> usize {
        (self.data >> 32) as usize
    }

    fn advance_dot(&self) -> Self {
        Item {
            data: self.data + 1,
        }
    }
}

// This structure implements the Earley table.
#[derive(Clone)]
struct Scratch {
    grammar: Arc<CGrammar>,

    // The current "working row"
    row_start: usize,
    row_end: usize,

    items: Vec<Item>,

    // Is this Earley table in "definitive" mode?
    // 'definitive' is set when the new lexeme is being 'defined',
    // as indicated by the creation of a 'struct Rowinfo' to track
    // the lexeme.  The opposite of definitive mode is "speculative"
    // mode, which is used for computing the token mask on the
    // pre-lexemes.
    definitive: bool,
}

#[derive(Clone)]
struct RowInfo {
    // TODO: possibly use u32 not usize here
    start_byte_idx: usize,
    lexeme: Lexeme,
    token_idx_start: usize,
    token_idx_stop: usize,

    // A hash whose key is a symbol index, and whose value is
    // the maximum tokens allowed for that symbol.  Undefined hash
    // keys allow an unlimited number of tokens.  That the number
    // of tokens is unlimited can be indicated explicitly by setting
    // the value of the hash entry to usize::MAX.

    // TODO: we are only interested in this for the last (current) row; remove from here?
    max_tokens: Arc<HashMap<LexemeIdx, usize>>,
}

impl RowInfo {
    fn apply_token_idx(&mut self, idx: usize) {
        self.token_idx_start = self.token_idx_start.min(idx);
        self.token_idx_stop = self.token_idx_stop.max(idx);
    }

    fn set_token_idx(&mut self, idx: usize) {
        self.token_idx_start = idx;
        self.token_idx_stop = idx;
    }

    fn dbg(&self, lexspec: &LexerSpec) -> String {
        format!(
            "token_idx: {}-{}; b:{}; {} {}",
            self.token_idx_start,
            self.token_idx_stop,
            self.start_byte_idx,
            lexspec.dbg_lexeme(&self.lexeme),
            if self.max_tokens.is_empty() {
                "".to_string()
            } else {
                format!("max_tokens={:?}", self.max_tokens)
            }
        )
    }
}

// State transition is:
// if (next_lexeme, next_lexer_state) := lexer(top.lexer_state, next_byte) {
//     row_idx = scan(top.row_idx, next_lexeme)
//     push(LexerState { row_idx, next_byte, next_lexer_state })
// }
//
// The LLM thinks in tokens, while the parser only deals with lexemes.
// There is no easy translation between these, and the parser cannot work
// with tokens. On the other hand, forcing the LLM to deal with lexemes will increase
// token perplexity and degrade the quality of the LLM's output.
//
// The data structure used to resolve this "impedance mismatch" is a stack of 'LexerState' items.
// Tokens are broken down into single bytes when they go into this stack,
// and the bytes are assembled into lexemes by the lexer.
// The 'LexerState' items are left on the stack (unless backtracking).
//
// The stack of lexer states also manages a virtual stack of Earley sets, via the
// 'row_idx' field.  The current Earley table/stack is rows 0 through 'row_idx'.
#[derive(Clone, Copy)]
struct LexerState {
    row_idx: u32,         // Index of corresponding row (Earley set)
    lexer_state: StateID, // state after consuming 'byte'
    byte: Option<u8>,
}

#[derive(Clone)]
struct ParserState {
    grammar: Arc<CGrammar>,
    scratch: Scratch,
    trie_lexer_stack: usize,
    captures: Vec<(String, Vec<u8>)>,

    // These are updated also in speculative mode.
    // Both are stacks only in the sense that items can be popped on backtracking
    // (when walking the token trie). Otherwise, they represent the full parsing
    // history - items are not popped in definitive mode.
    lexer_stack: Vec<LexerState>,
    rows: Vec<Row>,

    // These are only updated in definitive mode.
    row_infos: Vec<RowInfo>,
    token_idx: usize,
    bytes: Vec<u8>,
    // use u32 to save space
    byte_to_token_idx: Vec<u32>,

    first_row_with_max_token_limit: usize,
    stats: ParserStats,
    options: GenGrammarOptions,
    trie_gen_grammar: Option<CSymIdx>,
    trie_gen_grammar_accepting: bool,
    limits: ParserLimits,
    max_all_items: usize,
    parser_error: Option<String>,
    backtrack_byte_count: usize,
}

#[derive(Clone)]
struct SharedState {
    lexer: Lexer,
}

#[derive(Clone)]
pub struct Parser {
    shared: Arc<Mutex<SharedState>>,
    state: ParserState,
}

impl Scratch {
    fn new(grammar: Arc<CGrammar>) -> Self {
        Scratch {
            grammar,
            row_start: 0,
            row_end: 0,
            items: vec![],
            definitive: true,
        }
    }

    // Set current working Earley to empty set
    // The set backing data is at `pos`
    fn new_row(&mut self, pos: usize) {
        self.row_start = pos;
        self.row_end = pos;
    }

    // Number of items in the current working Earley set
    fn row_len(&self) -> usize {
        self.row_end - self.row_start
    }

    // Add a new row to the Earley table.  It will be the
    // current, working, row.
    fn work_row(&self, allowed_lexemes: SimpleVob) -> Row {
        Row {
            first_item: self.row_start,
            last_item: self.row_end,
            allowed_lexemes,
        }
    }

    // Make sure there is enough space in the Earley table,
    // usually in preparation for adding Earley items.
    #[inline(always)]
    fn ensure_items(&mut self, n: usize) {
        if self.items.len() < n {
            let missing = n - self.items.len();
            self.items.reserve(missing);
            unsafe { self.items.set_len(n) }
        }
    }

    // Add a new Earley item with default values to the Earley table.  It is
    // "just" added in the sense that no checks are performed, except the one
    // that ensures there is enough space in the table.  The other checks are
    // assumed to be unnecessary or to have been performed.  For example, it
    // is assumed the caller knows that this Earley item will be unique.
    #[inline(always)]
    fn just_add(&mut self, item: Item, _origin_item_idx: usize, info: &str) {
        self.ensure_items(self.row_end + 1);
        // SAFETY: we just ensured that there is enough space
        unsafe {
            self.items.as_mut_ptr().add(self.row_end).write(item);
        }
        // self.items[self.row_end] = item;
        if self.definitive {
            debug!(
                "      addu: {} ({})",
                self.item_to_string(self.row_end),
                info
            );
        }
        self.row_end += 1;
    }

    // Find 'item' in the current, working, row.
    #[inline(always)]
    fn find_item(&self, item: Item) -> Option<usize> {
        self.items[self.row_start..self.row_end]
            .iter()
            .position(|&x| x == item)
            .map(|x| x + self.row_start)
    }

    // Ensure that Earley table 'self' contains
    // Earley item 'item'.  That is, look for 'item' in 'self',
    // and add 'item' to 'self' if it is not there already.
    #[inline(always)]
    fn add_unique(&mut self, item: Item, origin_item_idx: usize, info: &str) {
        if self.find_item(item).is_none() {
            self.just_add(item, origin_item_idx, info);
        }
    }

    // Write item at index 'idx' as a string.
    fn item_to_string(&self, idx: usize) -> String {
        item_to_string(&self.grammar, &self.items[idx])
    }
}

macro_rules! ensure_internal {
    ($cond:expr, $msg:expr) => {
        ensure!($cond, "Internal error: {}", $msg)
    };
}

impl ParserState {
    // Create a new state for an empty parser.
    // The parser starts in definitive mode.
    fn new(
        grammar: Arc<CGrammar>,
        options: GenGrammarOptions,
        mut limits: ParserLimits,
    ) -> Result<(Self, Lexer)> {
        let start = grammar.start();
        let mut lexer = Lexer::from(grammar.lexer_spec(), &mut limits)?;
        let scratch = Scratch::new(Arc::clone(&grammar));
        let lexer_state = lexer.a_dead_state(); // placeholder
        let mut r = ParserState {
            grammar,
            trie_lexer_stack: usize::MAX,
            rows: vec![],
            row_infos: vec![],
            captures: vec![],
            scratch,
            stats: ParserStats::default(),
            token_idx: 0,
            byte_to_token_idx: vec![],
            bytes: vec![],
            max_all_items: usize::MAX,
            first_row_with_max_token_limit: 0,
            options,
            trie_gen_grammar: None,
            trie_gen_grammar_accepting: false,
            limits,
            backtrack_byte_count: 0,
            lexer_stack: vec![LexerState {
                row_idx: 0,
                lexer_state,
                byte: None,
            }],
            parser_error: None,
        };

        // Initialize the Earley table with the predictions in
        // row 0.
        for rule in r.grammar.rules_of(start).to_vec() {
            r.scratch.add_unique(Item::new(rule, 0), 0, "init");
        }
        debug!("initial push");
        let _ = r.push_row(0, r.scratch.row_start, &Lexeme::bogus());
        ensure_internal!(
            r.num_rows() == 1 && r.rows.len() == 1,
            "initial push failed"
        );
        assert!(r.lexer_stack.len() == 1);

        // Set the correct initial lexer state

        if !r.grammar.lexer_spec().allow_initial_skip {
            // Disallow initial SKIP if asked to.
            // This is done, for example, we are trying to force
            // the generation of JSON to start.
            r.rows[0]
                .allowed_lexemes
                .set(LexemeIdx::SKIP.as_usize(), false);
        }

        let state = lexer.start_state(&r.rows[0].allowed_lexemes, None);
        r.lexer_stack[0].lexer_state = state;
        r.assert_definitive();

        r.stats.lexer_cost = lexer.dfa.total_fuel_spent();

        Ok((r, lexer))
    }

    fn compute_bias(
        &mut self,
        shared: &mut SharedState,
        computer: &dyn BiasComputer,
        start: &[u8],
    ) -> SimpleVob {
        let t0 = Instant::now();
        let dfa = &mut shared.lexer.dfa;
        dfa.set_fuel(self.limits.step_lexer_fuel);
        dfa.set_max_states(self.limits.max_lexer_states);

        self.max_all_items = self.stats.all_items + self.limits.step_max_items as usize;

        let mut r = ParserRecognizer {
            shared,
            state: self,
        };
        let mut set = computer.compute_bias(&mut r, start);

        if self.stats.all_items > self.max_all_items && self.parser_error.is_none() {
            self.parser_error = Some(format!(
                "Too many items (limit {}); try avoiding single-byte/short lexemes",
                self.limits.step_max_items
            ));
        }

        self.max_all_items = usize::MAX;

        self.stats.lexer_cost = shared.lexer.dfa.total_fuel_spent();

        // The SPECIAL_TOKEN_PREFIX_BYTE should never be allowed by itself
        let toks = computer
            .trie()
            .greedy_tokenize(&[TokTrie::SPECIAL_TOKEN_PREFIX_BYTE]);
        assert!(toks.len() == 1);
        set.disallow_token(toks[0]);

        computer.trie().apply_duplicates(&mut set);

        if set.is_zero() {
            // nothing allowed
            // we're going to be stopped outside - we better flush the lexer
            let _ = self.flush_lexer(shared);
        }

        if start.is_empty() && self.lexer_allows_eos(shared) {
            set.allow_token(computer.trie().eos_token());
        }

        self.stats.compute_time_us += t0.elapsed().as_micros() as u64;

        set
    }

    fn after_dots(&self) -> impl Iterator<Item = RuleIdx> + '_ {
        self.curr_row()
            .item_indices()
            .map(|i| self.scratch.items[i].rule_idx())
    }

    fn after_dots_symdata(&self) -> impl Iterator<Item = &CSymbol> + '_ {
        self.after_dots().map(|pos| self.grammar.sym_data_dot(pos))
    }

    fn can_advance_inner(&self) -> bool {
        for data in self.after_dots_symdata() {
            if data.lexeme == Some(LexemeIdx::SKIP) || data.idx == CSymIdx::NULL {
                continue;
            }
            if data.is_terminal || data.gen_grammar.is_some() {
                return true;
            }
        }
        false
    }

    pub fn can_advance(&self) -> bool {
        self.has_pending_lexeme_bytes() || self.can_advance_inner()
    }

    pub fn has_pending_lexeme_bytes(&self) -> bool {
        self.curr_row_bytes().len() > 0
    }

    // Does the parse succeed in this Earley set?
    // That is, does this Earley set contain a completed
    // start rule?
    fn row_is_accepting(&self) -> bool {
        for pos in self.after_dots() {
            let after_dot = self.grammar.sym_idx_dot(pos);
            if after_dot == CSymIdx::NULL {
                let lhs = self.grammar.sym_idx_lhs(pos);
                if lhs == self.grammar.start() {
                    return true;
                }
            }
        }
        false
    }

    pub fn lexer_allows_eos(&mut self, shared: &mut SharedState) -> bool {
        if self.has_pending_lexeme_bytes() {
            shared.lexer.allows_eos(self.lexer_state().lexer_state)
        } else {
            // empty lexemes are not allowed
            false
        }
    }

    fn item_to_string(&self, idx: usize) -> String {
        self.scratch.item_to_string(idx)
    }

    #[allow(dead_code)]
    pub fn print_row(&self, row_idx: usize) {
        let row = &self.rows[row_idx];
        println!(
            "row {}; lexer_stack={} top_state={:?}",
            row_idx,
            self.lexer_stack.len(),
            self.lexer_stack.last().unwrap().lexer_state
        );

        println!(
            "  allowed: {}",
            self.lexer_spec().dbg_lexeme_set(&row.allowed_lexemes)
        );

        if row_idx < self.row_infos.len() {
            let info = &self.row_infos[row_idx];
            if info.lexeme.is_bogus() {
                println!("  lexeme: placeholder");
            } else {
                println!("  lexeme: {}", self.lexer_spec().dbg_lexeme(&info.lexeme));
            }
        } else {
            println!("  speculative");
        }
        for i in row.item_indices() {
            println!("  {}", self.item_to_string(i));
        }
    }

    #[inline(always)]
    fn lexer_state(&self) -> LexerState {
        self.lexer_stack[self.lexer_stack.len() - 1]
    }

    /// Current size of the Earley table -- that is,
    /// the number of Earley sets.
    #[inline(always)]
    pub fn num_rows(&self) -> usize {
        // The number of rows is taken, not from the physical Earley table,
        // but from the virtual Earley stack kept in the lexer state.
        self.lexer_state().row_idx as usize + 1
    }

    #[inline(always)]
    fn pop_lexer_states(&mut self, n: usize) {
        assert!(self.lexer_stack.len() > n);
        unsafe { self.lexer_stack.set_len(self.lexer_stack.len() - n) }
    }

    #[allow(dead_code)]
    pub fn print_stats(&mut self) {
        println!("stats: {:?}", self.stats);
        self.stats = ParserStats::default();
    }

    fn assert_definitive(&self) {
        assert!(self.scratch.definitive);
        assert!(self.backtrack_byte_count == 0);
        if self.num_rows() != self.row_infos.len() {
            panic!(
                "num_rows={} row_infos={}",
                self.num_rows(),
                self.row_infos.len()
            );
        }
    }

    pub fn get_bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn item_lhs(&self, item: &Item) -> CSymIdx {
        self.grammar.sym_idx_lhs(item.rule_idx())
    }

    fn item_sym_data(&self, item: &Item) -> &CSymbol {
        self.grammar.sym_data(self.item_lhs(item))
    }

    fn hidden_start(&self, shared: &mut SharedState) -> usize {
        let hidden_len = shared
            .lexer
            .possible_hidden_len(self.lexer_state().lexer_state);
        if hidden_len == 0 {
            return usize::MAX;
        }
        let last_lexeme_visible_len = self.curr_row_bytes().len() - hidden_len;
        let prefix_len = self.row_infos[self.num_rows() - 1].start_byte_idx;
        prefix_len + last_lexeme_visible_len
    }

    pub fn temperature(&self) -> Option<f32> {
        let mut temp = -1000.0f32;
        for data in self.after_dots_symdata() {
            if data.is_terminal {
                temp = temp.max(data.props.temperature);
            }
        }
        if let Some(t) = self.options.temperature {
            temp = temp.max(t);
        }
        if temp < 0.00000001 {
            None
        } else {
            Some(temp)
        }
    }

    // apply_tokens() "pushes" the bytes in 'tokens' into the lexer and parser.  It is a top-level
    // method in this file.  It is well below llguidance's top-level methods, but in the llguidance
    // LLInterpreter interface, it is called indirectly via the advance_parser() method.
    pub fn apply_token(&mut self, shared: &mut SharedState, tok_bytes: &[u8]) -> Result<usize> {
        self.assert_definitive();

        let mut check_lexer_max_tokens = false;

        let mut row_to_apply = self.num_rows() - 1;

        // find first row to apply new token idx
        let applied_idx0 = self.byte_to_token_idx.len();
        while row_to_apply > 0 {
            if self.row_infos[row_to_apply].start_byte_idx <= applied_idx0 {
                break;
            }
            row_to_apply -= 1;
        }

        for (bidx, &b) in tok_bytes.iter().enumerate() {
            check_lexer_max_tokens = false;
            let applied_idx = self.byte_to_token_idx.len();
            if applied_idx >= self.bytes.len() {
                assert!(applied_idx == self.bytes.len());

                let row_idx = self.num_rows() - 1;
                self.row_infos[row_idx].apply_token_idx(self.token_idx);

                let (ok, bt) = self.try_push_byte_definitive(shared, Some(b));
                if !ok {
                    bail!(
                        "byte {:?} fails parse; applying {:?}",
                        b as char,
                        String::from_utf8_lossy(tok_bytes)
                    );
                }
                if bt > 0 {
                    self.byte_to_token_idx.truncate(self.bytes.len());
                    let bt = bt + (tok_bytes.len() - bidx - 1);
                    return Ok(bt);
                }
                if row_idx == self.num_rows() - 1 {
                    // if we didn't push a new row, and are at the end of the current token,
                    // check on max_tokens
                    check_lexer_max_tokens = true;
                }
            } else {
                if self.bytes[applied_idx] != b {
                    bail!(
                        "expecting {:?} (forced bytes), got {:?}; applying {:?}",
                        self.bytes[applied_idx] as char,
                        b as char,
                        String::from_utf8_lossy(tok_bytes)
                    );
                }
            }

            self.byte_to_token_idx
                .push(self.token_idx.try_into().unwrap());
        }

        for idx in row_to_apply..self.num_rows() {
            // for all rows fully contained (so far) in the new token, reset token idx
            if self.row_infos[idx].start_byte_idx >= applied_idx0 {
                self.row_infos[idx].set_token_idx(self.token_idx);
            } else {
                // otherwise, just apply it
                self.row_infos[idx].apply_token_idx(self.token_idx);
            }
        }

        if check_lexer_max_tokens {
            let row_idx = self.num_rows() - 1;
            let info = &self.row_infos[row_idx];
            let info_tokens = std::cmp::max(
                0,
                self.token_idx as isize + 1 - info.token_idx_start as isize,
            ) as usize;
            let lex_state = self.lexer_state().lexer_state;
            let mut limit = self.lexer_spec().alloc_lexeme_set();
            let mut num_limit = 0;
            {
                let possible_lexemes = shared.lexer.possible_lexemes(lex_state);
                for idx in possible_lexemes.iter() {
                    let lex = LexemeIdx::new(idx as usize);
                    let max_tokens = *info.max_tokens.get(&lex).unwrap_or(&usize::MAX);
                    debug!(
                        "  max_tokens: {} max={} info={}",
                        self.lexer_spec().dbg_lexeme(&Lexeme::just_idx(lex)),
                        max_tokens,
                        info_tokens
                    );
                    if info_tokens < max_tokens {
                        limit.allow_token(idx);
                    } else {
                        num_limit += 1;
                    }
                }
            }
            if num_limit > 0 {
                debug!(
                    "  max_tokens limiting to: {}",
                    self.lexer_spec().dbg_lexeme_set(&limit)
                );
                let new_state = shared.lexer.limit_state_to(lex_state, &limit);
                if new_state.is_dead() {
                    debug!("  limited everything; forcing EOI");
                    let (ok, bt) = self.try_push_byte_definitive(shared, None);
                    assert!(bt == 0);
                    if !ok {
                        debug!("parse reject on max_tokens");
                        return Ok(0);
                    }
                } else {
                    self.lexer_stack.last_mut().unwrap().lexer_state = new_state;
                }
            }
        }

        let item_count = self.curr_row().item_indices().count();
        if item_count > self.limits.max_items_in_row {
            bail!(
                "Current row has {} items; max is {}; consider making your grammar left-recursive if it's right-recursive",
                item_count,
                self.limits.max_items_in_row,
            );
        }

        // self.print_row(self.num_rows() - 1);

        return Ok(0);
    }

    pub fn filter_max_tokens(&mut self) {
        self.assert_definitive();

        let start_idx = self.first_row_with_max_token_limit;

        if start_idx >= self.num_rows() {
            return;
        }

        self.first_row_with_max_token_limit = self.num_rows();

        let token_idx = self.token_idx;

        self.row_infos.push(RowInfo {
            lexeme: Lexeme::bogus(),
            start_byte_idx: 0,
            token_idx_start: token_idx,
            token_idx_stop: token_idx,
            max_tokens: Arc::new(HashMap::default()),
        });

        let mut dst = self.rows[start_idx].first_item;
        for idx in start_idx..self.num_rows() {
            let range = self.rows[idx].item_indices();
            self.rows[idx].first_item = dst;
            let mut has_max_tokens = false;
            for i in range {
                let item = self.scratch.items[i];
                let sym_data = self.item_sym_data(&item);
                let max_tokens = sym_data.props.max_tokens;
                if max_tokens != usize::MAX {
                    let start_token_idx = self.row_infos[item.start_pos() + 1].token_idx_start;
                    if token_idx - start_token_idx >= max_tokens {
                        debug!(
                            "  remove: {}-{} {}",
                            token_idx,
                            start_token_idx,
                            self.item_to_string(i)
                        );
                        continue;
                    }
                    has_max_tokens = true;
                }
                self.scratch.items[dst] = item;
                dst += 1;
            }
            self.rows[idx].last_item = dst;
            if has_max_tokens {
                self.first_row_with_max_token_limit =
                    std::cmp::min(self.first_row_with_max_token_limit, idx);
            }
        }

        self.row_infos.pop();
    }

    /// force_bytes() forces bytes into the parser, definitively.
    /// They must be, at each point, the only bytes allowed by
    /// the parser.  force_bytes() returns a 'Vec' of the bytes pushed.
    pub fn force_bytes(&mut self, shared: &mut SharedState) -> &[u8] {
        self.assert_definitive();
        trace!("force_bytes lexer_stack {}", self.lexer_stack.len());
        while let Some(b) = self.forced_byte(shared) {
            debug!("  forced: {:?} 0x{:x}", b as char, b);
            let (ok, bt) = self.try_push_byte_definitive(shared, Some(b));
            assert!(bt == 0);
            if !ok {
                // shouldn't happen?
                debug!("  force_bytes reject {}", b as char);
                break;
            }
        }
        self.assert_definitive();
        let bytes = &self.bytes[self.byte_to_token_idx.len()..];
        trace!(
            "force_bytes exit {} lexer_stack={}",
            bytes.len(),
            self.lexer_stack.len()
        );
        bytes
    }

    // Advance the parser or the lexer, depending on whether 'lex_result'
    // is a pre-lexeme or not.
    #[inline(always)]
    fn advance_lexer_or_parser(
        &mut self,
        shared: &mut SharedState,
        lex_result: LexerResult,
        curr: LexerState,
    ) -> bool {
        match lex_result {
            LexerResult::State(next_state, byte) => {
                // lexer advanced, but no lexeme - fast path
                self.lexer_stack.push(LexerState {
                    row_idx: curr.row_idx,
                    lexer_state: next_state,
                    byte: Some(byte),
                });
                true
            }
            LexerResult::Error => false,
            LexerResult::Lexeme(pre_lexeme) => self.advance_parser(shared, pre_lexeme),
        }
    }

    fn trie_started_inner(&mut self) {
        // debug!("trie_started: rows={} lexer={}", self.num_rows(), self.lexer_stack.len());
        self.assert_definitive();
        self.trie_lexer_stack = self.lexer_stack.len();
        self.scratch.definitive = false;
    }

    fn trie_finished_inner(&mut self) {
        // debug!("trie_finished: rows={} lexer={}", self.num_rows(), self.lexer_stack.len());
        assert!(self.scratch.definitive == false);
        assert!(self.row_infos.len() <= self.num_rows());
        // clean up stack
        self.pop_lexer_states(self.lexer_stack.len() - self.trie_lexer_stack);
        self.scratch.definitive = true;
        self.assert_definitive();
    }

    fn run_speculative<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.trie_started_inner();
        let r = f(self);
        self.trie_finished_inner();
        r
    }

    pub fn is_accepting(&mut self, shared: &mut SharedState) -> bool {
        self.run_speculative(|s| s.flush_lexer(shared) && s.row_is_accepting())
    }

    // try_push_byte_definitive() attempts to 'push' a byte (that is advance
    // the parse with 'byte') into the parse in definitive mode.
    // Returns 'false' if this is not possible.
    fn try_push_byte_definitive(
        &mut self,
        shared: &mut SharedState,
        byte: Option<u8>,
    ) -> (bool, usize) {
        assert!(self.scratch.definitive);

        let curr = self.lexer_state();
        let row = &self.rows[curr.row_idx as usize];

        let res = if byte.is_none() {
            let lexeme = shared.lexer.force_lexeme_end(curr.lexer_state);
            if lexeme.is_error() {
                debug!(
                    "    lexer fail on forced end; allowed: {}",
                    self.lexer_spec().dbg_lexeme_set(&row.allowed_lexemes)
                );
            }
            lexeme
        } else {
            self.stats.definitive_bytes += 1;
            shared
                .lexer
                .advance(curr.lexer_state, byte.unwrap(), self.scratch.definitive)
        };

        if res.is_error() {
            debug!(
                "  lexer fail; allowed: {}",
                self.lexer_spec().dbg_lexeme_set(&row.allowed_lexemes)
            );
        }

        assert!(self.backtrack_byte_count == 0);
        if self.advance_lexer_or_parser(shared, res, curr) {
            if let Some(b) = byte {
                self.bytes.push(b);
            }
            let bt = std::mem::take(&mut self.backtrack_byte_count);
            if bt > 0 {
                self.bytes.truncate(self.bytes.len() - bt);
            }
            (true, bt)
        } else {
            (false, 0)
        }
    }

    /// The current Earley set (row) as kept track of
    /// in the lexer stack.
    fn curr_row(&self) -> &Row {
        &self.rows[self.lexer_state().row_idx as usize]
    }

    /// forced_byte() finds the unique byte allowed by the
    /// parser at this point, and returns it.  If there is
    /// no such byte, forced_byte() returns 'None'.
    fn forced_byte(&mut self, shared: &mut SharedState) -> Option<u8> {
        if self.is_accepting(shared) {
            debug!("  in accept state, not forcing");
            return None;
        }

        // self.print_row(self.num_rows() - 1);

        // TODO: use RegexVec::next_byte() here if possible ?
        // currently, this will likely take a few thousand cycles to produce a byte

        self.run_speculative(|s| {
            let mut r = ParserRecognizer { shared, state: s };
            let mut byte_sym = None;
            for b in 0..=255 {
                if r.try_push_byte(b) {
                    r.pop_bytes(1);
                    // debug!("  forced: {:?}", b as char);
                    if byte_sym.is_some() {
                        // debug!("  forced multiple");
                        return None; // more than one option
                    } else {
                        byte_sym = Some(b);
                    }
                }
            }
            byte_sym
        })
    }

    /// Advance the parser as if the current lexeme (if any)
    /// finished right here.
    /// Returns true if the parser was able to advance (or there were no pending bytes for a lexeme).
    fn flush_lexer(&mut self, shared: &mut SharedState) -> bool {
        if !self.has_pending_lexeme_bytes() {
            return true;
        }
        let curr = self.lexer_state();
        let lex_result = shared.lexer.try_lexeme_end(curr.lexer_state);
        let r = self.advance_lexer_or_parser(shared, lex_result, curr);
        assert!(self.backtrack_byte_count == 0);
        r
    }

    pub fn maybe_gen_grammar(&mut self) -> Option<(String, CSymIdx, GenGrammarOptions)> {
        self.assert_definitive();
        let mut res: Option<GenGrammarOptions> = None;
        let mut res_idx = None;
        let mut gen_grm = vec![];
        for pos in self.after_dots() {
            let idx = self.grammar.sym_idx_dot(pos);
            let sym_data = self.grammar.sym_data_dot(pos);
            if let Some(ref gg) = sym_data.gen_grammar {
                // break ties by preferring the one with the lowest grammar number
                if res.is_none()
                    || res.as_ref().unwrap().grammar.to_index().unwrap()
                        > gg.grammar.to_index().unwrap()
                {
                    res = Some(gg.clone());
                    res_idx = Some(idx);
                }
                gen_grm.push(idx);
            } else if sym_data.is_terminal {
                gen_grm.push(idx);
            }
        }

        if res.is_none() {
            return None;
        }

        let msg = if gen_grm.len() > 1 {
            format!(
                "ambiguity between GenGrammar and terminals {:?}",
                gen_grm
                    .iter()
                    .map(|&x| self.grammar.sym_name(x))
                    .collect::<Vec<_>>()
            )
        } else {
            String::new()
        };

        Some((msg, res_idx.unwrap(), res.unwrap()))
    }

    fn flush_gen_grammar(&mut self, shared: &mut SharedState) {
        if let Some(idx) = self.trie_gen_grammar.take() {
            let r = self.scan_gen_grammar_inner(shared, idx, vec![]);
            self.trie_gen_grammar_accepting = r && self.row_is_accepting();
        }
    }

    fn scan_gen_grammar_inner(
        &mut self,
        shared: &mut SharedState,
        symidx: CSymIdx,
        inner_bytes: Vec<u8>,
    ) -> bool {
        debug!("  scan gen_grammar: {}", self.grammar.sym_name(symidx));

        self.scratch.new_row(self.curr_row().last_item);

        for idx in self.curr_row().item_indices() {
            let item = self.scratch.items[idx];
            let sidx = self.grammar.sym_idx_dot(item.rule_idx());
            if sidx == symidx {
                self.scratch
                    .add_unique(item.advance_dot(), idx, "gen_grammar");
            }
        }

        assert!(self.scratch.row_len() > 0);

        let lexeme = Lexeme::new(LexemeIdx::SKIP, inner_bytes, 0);

        let r = self.push_row(self.num_rows(), self.scratch.row_start, &lexeme);
        if r {
            debug!("  gen_grammar OK");
            let lexer_state = self.lexer_state_for_added_row(shared, lexeme, None);
            self.lexer_stack.push(lexer_state);
            true
        } else {
            debug!("  gen_grammar failed!");
            false
        }
    }

    pub fn scan_eos(&mut self, shared: &mut SharedState) -> bool {
        self.assert_definitive(); // ???

        let lexer_eos = self.lexer_allows_eos(shared);

        debug!("  scan eos: lexer_eos={}", lexer_eos);

        if !self.flush_lexer(shared) {
            debug!("  flush_lexer() failed");
            return false;
        }

        debug!("  flush_lexer() OK");

        if lexer_eos {
            return true;
        }
        // This is really for EOS tokens in the middle of the grammar
        // that need to be eaten; so don't check for accepting state here
        // if self.is_accepting() {
        //     return true;
        // }

        return false;
    }

    // this just copies current row
    fn scan_skip_lexeme(&mut self, lexeme: &Lexeme) -> bool {
        let src = self.curr_row().item_indices();
        let allowed_lexemes = self.curr_row().allowed_lexemes.clone();
        let n = src.len();
        if n == 0 {
            return false;
        }
        self.scratch.ensure_items(src.end + n + 100);
        self.scratch.new_row(src.end);

        for i in src {
            self.scratch
                .just_add(self.scratch.items[i], i, "skip_lexeme");
        }

        // note that we pass 'row_end' not 'row_start' as the agenda pointer
        // this will skip processing any items, and only push the row
        let push_res = self.push_row(self.num_rows(), self.scratch.row_end, lexeme);
        assert!(push_res);
        let added_row_idx = self.num_rows();
        // the allowed_lexemes were not computed correctly due to us messing
        // with agenda pointer above
        self.rows[added_row_idx].allowed_lexemes = allowed_lexemes;
        if self.scratch.definitive {
            self.row_infos[added_row_idx].max_tokens =
                self.row_infos[added_row_idx - 1].max_tokens.clone();
        }
        true
    }

    // scan() implements the version of Earley described in Kallmeyer 2018.
    // An important difference between the algorithm implemented here
    // and Kallmeyer's is that in scan(), the token scan is performed
    // first, while in Kallmeyer it is performed last.

    // Returns false if the parse is exhausted, true otherwise.

    // lexeme body only used for captures (in definitive mode)
    // and debugging (lexeme.idx used always)
    fn scan(&mut self, lexeme: &Lexeme) -> bool {
        let row_idx = self.num_rows() - 1;
        let last = self.rows[row_idx].last_item;
        let mut i = self.rows[row_idx].first_item;
        let n = last - i;
        self.scratch.ensure_items(last + n + 100);
        self.scratch.new_row(last);

        if self.scratch.definitive {
            debug!(
                "  scan: {} at {} (spec: {:?})",
                self.lexer_spec().dbg_lexeme(&lexeme),
                row_idx,
                self.lexer_spec().lexeme_spec(lexeme.idx),
            );
        }

        // This loop performs the scan inference rule
        // (slide 21 of Kallmeyer 2018).  It is an
        // initialization inference rule, performed "just
        // in time" at the beginning of the creation of
        // each row
        while i < last {
            let item = self.scratch.items[i];
            let sym = self.grammar.sym_data_dot(item.rule_idx());
            if sym.lexeme == Some(lexeme.idx) {
                self.scratch.just_add(item.advance_dot(), i, "scan");
            }
            i += 1;
        }

        // Perform the other inference rules on this Earley set.
        self.push_row(self.num_rows(), self.scratch.row_start, lexeme)
    }

    // push_row() does the agenda processing.  There is an agenda for
    // each Earley set (aka row).

    // Returns false if an empty Earley set is added (and therefore
    // the parse is exhausted); and true otherwise.

    // lexeme only used for captures (in definitive mode)
    #[inline(always)]
    fn push_row(&mut self, curr_idx: usize, mut agenda_ptr: usize, lexeme: &Lexeme) -> bool {
        let mut allowed_lexemes = self.lexer_spec().alloc_lexeme_set();
        let mut max_tokens = vec![];

        // Agenda retrieval is a simplification of Kallmeyer 2018.
        // There is no separate data structure for the agenda --
        // the Earley table is used, so that adding to the Earley
        // table (aka chart) also adds an item to the agenda.  No duplicate
        // agenda items are added.  Agenda items are never removed --
        // instead 'agenda_ptr' is advanced through the combined agenda/chart.
        // Only one pass is made.
        while agenda_ptr < self.scratch.row_end {
            let item_idx = agenda_ptr;
            let item = self.scratch.items[agenda_ptr];
            agenda_ptr += 1;
            if self.scratch.definitive {
                debug!("    agenda: {}", self.item_to_string(item_idx));
            }

            let rule = item.rule_idx();
            let after_dot = self.grammar.sym_idx_dot(rule);

            // If 'rule' is a complete Earley item ...
            if after_dot == CSymIdx::NULL {
                let flags = self.grammar.sym_flags_lhs(rule);
                let lhs = self.grammar.sym_idx_lhs(rule);

                if self.scratch.definitive && flags.stop_capture() {
                    let var_name = self
                        .grammar
                        .sym_data(lhs)
                        .props
                        .stop_capture_name
                        .as_ref()
                        .unwrap();

                    let bytes = lexeme.hidden_bytes();
                    self.captures.push((var_name.clone(), bytes.to_vec()));
                }

                if self.scratch.definitive && flags.capture() {
                    let var_name = self
                        .grammar
                        .sym_data(lhs)
                        .props
                        .capture_name
                        .as_ref()
                        .unwrap();
                    let mut bytes = Vec::new();
                    let capture_start = item.start_pos();
                    if capture_start < curr_idx {
                        bytes = self.row_infos[capture_start..curr_idx]
                            .iter()
                            .map(|ri| ri.lexeme.visible_bytes())
                            .collect::<Vec<_>>()
                            .concat();
                    }
                    bytes.extend_from_slice(lexeme.visible_bytes());
                    debug!(
                        "      capture: {} {:?}",
                        var_name,
                        String::from_utf8_lossy(&bytes)
                    );
                    self.captures.push((var_name.clone(), bytes));
                }

                if item.start_pos() < curr_idx {
                    // if item.start_pos() == curr_idx, then we handled it below in the nullable check

                    // The main completion inference rule (slide 21 in Kallmeyer 2018)
                    for i in self.rows[item.start_pos()].item_indices() {
                        let item = self.scratch.items[i];
                        if self.grammar.sym_idx_dot(item.rule_idx()) == lhs {
                            self.scratch.add_unique(item.advance_dot(), i, "complete");
                        }
                    }
                }
            } else {
                // ... if 'rule' is an incompletion
                let sym_data = self.grammar.sym_data(after_dot);
                if let Some(lx) = sym_data.lexeme {
                    allowed_lexemes.set(lx.as_usize(), true);
                    if self.scratch.definitive {
                        // In definitive mode, set 'max_tokens' for
                        // the post-dot symbol.
                        max_tokens.push((lx, sym_data.props.max_tokens));
                    }
                }

                // The completion inference rule for nullable symbols
                // (slide 20 in Kallmeyer 2018).
                if sym_data.is_nullable {
                    self.scratch
                        .add_unique(item.advance_dot(), item_idx, "null");
                    if self.scratch.definitive && sym_data.props.capture_name.is_some() {
                        // nullable capture
                        let var_name = sym_data.props.capture_name.as_ref().unwrap();
                        debug!("      capture: {} NULL", var_name);
                        self.captures.push((var_name.clone(), vec![]));
                    }
                }

                // The top-down, or prediction, inference rule.
                // (slide 20 in Kallmeyer 2018)
                for rule in &sym_data.rules {
                    let new_item = Item::new(*rule, curr_idx);
                    self.scratch.add_unique(new_item, item_idx, "predict");
                }
            }
        }

        let row_len = self.scratch.row_len();

        self.stats.rows += 1;

        if row_len == 0 {
            false
        } else {
            self.stats.all_items += row_len;

            // Always accept a SKIP lexeme
            allowed_lexemes.set(LexemeIdx::SKIP.as_usize(), true);

            if self.scratch.definitive {
                debug!(
                    "  push row: {}",
                    self.lexer_spec().dbg_lexeme_set(&allowed_lexemes),
                );
            }

            // Add the working row to the parser state
            let idx = self.num_rows();
            let row = self.scratch.work_row(allowed_lexemes);
            if self.rows.len() == 0 || self.rows.len() == idx {
                // If the physical 'rows' Vec is full, we push a new row
                // otherwise ...
                self.rows.push(row);
            } else {
                // ... we put the new row at the end of the virtual
                // stack as tracked by the lexer.
                self.rows[idx] = row;
            }

            if self.scratch.definitive {
                // Clear all row info data after the
                // working row.
                if self.row_infos.len() > idx {
                    self.row_infos.drain(idx..);
                }

                // We collect and prune the information in
                // 'max_tokens' into a new hash, 'max_tokens_map',
                // which will replace 'max_tokens'.
                let mut max_tokens_map = HashMap::default();

                // If a lexeme allowed twice, set its value in the
                // 'max_tokens_map' to the maximum of the max_tokens
                // values specified.
                for (lx, mx) in max_tokens {
                    if let Some(ex) = max_tokens_map.get(&lx) {
                        if *ex < mx {
                            max_tokens_map.insert(lx, mx);
                        }
                    } else {
                        max_tokens_map.insert(lx, mx);
                    }
                }

                // Some entries in 'max_tokens_map' will
                // explicitly indicate that the number of tokens is
                // unlimited with a value of usize::MAX.  Since this
                // is the default for non-existent hash entries, we
                // save space by removing entries whose value is
                // usize::MAX.
                let mut to_remove = vec![];
                for (lx, mx) in max_tokens_map.iter() {
                    if *mx == usize::MAX {
                        to_remove.push(*lx);
                    }
                }
                for lx in to_remove {
                    max_tokens_map.remove(&lx);
                }

                // Typically, the current byte was not yet pushed,
                // yet it's part of the previous lexeme.
                // This is not true for the first row (which is checked here),
                // or when there is a transition byte (which is corrected in
                // lexer_state_for_added_row())
                let mut start_byte_idx = self.bytes.len();
                if start_byte_idx > 0 {
                    start_byte_idx += 1;
                }

                self.row_infos.push(RowInfo {
                    lexeme: Lexeme::bogus(),
                    token_idx_start: self.token_idx,
                    token_idx_stop: self.token_idx,
                    start_byte_idx,
                    max_tokens: Arc::new(max_tokens_map),
                });
                // debug!("  push: {idx} {} {}", self.rows.len(), self.row_infos.len());
            }

            true
        }
    }

    // curr_row_bytes() looks in the lexer stack, and returns
    // the bytes for the current row as a 'Vec'.
    #[inline(always)]
    fn curr_row_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![];
        let row_idx = self.num_rows() - 1;
        for back in self.lexer_stack.iter().rev() {
            if back.row_idx as usize != row_idx {
                break;
            }
            if let Some(b) = back.byte {
                bytes.push(b);
            }
        }
        bytes.reverse();
        bytes
    }

    fn lexer_spec(&self) -> &LexerSpec {
        self.grammar.lexer_spec()
    }

    // mk_lexeme() converts a pre-lexeme for the current row into
    // a lexeme (ie., it determines the bytes that go into the lexeme), and returns it.
    #[inline(always)]
    fn mk_lexeme(&self, byte: Option<u8>, pre_lexeme: PreLexeme) -> Lexeme {
        let mut bytes = self.curr_row_bytes();
        if byte.is_some() {
            bytes.push(byte.unwrap());
        }

        Lexeme::new(pre_lexeme.idx, bytes, pre_lexeme.hidden_len)
    }

    fn has_forced_bytes(&self, allowed_lexemes: &SimpleVob, bytes: &[u8]) -> bool {
        // note that this is also used when computing token mask
        if allowed_lexemes.is_zero() {
            return false;
        }
        let mut matched_something = false;
        for lexeme_idx in allowed_lexemes.iter() {
            let lex_spec = &self.lexer_spec().lexemes[lexeme_idx as usize];
            if lexeme_idx as usize == LexemeIdx::SKIP.as_usize()
                && matches!(lex_spec.rx, RegexAst::NoMatch)
            {
                continue;
            }
            if !lex_spec.has_forced_bytes(bytes) {
                return false;
            }
            matched_something = true;
        }
        // debug!("   forced ok {:?}", String::from_utf8_lossy(bytes));
        matched_something
    }

    #[inline(always)]
    fn lexer_state_for_added_row(
        &mut self,
        shared: &mut SharedState,
        lexeme: Lexeme,
        transition_byte: Option<u8>,
    ) -> LexerState {
        // note, that while self.rows[] is updated, the lexer stack is not
        // so the last added row is at self.num_rows(), and not self.num_rows() - 1
        let added_row = self.num_rows();
        let added_row_lexemes = &self.rows[added_row].allowed_lexemes;

        let no_hidden = LexerState {
            row_idx: added_row as u32,
            lexer_state: shared.lexer.start_state(added_row_lexemes, transition_byte),
            byte: transition_byte,
        };

        if self.scratch.definitive {
            // save lexeme at the last row, before we mess with the stack
            self.row_infos[added_row - 1].lexeme = lexeme;
            // if there is a transition byte it means it goes to the next lexeme,
            // and thus we were overeager assigning start_byte_idx,
            // so we need to correct it
            if transition_byte.is_some() {
                self.row_infos[added_row].start_byte_idx -= 1;
            }
            debug!(
                "lex: re-start {:?} (via {:?}); allowed: {}",
                no_hidden.lexer_state,
                transition_byte.map(|b| b as char),
                self.lexer_spec().dbg_lexeme_set(added_row_lexemes)
            );
        }

        no_hidden
    }

    #[inline(always)]
    fn handle_hidden_bytes(
        &mut self,
        shared: &mut SharedState,
        no_hidden: LexerState,
        lexeme_byte: Option<u8>,
        pre_lexeme: PreLexeme,
    ) -> bool {
        let added_row_lexemes = &self.rows[self.num_rows()].allowed_lexemes;

        // make sure we have a real lexeme
        let lexeme = self.mk_lexeme(lexeme_byte, pre_lexeme);

        let hidden_bytes = lexeme.hidden_bytes();
        assert!(hidden_bytes.len() == pre_lexeme.hidden_len);

        if self.scratch.definitive {
            trace!(
                "  hidden_bytes: {} {}",
                self.lexer_spec().dbg_lexeme_set(added_row_lexemes),
                String::from_utf8_lossy(&hidden_bytes)
            );
        }

        if self.has_forced_bytes(added_row_lexemes, &hidden_bytes) {
            if self.scratch.definitive {
                trace!("  hidden forced");
            }
            let mut lexer_state = shared.lexer.start_state(added_row_lexemes, None);
            // if the bytes are forced, we just advance the lexer
            // by replacing the top lexer states
            self.pop_lexer_states(hidden_bytes.len() - 1);
            for idx in 0..hidden_bytes.len() {
                let b = hidden_bytes[idx];
                match shared
                    .lexer
                    .advance(lexer_state, b, self.scratch.definitive)
                {
                    LexerResult::State(next_state, _) => {
                        lexer_state = next_state;
                    }
                    LexerResult::Error => panic!("hidden byte failed; {:?}", hidden_bytes),
                    LexerResult::Lexeme(second_lexeme) => {
                        if self.scratch.definitive {
                            debug!("hidden bytes lexeme: {:?}", second_lexeme);
                        }
                        assert!(
                            idx == hidden_bytes.len() - 1,
                            "lexeme in the middle of hidden bytes"
                        );

                        // save current state, we'll need to pop it later
                        self.lexer_stack.push(LexerState {
                            lexer_state,
                            byte: None,
                            ..no_hidden
                        });
                        let r = self.advance_parser(shared, second_lexeme);
                        // println!("hidden bytes lexeme: {:?} -> {r}", second_lexeme);
                        if r {
                            // here, advance_parser() has pushed a state; we replace our state with it
                            let new_top = self.lexer_stack.pop().unwrap();
                            *self.lexer_stack.last_mut().unwrap() = new_top;
                            return true;
                        } else {
                            // otherwise, we just pop our state
                            // This shouldn't happen though
                            // (the parser was allowing this lexeme and now it doesn't like it)
                            self.lexer_stack.pop();
                            return false;
                        }
                    }
                }
                self.lexer_stack.push(LexerState {
                    lexer_state,
                    byte: Some(b),
                    ..no_hidden
                });
            }
            if self.scratch.definitive {
                self.assert_definitive();
            }
        } else {
            if self.scratch.definitive {
                self.assert_definitive();
                self.backtrack_byte_count = hidden_bytes.len();
                // set it up for matching after backtrack
                self.lexer_stack.push(LexerState {
                    lexer_state: shared.lexer.start_state(added_row_lexemes, None),
                    byte: None,
                    ..no_hidden
                });
            } else {
                // prevent any further matches in this branch
                self.lexer_stack.push(LexerState {
                    lexer_state: shared.lexer.a_dead_state(),
                    byte: None,
                    ..no_hidden
                });
            }
            // panic!("hidden bytes not forced");
        }

        true
    }

    /// Advance the parser with given 'pre_lexeme'.
    /// On return, the lexer_state will be the state *after* consuming
    /// 'pre_lexeme'.  As a special case, a following single byte lexeme
    /// is also consumed.
    ///
    // The new lexer state will be an initial lexer states when the lexing
    // is lazy.  If the lexing was greedy, it will be an initial lexer state
    // advanced to the byte which produced the greedy lexeme.

    // This is never inlined anyways, so better make it formal
    #[inline(never)]
    fn advance_parser(&mut self, shared: &mut SharedState, pre_lexeme: PreLexeme) -> bool {
        if self.stats.all_items > self.max_all_items {
            return false;
        }

        // this byte will be applied to the next lexeme
        let transition_byte = if pre_lexeme.byte_next_row {
            pre_lexeme.byte
        } else {
            None
        };
        // this is the last byte of the lexeme
        let lexeme_byte = if pre_lexeme.byte_next_row {
            None
        } else {
            pre_lexeme.byte
        };
        let lexeme_idx = pre_lexeme.idx;

        let lexeme = if self.scratch.definitive {
            self.mk_lexeme(lexeme_byte, pre_lexeme)
        } else {
            Lexeme::just_idx(lexeme_idx)
        };

        let scan_res = if lexeme.idx == LexemeIdx::SKIP {
            // If this is the SKIP lexeme, then skip it
            self.scan_skip_lexeme(&lexeme)
        } else {
            // For all but the SKIP lexeme, process this lexeme
            // with the parser
            self.scan(&lexeme)
        };

        if scan_res {
            let mut no_hidden = self.lexer_state_for_added_row(shared, lexeme, transition_byte);

            if pre_lexeme.hidden_len > 0 {
                return self.handle_hidden_bytes(shared, no_hidden, lexeme_byte, pre_lexeme);
            } else {
                if pre_lexeme.byte_next_row && no_hidden.lexer_state.is_dead() {
                    if self.scratch.definitive {
                        // clean up row infos if needed
                        self.row_infos.drain(no_hidden.row_idx as usize..);
                    }
                    return false;
                }
                if let Some(b) = transition_byte {
                    // At this point there may be a single-byte lexeme after the one
                    // we just recognized.  For example, assuming C language, in the
                    // token "foo(", once we recognize the "foo" lexeme, we immediately
                    // have a single byte "(" lexeme.  We deal with these here.
                    let single = shared
                        .lexer
                        .check_for_single_byte_lexeme(no_hidden.lexer_state, b);
                    if let Some(second_lexeme) = single {
                        if self.scratch.definitive {
                            debug!("single byte lexeme: {:?}", second_lexeme);
                        }
                        no_hidden.byte = None;
                        self.lexer_stack.push(no_hidden);

                        // disallow recursion depth > 2
                        assert!(pre_lexeme.byte_next_row);
                        assert!(!second_lexeme.byte_next_row);

                        let r = self.advance_parser(shared, second_lexeme);
                        if r {
                            let new_top = self.lexer_stack.pop().unwrap();
                            *self.lexer_stack.last_mut().unwrap() = new_top;
                            return true;
                        } else {
                            self.lexer_stack.pop();
                            return false;
                        }
                    }
                }
                self.lexer_stack.push(no_hidden);
            }
            if self.scratch.definitive {
                self.assert_definitive();
            }
            true
        } else {
            if self.scratch.definitive {
                debug!("  scan failed");
            }
            false
        }
    }
}

pub struct ParserRecognizer<'a> {
    state: &'a mut ParserState,
    shared: &'a mut SharedState,
}

impl<'a> ParserRecognizer<'a> {
    pub fn lexer(&mut self) -> &mut Lexer {
        &mut self.shared.lexer
    }
    pub fn lexer_state(&self) -> StateID {
        self.state.lexer_state().lexer_state
    }
}

pub trait BiasComputer: Send + Sync {
    fn compute_bias<'a>(&self, rec: &mut ParserRecognizer<'a>, start: &[u8]) -> SimpleVob;
    fn trie(&self) -> &TokTrie;
}

pub struct DefaultBiasComputer {
    tok_env: TokEnv,
}

impl DefaultBiasComputer {
    pub fn new(tok_env: TokEnv) -> Self {
        DefaultBiasComputer { tok_env }
    }
}

impl BiasComputer for DefaultBiasComputer {
    fn compute_bias<'b>(&self, rec: &mut ParserRecognizer<'b>, start: &[u8]) -> SimpleVob {
        let mut set = self.trie().alloc_token_set();
        self.trie().add_bias(rec, &mut set, start);
        set
    }

    fn trie(&self) -> &TokTrie {
        self.tok_env.tok_trie()
    }
}

// Processing of the parser and the lexer is heavily interlocked.
// The 'Recognizer' trait is used as the interface for this.
// See the documentation for TokTrie in README.md and implementation.md:
// https://github.com/microsoft/toktrie
// and
// https://github.com/microsoft/toktrie/blob/main/implementation.md .
impl<'a> Recognizer for ParserRecognizer<'a> {
    #[inline(always)]
    fn pop_bytes(&mut self, num: usize) {
        self.state.pop_lexer_states(num);
    }

    // For this Earley parser, collapse does nothing -- it is a no-op
    fn collapse(&mut self) {
        // This actually means "commit" - can no longer backtrack past this point.
        // However, this parser ignores it.
    }

    fn special_allowed(&mut self, _tok: SpecialToken) -> bool {
        // handle EOS logic outside
        unreachable!("special_allowed")
    }

    fn trie_started(&mut self) {
        self.state.trie_started_inner();
        self.state.flush_gen_grammar(self.shared);
    }

    fn trie_finished(&mut self) {
        self.state.trie_finished_inner();
    }

    // try_push_byte() is the "speculative" version of try_push_byte_definitive().
    // It attempts to advance the lexer and parser one byte.  It returns true
    // if it succeeds in doing this, true otherwise.  It is often invoked indirectly by the
    // add_bias_inner() method of TokTrie.  In this file, that can happen via the add_bias()
    // and the various compute_bias() methods.
    #[inline(always)]
    fn try_push_byte(&mut self, byte: u8) -> bool {
        let stats = false;

        let lexer_logging = false;
        let curr = self.state.lexer_state();
        let res = self
            .shared
            .lexer
            .advance(curr.lexer_state, byte, lexer_logging);

        if stats {
            // this is always true (not only with stats) but checking it has significant cost
            assert!(!self.state.scratch.definitive);

            self.state.stats.lexer_ops += 1;
            match res {
                LexerResult::State(_, _) => {}
                LexerResult::Error => self.state.stats.num_lex_errors += 1,
                LexerResult::Lexeme(_) => self.state.stats.num_lexemes += 1,
            }
        }

        self.state.advance_lexer_or_parser(self.shared, res, curr)
    }
}

fn item_to_string(g: &CGrammar, item: &Item) -> String {
    format!(
        "{} @{}",
        g.rule_to_string(item.rule_idx()),
        item.start_pos(),
    )
}

pub enum ParserError {
    LexerError(String),
    ParserError(String),
}

impl ParserError {
    pub fn stop_reason(&self) -> StopReason {
        match self {
            ParserError::LexerError(_) => StopReason::LexerTooComplex,
            ParserError::ParserError(_) => StopReason::ParserTooComplex,
        }
    }

    pub fn message(&self) -> String {
        match self {
            ParserError::LexerError(s) => format!("lexer error: {}", s),
            ParserError::ParserError(s) => format!("parser error: {}", s),
        }
    }
}

impl Parser {
    pub fn new(
        grammar: Arc<CGrammar>,
        options: GenGrammarOptions,
        limits: ParserLimits,
    ) -> Result<Self> {
        let (state, lexer) = ParserState::new(grammar, options, limits)?;
        let shared = Arc::new(Mutex::new(SharedState { lexer }));
        Ok(Parser { shared, state })
    }

    /// This is a top-level method in this file.  It is called by mid_process_inner()
    /// in TokenParser in tokenparser.rs.  It is used by the mid_process() method of
    /// the LLInterpreter interface.
    pub fn compute_bias_after_gen_grammar(
        &mut self,
        computer: &dyn BiasComputer,
        symidx: CSymIdx,
    ) -> (bool, SimpleVob) {
        self.state.trie_gen_grammar = Some(symidx);
        let r = self.compute_bias(computer, &[]);
        assert!(self.state.trie_gen_grammar.is_none());
        (self.state.trie_gen_grammar_accepting, r)
    }

    /// This is a top-level method in this file.  It is called by mid_process_inner()
    /// in TokenParser in tokenparser.rs.  It is used by the mid_process() method of
    /// the LLInterpreter interface.
    pub fn compute_bias(&mut self, computer: &dyn BiasComputer, start: &[u8]) -> SimpleVob {
        let mut shared = self.shared.lock().unwrap();
        self.state.compute_bias(&mut shared, computer, start)
    }

    pub fn captures(&self) -> &[(String, Vec<u8>)] {
        &self.state.captures
    }

    pub fn stats(&self) -> &ParserStats {
        &self.state.stats
    }

    pub(crate) fn take_global_state_from(&mut self, other: &mut Parser) {
        self.state.stats = other.state.stats.clone();
        self.state.captures = std::mem::take(&mut other.state.captures);
    }

    // The "hidden" feature must be supported for historical reasons.
    // It is used for 'gen(stop="foo')'.  The result of this 'gen'
    // must not include 'foo', even though the LLM generated 'foo'.
    // The bytes in 'foo' are therefore said to be "hidden".
    pub fn hidden_start(&self) -> usize {
        let mut shared = self.shared.lock().unwrap();
        self.state.hidden_start(&mut shared)
    }

    pub fn lexer_stats(&self) -> String {
        let shared = self.shared.lock().unwrap();
        shared.lexer.dfa.stats()
    }

    pub fn get_error(&self) -> Option<ParserError> {
        let shared = self.shared.lock().unwrap();
        if let Some(e) = shared.lexer.dfa.get_error() {
            return Some(ParserError::LexerError(e));
        }
        if let Some(e) = &self.state.parser_error {
            return Some(ParserError::ParserError(e.clone()));
        }
        None
    }

    pub fn with_recognizer<T>(&mut self, f: impl FnOnce(&mut ParserRecognizer) -> T) -> T {
        let mut shared = self.shared.lock().unwrap();
        let mut p = ParserRecognizer {
            shared: &mut shared,
            state: &mut self.state,
        };
        f(&mut p)
    }

    pub fn scan_gen_grammar(&mut self, symidx: CSymIdx, inner_bytes: Vec<u8>) -> bool {
        self.state.assert_definitive();
        let mut shared = self.shared.lock().unwrap();
        self.state
            .scan_gen_grammar_inner(&mut shared, symidx, inner_bytes)
    }

    pub fn get_bytes(&self) -> &[u8] {
        self.state.get_bytes()
    }

    pub fn force_bytes(&mut self) -> &[u8] {
        let mut shared = self.shared.lock().unwrap();
        self.state.force_bytes(&mut shared)
    }

    pub fn scan_eos(&mut self) -> bool {
        let mut shared = self.shared.lock().unwrap();
        self.state.scan_eos(&mut shared)
    }

    pub(crate) fn apply_forced(&mut self, byte_idx: usize) {
        self.state.byte_to_token_idx.resize(byte_idx, 0);
    }

    pub fn apply_token(&mut self, tok_bytes: &[u8]) -> Result<usize> {
        let mut shared = self.shared.lock().unwrap();
        let r = self.state.apply_token(&mut shared, tok_bytes);
        self.state.token_idx += 1;
        r
    }

    pub fn filter_max_tokens(&mut self) {
        self.state.filter_max_tokens();
    }

    pub fn dbg_row_infos(&self, label: &str) {
        debug!(
            "row infos {}: token_idx: {}; applied bytes: {}/{}",
            label,
            self.state.token_idx,
            self.state.byte_to_token_idx.len(),
            self.state.bytes.len()
        );
        for infos in self.state.row_infos.iter() {
            debug!("  {}", infos.dbg(self.state.lexer_spec()));
        }
    }

    pub fn is_accepting(&mut self) -> bool {
        let mut shared = self.shared.lock().unwrap();
        self.state.is_accepting(&mut shared)
    }

    pub fn has_pending_lexeme_bytes(&self) -> bool {
        self.state.has_pending_lexeme_bytes()
    }

    pub fn grammar(&self) -> &CGrammar {
        &self.state.grammar
    }

    pub fn can_advance(&self) -> bool {
        self.state.can_advance()
    }

    pub fn temperature(&self) -> Option<f32> {
        self.state.temperature()
    }

    pub fn maybe_gen_grammar(&mut self) -> Option<(String, CSymIdx, GenGrammarOptions)> {
        self.state.maybe_gen_grammar()
    }

    pub fn deep_clone(&self) -> Self {
        let mut copy = self.clone();
        let shared = self.shared.lock().unwrap();
        copy.shared = Arc::new(Mutex::new(shared.clone()));
        copy
    }
}
