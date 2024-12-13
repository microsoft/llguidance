use std::sync::Arc;

use crate::{
    derivre::Regex,
    earley::{BiasComputer, ParserRecognizer},
    toktrie::{SimpleVob, TokEnv, TokTrie, TokenId},
};

struct TokenizerSlice {
    idx: usize,
    regex: String,
    trie: TokTrie,
    mask: SimpleVob,
}

pub struct SlicedBiasComputer {
    tok_env: TokEnv,
    slices: Arc<Vec<TokenizerSlice>>,
}

const DEBUG: bool = false;
macro_rules! debug {
    ($($arg:tt)*) => {
        if DEBUG {
            eprint!(">>> ");
            eprintln!($($arg)*);
        }
    };
}

impl SlicedBiasComputer {
    pub fn new(tok_env: &TokEnv, regexes: &Vec<String>) -> Self {
        let mut slices = vec![];

        let trie = tok_env.tok_trie();
        let n_vocab = trie.vocab_size() as TokenId;
        let mut covered = trie.alloc_token_set();
        let mut idx = 0;
        let mut total_nodes = 0;
        let mut regexes = regexes.clone();
        if regexes.len() > 0 {
            regexes.push("".to_string()); // catch-all
        }

        for rx_str in regexes {
            let mut tokens = vec![];
            let mut mask = trie.alloc_token_set();
            if rx_str == "" {
                for tok_idx in 0..n_vocab {
                    if covered.is_allowed(tok_idx) {
                        tokens.push(vec![]);
                    } else {
                        let b = trie.token(tok_idx);
                        tokens.push(b.to_vec());
                        mask.allow_token(tok_idx);
                    }
                }
            } else {
                let mut rx = Regex::new(&rx_str).unwrap();
                for tok_idx in 0..n_vocab {
                    let b = trie.token(tok_idx);
                    if b.is_empty() {
                        tokens.push(vec![]);
                    } else if rx.is_match_bytes(b) && !covered.is_allowed(tok_idx) {
                        covered.allow_token(tok_idx);
                        mask.allow_token(tok_idx);
                        tokens.push(b.to_vec());
                    } else {
                        tokens.push(vec![]);
                    }
                }
            }

            let entry = TokenizerSlice {
                idx,
                regex: rx_str,
                trie: TokTrie::from(trie.info(), &tokens),
                mask,
            };
            debug!(
                "slice{}: /{}/ -> {}",
                idx,
                entry.regex,
                entry.trie.trie_stats()
            );
            if false && DEBUG && entry.regex == "" {
                for (tok_idx, b) in entry.trie.sorted_tokens() {
                    if b.len() > 0 {
                        debug!("  tok{}-> {}", tok_idx, entry.trie.token_dbg(tok_idx));
                    }
                }
            }
            total_nodes += entry.trie.root().subtree_size();

            slices.push(entry);

            idx += 1;
        }
        if total_nodes > 0 {
            debug!("total_nodes: {}", total_nodes);
        }

        SlicedBiasComputer {
            tok_env: tok_env.clone(),
            slices: Arc::new(slices),
        }
    }

    pub fn extra_lexemes(&self) -> Vec<String> {
        self.slices.iter().map(|s| s.regex.clone()).collect()
    }
}

impl BiasComputer for SlicedBiasComputer {
    fn compute_bias<'b>(&self, rec: &mut ParserRecognizer<'b>, start: &[u8]) -> SimpleVob {
        let mut set = self.trie().alloc_token_set();
        let lexer_state = rec.lexer_state();
        if self.slices.len() > 0
            && start.is_empty()
            && rec.lexer_mut().subsume_possible(lexer_state)
        {
            // for JSON string lexer and /[a-zA-Z\u{0080}-\u{10FFFF}]+/ kind of slices
            // we use about 200 of the budget and it takes around 20us
            let budget = 5500;
            let slice_matches = self
                .slices
                .iter()
                .map(|slice| {
                    slice.regex != ""
                        && rec
                            .lexer_mut()
                            .check_subsume(lexer_state, slice.idx, budget)
                            .unwrap_or(false)
                })
                .collect::<Vec<bool>>();

            if slice_matches.iter().all(|&x| x == false) {
                // if nothing matches, just run the full trie
                self.trie().add_bias(rec, &mut set, start);
            } else {
                // otherwise, apply the matching slices, and compute the rest
                for (i, slice) in self.slices.iter().enumerate() {
                    if slice_matches[i] {
                        rec.stats_mut().slices_applied += 1;
                        set.or(&slice.mask);
                    } else {
                        // assert!(slice.regex == "");
                        let t0 = std::time::Instant::now();
                        slice.trie.add_bias(rec, &mut set, start);
                        let us = t0.elapsed().as_micros() as usize;
                        rec.metrics_mut().slicer_leftover_us += us;
                        if slice.regex != "" && set.num_set() > 120_000 {
                            if rec.metrics_mut().rand.one_in(500) {
                                let pos = rec.lexer().possible_lexemes(lexer_state);
                                let spec = rec.lexer().lexer_spec();
                                let msg = format!("{}", spec.dbg_lexeme_set_ext(&pos));
                                println!("{}", msg);
                                rec.metrics_mut().message = msg;
                            }
                        }
                    }
                }
            }
        } else {
            self.trie().add_bias(rec, &mut set, start);
        }

        set
    }

    fn trie(&self) -> &TokTrie {
        self.tok_env.tok_trie()
    }
}
