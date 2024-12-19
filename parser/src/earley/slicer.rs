use std::sync::Arc;

use derivre::AlphabetInfo;

use crate::{
    derivre::Regex,
    earley::{BiasComputer, ParserRecognizer},
    toktrie::{SimpleVob, TokEnv, TokTrie, TokenId},
};

use super::parser::ITEM_TRACE;

struct TokenizerSlice {
    idx: usize,
    regex: String,
    trie: TokTrie,
    mask: SimpleVob,
}

pub struct SlicedBiasComputer {
    wildcard_slice: TokTrie,
    slices: Arc<Vec<TokenizerSlice>>,
    tok_env: TokEnv,
}

const DEBUG: bool = ITEM_TRACE;
macro_rules! debug {
    ($($arg:tt)*) => {
        if DEBUG {
            eprint!(">>> ");
            eprintln!($($arg)*);
        }
    };
}

impl SlicedBiasComputer {
    pub fn json_slices() -> Vec<String> {
        vec![
            r#"[^"\\\x00-\x1F\x7F]{1,10}"#.to_string(),
            r#"[^"\\\x00-\x1F\x7F]{1,30}"#.to_string(),
            r#"[^"\\\x00-\x1F\x7F]+"#.to_string(),
        ]
    }

    pub fn general_slices() -> Vec<String> {
        // to be improved in future
        Self::json_slices()
    }

    pub fn new(tok_env: &TokEnv, regexes: &Vec<String>) -> Self {
        let mut slices = vec![];

        let trie = tok_env.tok_trie();
        let n_vocab = trie.vocab_size() as TokenId;
        let mut covered = trie.alloc_token_set();
        let mut idx = 0;
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

            slices.push(entry);

            idx += 1;
        }

        let r = SlicedBiasComputer {
            slices: Arc::new(slices),
            wildcard_slice: trie.clone(),
            tok_env: tok_env.clone(),
        };

        debug!("slicer:\n{}", r.stats(false));

        r
    }

    pub fn stats(&self, include_tokens: bool) -> String {
        let mut total_nodes = 0;
        let mut s = String::new();
        for (i, slice) in self.slices.iter().enumerate() {
            total_nodes += slice.trie.root().subtree_size();
            s.push_str(&format!(
                "slice{}: /{}/ -> {}\n",
                i,
                slice.regex,
                slice.trie.trie_stats()
            ));
            if include_tokens {
                for (tok_idx, b) in slice.trie.sorted_tokens() {
                    if b.len() > 0 {
                        s.push_str(&format!(
                            "  tok{}-> {}\n",
                            tok_idx,
                            slice.trie.token_dbg(tok_idx)
                        ));
                    }
                }
            }
        }
        s.push_str(&format!("total_nodes: {}\n", total_nodes));
        s.push_str(&format!("WILDCARD: {}\n", self.wildcard_slice.trie_stats()));
        s
    }

    pub fn extra_lexemes(&self) -> Vec<String> {
        self.slices.iter().map(|s| s.regex.clone()).collect()
    }

    pub fn compress(&self, ai: &AlphabetInfo) -> Self {
        let slices = self
            .slices
            .iter()
            .map(|s| TokenizerSlice {
                idx: s.idx,
                regex: s.regex.clone(),
                trie: compress_trie(&s.trie, ai),
                mask: s.mask.clone(),
            })
            .collect();
        SlicedBiasComputer {
            wildcard_slice: compress_trie(&self.wildcard_slice, ai),
            slices: Arc::new(slices),
            tok_env: self.tok_env.clone(),
        }
    }
}

fn compress_trie(trie: &TokTrie, ai: &AlphabetInfo) -> TokTrie {
    let mut tokens = trie.all_tokens();
    let mut repr = vec![None; 256];
    let repr2 = (0..=255)
        .map(|b| {
            if repr[ai.map(b)].is_none() {
                repr[ai.map(b)] = Some(b);
            }
            repr[ai.map(b)].unwrap()
        })
        .collect::<Vec<u8>>();
    for t in tokens.iter_mut() {
        for i in 0..t.len() {
            t[i] = repr2[t[i] as usize];
        }
    }
    TokTrie::from(trie.info(), &tokens)
}

impl BiasComputer for SlicedBiasComputer {
    fn compute_bias<'b>(&self, rec: &mut ParserRecognizer<'b>, start: &[u8]) -> SimpleVob {
        let mut set = self.trie().alloc_token_set();
        let lexer_state = rec.lexer_state();
        if self.slices.len() > 0
            && start.is_empty()
            && rec.lexer_mut().subsume_possible(lexer_state)
        {
            // set to at least 500
            let budget = 1000;
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
                self.wildcard_slice.add_bias(rec, &mut set, start);
                debug!("no slice matches; {} tokens", set.num_set());
            } else {
                // otherwise, apply the matching slices, and compute the rest
                for (i, slice) in self.slices.iter().enumerate() {
                    if slice_matches[i] {
                        rec.stats_mut().slices_applied += 1;
                        set.or(&slice.mask);
                    } else {
                        // assert!(slice.regex == "");
                        let c0 = if DEBUG { set.num_set() } else { 0 };
                        let t0 = std::time::Instant::now();
                        slice.trie.add_bias(rec, &mut set, start);
                        let us = t0.elapsed().as_micros() as usize;
                        rec.metrics_mut().slicer_leftover_us += us;
                        debug!("slice matches #{}; {} tokens", i, set.num_set() - c0);
                        // if slice.regex != "" && set.num_set() > 120_000 {
                        //     if rec.metrics_mut().rand.one_in(500) {
                        //         let pos = rec.lexer().possible_lexemes(lexer_state);
                        //         let spec = rec.lexer().lexer_spec();
                        //         let msg = format!("{}", spec.dbg_lexeme_set_ext(&pos));
                        //         println!("{}", msg);
                        //         rec.metrics_mut().message = msg;
                        //     }
                        // }
                    }
                }
            }
        } else {
            self.wildcard_slice.add_bias(rec, &mut set, start);
            debug!("slicer disabled; {} tokens", set.num_set());
        }

        debug!("");

        set
    }

    fn trie(&self) -> &TokTrie {
        self.tok_env.tok_trie()
    }
}
