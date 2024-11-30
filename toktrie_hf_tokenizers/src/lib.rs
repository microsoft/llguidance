use anyhow::{anyhow, bail, Result};
use rustc_hash::FxHashMap;
use std::{collections::BTreeMap, sync::Arc};
use tokenizers::{normalizers::Sequence, FromPretrainedParameters, NormalizerWrapper, Tokenizer};
use toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};

pub struct ByteTokenizer {
    pub hf_model: String,
    pub hf_tokenizer: Tokenizer,
    info: TokRxInfo,
    token_bytes: Vec<Vec<u8>>,
    pub special: BTreeMap<String, u32>,
}

// useful when debugging this: https://www.cogsci.ed.ac.uk/~richard/utf-8.cgi

fn is_self_mapped(c: char) -> bool {
    match c {
        '!'..='~' | '\u{00A1}'..='\u{00AC}' | '\u{00AE}'..='\u{00FF}' => true,
        _ => false,
    }
}

fn build_char_map() -> FxHashMap<char, u8> {
    let mut res = FxHashMap::default();
    let mut k = 0x100u32;
    for byte in 0..=255u8 {
        let c = byte as char;
        if is_self_mapped(c) {
            res.insert(c, byte);
        } else {
            res.insert(char::from_u32(k).unwrap(), byte);
            k += 1;
        }
    }
    res
}

fn strip_suffix(sep: &str, s: &mut String) -> Option<String> {
    let mut parts = s.splitn(2, sep);
    let core = parts.next().unwrap().to_string();
    let suff = parts.next().map(|s| s.to_string());
    *s = core;
    suff
}

impl ByteTokenizer {
    pub fn from_name(name: &str) -> Result<ByteTokenizer> {
        let loaded = if name.starts_with(".") || name.starts_with("/") {
            Tokenizer::from_file(name)
        } else {
            let mut name2 = name.to_string();
            let mut args = FromPretrainedParameters::default();
            match strip_suffix("@", &mut name2) {
                Some(s) => args.revision = s,
                None => {}
            }
            Tokenizer::from_pretrained(name2, Some(args))
        };

        let tok = loaded.map_err(|e| anyhow!("error loading tokenizer: {}", e))?;

        ByteTokenizer::from_tokenizer(tok)
    }

    pub fn from_file(name: &str) -> Result<ByteTokenizer> {
        let tok =
            Tokenizer::from_file(name).map_err(|e| anyhow!("error loading tokenizer: {}", e))?;
        ByteTokenizer::from_tokenizer(tok)
    }

    pub fn from_tokenizer(mut hft: Tokenizer) -> Result<ByteTokenizer> {
        let mut is_byte_level = false;
        let mut is_byte_fallback = false;
        let mut space_ch = ' ';

        // remove the "Prepend space"
        if let Some(n) = hft.get_normalizer() {
            let n = match n {
                NormalizerWrapper::Sequence(x) => NormalizerWrapper::Sequence(Sequence::new(
                    x.get_normalizers()
                        .iter()
                        .filter_map(|n| match n {
                            NormalizerWrapper::Prepend(_) => None,
                            _ => Some(n.clone()),
                        })
                        .collect(),
                )),
                _ => n.clone(),
            };
            hft.with_normalizer(n);
        }

        if let Some(d) = hft.get_decoder() {
            // DecoderWrapper::Sequence() doesn't let one access the decoders
            // so we resort to json munching
            let v = serde_json::to_value(d).unwrap();
            if v["type"].as_str() == Some("ByteLevel") {
                is_byte_level = true;
            } else if v["type"].as_str() == Some("Sequence") {
                if let Some(decoders) = v["decoders"].as_array() {
                    for decoder in decoders {
                        if decoder["type"].as_str() == Some("ByteFallback") {
                            is_byte_fallback = true;
                        } else if decoder["type"].as_str() == Some("Replace")
                            && decoder["content"].as_str() == Some(" ")
                        {
                            if let Some(s) = decoder["pattern"]["String"].as_str() {
                                let s: Vec<char> = s.chars().collect();
                                if s.len() == 1 {
                                    space_ch = s[0];
                                }
                            }
                        }
                    }
                }
            }
        }

        if !is_byte_fallback && !is_byte_level {
            bail!("can't determine decoder type: {:?}", hft.get_decoder());
        }

        let vocab_size = hft.get_vocab_size(true) as u32;
        let added = hft.get_added_tokens_decoder();

        let mut res = ByteTokenizer {
            hf_model: "foobar".to_string(),
            info: TokRxInfo::new(vocab_size, 0),
            special: BTreeMap::new(),
            token_bytes: (0..vocab_size).map(|_| Vec::new()).collect(),
            hf_tokenizer: hft,
        };

        for (id, info) in added.iter() {
            if info.special {
                match info.content.as_str() {
                    "</s>" | "<|endoftext|>" | "<|end_of_text|>" => res.info.tok_eos = *id,
                    "<|end|>" | "<|eot_id|>" => res.info.tok_end_of_turn = Some(*id),
                    "<unk>" | "<|unk|>" => res.info.tok_unk = Some(*id),
                    "<pad>" | "<|pad|>" => res.info.tok_pad = Some(*id),
                    _ => {}
                }
                res.special.insert(info.content.clone(), *id);
            } else {
                res.token_bytes[*id as usize] = info.content.clone().into_bytes();
            }
        }

        let char_map = build_char_map();

        for tok_id in 0..vocab_size {
            if let Some(tok_name) = res.hf_tokenizer.id_to_token(tok_id) {
                let bytes = if added.contains_key(&tok_id) {
                    let mut bytes = tok_name.as_bytes().to_vec();
                    bytes.insert(0, TokTrie::SPECIAL_TOKEN_MARKER);
                    bytes
                } else if is_byte_fallback {
                    if tok_name.len() == 6 && tok_name.starts_with("<0x") && tok_name.ends_with(">")
                    {
                        // parse hex number from tok_name
                        let hex_str = &tok_name[3..5];
                        let byte = u8::from_str_radix(hex_str, 16).unwrap();
                        vec![byte]
                    } else {
                        assert!(!tok_name.starts_with("<0x"));
                        let tok_name = tok_name.replace(space_ch, " ");
                        tok_name.as_bytes().to_vec()
                    }
                } else if is_byte_level {
                    let bytes: Result<Vec<u8>> = tok_name
                        .chars()
                        .map(|c| {
                            char_map
                                .get(&c)
                                .map(|c| *c)
                                .ok_or_else(|| anyhow!("missing char: {}", c))
                        })
                        .collect();
                    match bytes {
                        Ok(b) => b,
                        Err(e) => {
                            log::warn!("error: {} for {:?}", e, tok_name);
                            continue;
                        }
                    }
                } else {
                    panic!();
                };
                res.token_bytes[tok_id as usize] = bytes;
            } else {
                log::warn!("missing token: {}", tok_id);
            }
        }

        Ok(res)
    }

    pub fn tokrx_info(&self) -> TokRxInfo {
        self.info.clone()
    }
    pub fn token_bytes(&self) -> Vec<Vec<u8>> {
        self.token_bytes.clone()
    }

    pub fn add_missing_tokens(&mut self, vocab_size: usize) {
        assert!(self.info.vocab_size == self.token_bytes.len() as u32);
        assert!(vocab_size >= self.token_bytes.len());
        assert!(vocab_size - self.token_bytes.len() <= 200);
        while self.token_bytes.len() < vocab_size {
            let idx = self.token_bytes.len();
            let name = format!("<AddedToken_{idx}>");
            self.token_bytes.push(name.as_bytes().to_vec());
            self.info.vocab_size += 1;
            self.special.insert(name, idx as u32);
        }
    }
}

pub struct ByteTokenizerEnv {
    pub tokenizer: ByteTokenizer,
    pub tok_trie: TokTrie,
}

impl ByteTokenizerEnv {
    pub fn from_name(name: &str, n_vocab: Option<usize>) -> Result<ByteTokenizerEnv> {
        let tokenizer = ByteTokenizer::from_name(name)?;
        ByteTokenizerEnv::new(tokenizer, n_vocab)
    }

    pub fn new(tokenizer: ByteTokenizer, n_vocab: Option<usize>) -> Result<ByteTokenizerEnv> {
        let mut info = tokenizer.tokrx_info();
        let mut token_bytes = tokenizer.token_bytes();
        if let Some(n_vocab) = n_vocab {
            if n_vocab < token_bytes.len() {
                bail!("vocab size too small; {} vs {}", n_vocab, token_bytes.len());
            }
            while n_vocab > token_bytes.len() {
                token_bytes.push(Vec::new());
            }
            info.vocab_size = n_vocab as u32;
        }
        let tok_trie = TokTrie::from(&info, &token_bytes);
        Ok(ByteTokenizerEnv {
            tokenizer,
            tok_trie,
        })
    }

    pub fn to_env(self) -> TokEnv {
        Arc::new(self)
    }
}

impl TokenizerEnv for ByteTokenizerEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.tok_trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie.tokenize_with_greedy_fallback(s, |s| {
            self.tokenizer
                .hf_tokenizer
                .encode(s, false)
                .expect("tokenizer error")
                .get_ids()
                .to_vec()
        })
    }
}
