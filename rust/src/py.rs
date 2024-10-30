use std::fmt::Display;
use std::{borrow::Cow, sync::Arc};

use llguidance_parser::api::ParserLimits;
use llguidance_parser::toktrie::{
    self, InferenceCapabilities, TokRxInfo, TokTrie, TokenId, TokenizerEnv,
};
use llguidance_parser::{api::TopLevelGrammar, output::ParserOutput, TokenParser};
use llguidance_parser::{Constraint, Logger};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
#[pyclass]
struct LLInterpreter {
    inner: Constraint,
    #[pyo3(get, set)]
    log_level: isize,
}

#[derive(Clone)]
#[pyclass]
struct LLTokenizer {
    tok_trie: Arc<toktrie::TokTrie>,
    tokenizer_fun: Py<PyAny>,
    #[allow(dead_code)]
    tok_bos: Option<u32>,
}

#[pymethods]
impl LLInterpreter {
    #[new]
    fn py_new(
        tokenizer: &LLTokenizer,
        llguidance_json: &str,
        enable_backtrack: Option<bool>,
        enable_ff_tokens: Option<bool>,
        log_level: Option<isize>,
    ) -> PyResult<Self> {
        let env = tokenizer.clone();
        let arg: TopLevelGrammar = serde_json::from_str(llguidance_json).map_err(val_error)?;
        let log_level = log_level.unwrap_or(1);
        let inference_caps = InferenceCapabilities {
            backtrack: enable_backtrack.unwrap_or(true),
            ff_tokens: enable_ff_tokens.unwrap_or(true),
            conditional_ff_tokens: enable_ff_tokens.unwrap_or(true),
            fork: false,
        };
        let logger = Logger::new(0, std::cmp::max(0, log_level) as u32);
        let inner = TokenParser::from_llguidance_json(
            Arc::new(env),
            arg,
            logger,
            inference_caps,
            ParserLimits::default(),
            vec![],
        )
        .map_err(val_error)?;
        let inner = Constraint::new(inner);
        Ok(LLInterpreter { inner, log_level })
    }

    fn deep_copy(&self) -> Self {
        self.clone()
    }

    fn is_accepting(&self) -> bool {
        self.inner.parser.mid_process_was_accepting()
    }

    fn stop_reason(&self) -> String {
        self.inner.parser.stop_reason().to_string()
    }

    fn process_prompt(&mut self, prompt: Vec<TokenId>) -> Vec<TokenId> {
        self.inner.process_prompt(prompt)
    }

    fn mid_process(&mut self, py: Python<'_>) -> PyResult<(Option<Cow<[u8]>>, String)> {
        let r = py
            .allow_threads(|| self.inner.compute_mask())
            .map_err(val_error)?
            .clone();
        let is_final = r.is_stop();
        let res = PyMidProcessResult {
            progress: self.inner.flush_progress(),
            stop: is_final,
            temperature: self.inner.temperature,
        };
        if is_final {
            Ok((None, serde_json::to_string(&res).unwrap()))
        } else {
            let mask = if r.unconditional_splice().is_some() {
                None
            } else {
                let m = r
                    .sample_mask
                    .as_ref()
                    .expect("expecting unconditional splice or mask");
                let mut res = vec![0u8; m.len()];
                m.iter_set_entries(|i| res[i] = 200);
                Some(Cow::Owned(res))
            };

            Ok((mask, serde_json::to_string(&res).unwrap()))
        }
    }

    fn advance_parser(&mut self, sampled_token: Option<TokenId>) -> PyResult<(u32, Vec<TokenId>)> {
        let pres = self.inner.commit_token(sampled_token).map_err(val_error)?;

        if pres.stop {
            // let the next mid_process() call handle it
            return Ok((0, vec![]));
        }

        Ok((pres.backtrack, pres.ff_tokens))
    }

    fn post_process(&mut self, sampled_token: Option<TokenId>) -> PyResult<(u32, Vec<TokenId>)> {
        self.advance_parser(sampled_token)
    }

    fn has_pending_stop(&self) -> bool {
        self.inner.has_pending_stop()
    }
}

#[derive(Serialize, Deserialize)]
struct PyMidProcessResult {
    progress: Vec<ParserOutput>,
    stop: bool,
    temperature: f32,
}

#[pymethods]
impl LLTokenizer {
    #[new]
    fn py_new(tokenizer: Bound<'_, PyAny>) -> PyResult<Self> {
        let is_tokenizer = tokenizer
            .getattr("is_tokenizer_wrapper")
            .map(|v| v.extract::<bool>())
            .unwrap_or(Ok(false))
            .unwrap_or(false);
        if !is_tokenizer {
            return Err(PyValueError::new_err(
                "Expecting a TokenizerWrapper() class",
            ));
        }

        let mut tokens = tokenizer.getattr("tokens")?.extract::<Vec<Vec<u8>>>()?;

        // no eos_token only applies to ByteTokenizer from Guidance, which we
        // hopefully will not actually use
        let tok_eos = tokenizer
            .getattr("eos_token_id")?
            .extract::<Option<u32>>()?
            .unwrap_or_else(|| {
                let r = tokens.len() as u32;
                tokens.push(vec![]);
                r
            });
        let tok_bos = tokenizer
            .getattr("bos_token_id")?
            .extract::<Option<u32>>()?;

        // we want decode_bytes([EOS]) etc to be empty
        tokens[tok_eos as usize] = vec![];
        // if let Some(t) = tok_bos {
        //     tokens[t as usize] = vec![];
        // }

        let info = TokRxInfo::new(tokens.len() as u32, tok_eos);

        let tok_trie = TokTrie::from(&info, &tokens);
        Ok(LLTokenizer {
            tok_trie: Arc::new(tok_trie),
            tokenizer_fun: tokenizer.into(),
            tok_bos,
        })
    }

    fn tokenize_bytes(&self, utf8bytes: &[u8]) -> Vec<TokenId> {
        self.tok_trie.tokenize_with_greedy_fallback(utf8bytes, |s| {
            Python::with_gil(|py| {
                let r = self.tokenizer_fun.call1(py, (s,)).unwrap();
                r.extract::<Vec<TokenId>>(py).unwrap()
            })
        })
    }

    fn tokenize_str(&self, text: &str) -> Vec<TokenId> {
        self.tokenize_bytes(text.as_bytes())
    }

    fn greedy_tokenize(&self, text: &str) -> Vec<u32> {
        self.tok_trie.greedy_tokenize(text.as_bytes())
    }

    fn test_trace_tokens(&self, tokens: Vec<u32>) -> String {
        self.tok_trie.test_trace_tokens(&tokens)
    }

    fn dbg_tokens(&self, tokens: Vec<u32>) -> String {
        self.tok_trie.tokens_dbg(&tokens)
    }

    fn decode_str(&self, tokens: Vec<u32>) -> String {
        self.tok_trie.decode_str(&tokens)
    }

    fn decode_bytes(&self, tokens: Vec<u32>) -> Cow<[u8]> {
        let r = self.tok_trie.decode(&tokens);
        Cow::Owned(r)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.tok_trie.vocab_size() as usize
    }

    #[getter]
    fn eos_token(&self) -> u32 {
        self.tok_trie.eos_token()
    }
}

impl TokenizerEnv for LLTokenizer {
    fn stop(&self) -> ! {
        panic!("STOP"); // TODO?
    }

    fn tok_trie(&self) -> &toktrie::TokTrie {
        &self.tok_trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.tokenize_bytes(s)
    }
}

pub(crate) fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LLTokenizer>()?;
    m.add_class::<LLInterpreter>()?;
    Ok(())
}

fn val_error(e: impl Display) -> PyErr {
    PyValueError::new_err(format!("{e}"))
}
