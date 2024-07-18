use std::{borrow::Cow, sync::Arc};

use llguidance_parser::toktrie::{
    self, StepArg, StepResult, TokRxInfo, TokTrie, TokenId, TokenizerEnv,
};
use llguidance_parser::{
    api::TopLevelGrammar,
    output::{ParserOutput, Reporter},
    TokenParser,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
#[pyclass]
struct LLInterpreter {
    inner: TokenParser,
    temperature: f32,
    reporter: Reporter,
    step_arg: StepArg,
    last_result: StepResult,
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
        log_level: Option<isize>,
    ) -> PyResult<Self> {
        let env = tokenizer.clone();
        let arg: TopLevelGrammar = serde_json::from_str(llguidance_json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let log_level = log_level.unwrap_or(1);
        let backtrack_supported = true;
        let inner =
            TokenParser::from_llguidance_json(Arc::new(env), arg, log_level, backtrack_supported)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let reporter = Reporter::new(&inner);
        Ok(LLInterpreter {
            inner,
            reporter,
            temperature: 0.0,
            log_level,
            step_arg: StepArg::empty(),
            last_result: StepResult::noop(),
        })
    }

    fn deep_copy(&self) -> Self {
        self.clone()
    }

    fn is_accepting(&self) -> bool {
        self.inner.mid_process_was_accepting()
    }

    fn stop_reason(&self) -> String {
        self.inner.stop_reason().to_string()
    }

    fn process_prompt(&mut self, prompt: Vec<TokenId>) -> Vec<TokenId> {
        self.inner.process_prompt(prompt)
    }

    fn mid_process(&mut self) -> (Option<Cow<[u8]>>, String) {
        let arg = std::mem::replace(&mut self.step_arg, StepArg::empty());
        self.last_result = self.inner.mid_process(arg);
        let r = &self.last_result;
        let is_final = r.is_stop();
        if let Some(t) = r.temperature {
            self.temperature = t;
        }
        let res = PyMidProcessResult {
            progress: self.reporter.get_progress(&mut self.inner, r),
            stop: is_final,
            temperature: self.temperature,
        };
        if is_final {
            (None, serde_json::to_string(&res).unwrap())
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
                res.pop();
                Some(Cow::Owned(res))
            };

            (mask, serde_json::to_string(&res).unwrap())
        }
    }

    fn post_process(&mut self, sampled_token: Option<TokenId>) -> PyResult<(u32, Vec<TokenId>)> {
        let splice = if let Some(t) = sampled_token {
            self.last_result.spliced(t)
        } else {
            if let Some(s) = self.last_result.unconditional_splice() {
                s.clone()
            } else {
                return Err(PyValueError::new_err("Expecting sampled token"));
            }
        };

        self.step_arg = StepArg {
            backtrack: splice.backtrack,
            tokens: splice.ff_tokens.clone(),
            sampled: sampled_token,
        };

        Ok((splice.backtrack, splice.ff_tokens))
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

        let info = TokRxInfo {
            vocab_size: tokens.len() as u32,
            tok_eos,
        };

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
