use std::fmt::Display;
use std::{borrow::Cow, sync::Arc};

use llguidance_parser::api::{GrammarWithLexer, ParserLimits};
use llguidance_parser::toktrie::{
    self, InferenceCapabilities, TokRxInfo, TokTrie, TokenId, TokenizerEnv,
};
use llguidance_parser::{api::TopLevelGrammar, output::ParserOutput, TokenParser};
use llguidance_parser::{
    lark_to_llguidance, Constraint, GrammarBuilder, JsonCompileOptions, Logger,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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

// This is the interface from llguidance to the LLM's.
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

    fn is_accepting(&mut self) -> bool {
        self.inner.parser.is_accepting()
    }

    fn stop_reason(&self) -> String {
        self.inner.parser.stop_reason().to_string()
    }

    fn process_prompt(&mut self, prompt: Vec<TokenId>) -> Vec<TokenId> {
        self.inner.process_prompt(prompt)
    }

    fn compute_mask(&mut self, py: Python<'_>) -> PyResult<(Option<Cow<[u8]>>, String)> {
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

    fn commit_token(
        &mut self,
        sampled_token: Option<TokenId>,
    ) -> PyResult<(u32, Vec<TokenId>)> {
        let pres = self.inner.commit_token(sampled_token).map_err(val_error)?;

        if pres.stop {
            // inner.commit_token() only returns stop, when compute_mask()
            // had already returned stop
            Ok((0, vec![]))
        } else {
            Ok((pres.backtrack, pres.ff_tokens))
        }
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
    fn tok_trie(&self) -> &toktrie::TokTrie {
        &self.tok_trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.tokenize_bytes(s)
    }
}

#[derive(Clone)]
#[pyclass]
struct JsonCompiler {
    item_separator: String,
    key_separator: String,
    whitespace_flexible: bool,
    coerce_one_of: bool,
}

#[pymethods]
impl JsonCompiler {
    #[new]
    #[pyo3(signature = (separators = None, whitespace_flexible = false, coerce_one_of = false))]
    fn py_new(separators: Option<(String, String)>, whitespace_flexible: bool, coerce_one_of: bool) -> Self {
        let (item_separator, key_separator) = separators.unwrap_or_else(|| {
            if whitespace_flexible {
                (",".to_owned(), ":".to_owned())
            } else {
                (", ".to_owned(), ": ".to_owned())
            }
        });
        JsonCompiler {
            item_separator: item_separator,
            key_separator: key_separator,
            whitespace_flexible,
            coerce_one_of,
        }
    }
    fn compile(&self, schema: &str) -> PyResult<String> {
        let schema: Value = serde_json::from_str(schema).map_err(val_error)?;
        let compile_options = JsonCompileOptions {
            item_separator: self.item_separator.clone(),
            key_separator: self.key_separator.clone(),
            whitespace_flexible: self.whitespace_flexible,
            coerce_one_of: self.coerce_one_of,
        };
        let grammar = compile_options.json_to_llg(schema).map_err(val_error)?;
        serde_json::to_string(&grammar).map_err(val_error)
    }
}

#[derive(Clone)]
#[pyclass]
struct LarkCompiler {}

#[pymethods]
impl LarkCompiler {
    #[new]
    fn py_new() -> Self {
        LarkCompiler {}
    }
    fn compile(&self, lark: &str) -> PyResult<String> {
        let grammar = lark_to_llguidance(lark).map_err(val_error)?;
        serde_json::to_string(&grammar).map_err(val_error)
    }
}

#[derive(Clone)]
#[pyclass]
struct RegexCompiler {}

#[pymethods]
impl RegexCompiler {
    #[new]
    fn py_new() -> Self {
        RegexCompiler {}
    }
    fn compile(&self, regex: &str, stop_regex: Option<&str>) -> PyResult<String> {
        let mut builder = GrammarBuilder::new();
        builder.add_grammar(GrammarWithLexer::default());
        let noderef = builder.gen_rx(regex, stop_regex.unwrap_or(""));
        builder.set_start_node(noderef);
        let grammar = builder.finalize().map_err(val_error)?;
        serde_json::to_string(&grammar).map_err(val_error)
    }
}

pub(crate) fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LLTokenizer>()?;
    m.add_class::<LLInterpreter>()?;
    m.add_class::<JsonCompiler>()?;
    m.add_class::<LarkCompiler>()?;
    m.add_class::<RegexCompiler>()?;
    Ok(())
}

fn val_error(e: impl Display) -> PyErr {
    PyValueError::new_err(format!("{e}"))
}
