use std::sync::{Arc, Mutex};

use anyhow::Result;
use toktrie::{InferenceCapabilities, TokEnv};

use crate::{
    api::{ParserLimits, TopLevelGrammar},
    earley::{SlicedBiasComputer, XorShift},
    Logger, TokenParser,
};

pub struct ParserFactory {
    tok_env: TokEnv,
    slicer: Arc<SlicedBiasComputer>,
    inference_caps: InferenceCapabilities,
    stderr_log_level: u32,
    buffer_log_level: u32,
    limits: ParserLimits,
    seed: Mutex<XorShift>,
}

impl ParserFactory {
    pub fn new(
        tok_env: &TokEnv,
        inference_caps: InferenceCapabilities,
        regexes: &Vec<String>,
    ) -> Self {
        let slicer = Arc::new(SlicedBiasComputer::new(tok_env, regexes));
        ParserFactory {
            tok_env: tok_env.clone(),
            slicer,
            inference_caps,
            stderr_log_level: 1,
            buffer_log_level: 0,
            seed: Mutex::new(XorShift::default()),
            limits: ParserLimits::default(),
        }
    }

    pub fn limits_mut(&mut self) -> &mut ParserLimits {
        &mut self.limits
    }

    pub fn tok_env(&self) -> &TokEnv {
        &self.tok_env
    }

    pub fn quiet(&mut self) -> &mut Self {
        self.stderr_log_level = 0;
        self.buffer_log_level = 0;
        self
    }

    pub fn extra_lexemes(&self) -> Vec<String> {
        self.slicer.extra_lexemes()
    }

    pub fn slicer(&self) -> Arc<SlicedBiasComputer> {
        self.slicer.clone()
    }

    pub fn post_process_parser(&self, parser: &mut TokenParser) {
        parser.bias_computer = self.slicer.clone();
        let mut rng = self.seed.lock().unwrap();
        rng.next_alt();
        parser.parser.metrics_mut().rand = rng.clone();
    }

    pub fn create_parser(&self, grammar: TopLevelGrammar) -> Result<TokenParser> {
        let mut parser = TokenParser::from_llguidance_json(
            self.tok_env.clone(),
            grammar,
            Logger::new(self.buffer_log_level, self.stderr_log_level),
            self.inference_caps.clone(),
            self.limits.clone(),
            self.extra_lexemes(),
        )?;
        self.post_process_parser(&mut parser);
        Ok(parser)
    }
}
