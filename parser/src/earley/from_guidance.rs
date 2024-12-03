use std::collections::HashMap;
use std::fmt::Write;
use std::{sync::Arc, vec};

use super::grammar::SymIdx;
use super::{grammar::SymbolProps, lexerspec::LexerSpec, CGrammar, Grammar};
use crate::api::{
    GrammarId, GrammarWithLexer, Node, ParserLimits, RegexId, RegexNode, RegexSpec,
    TopLevelGrammar, DEFAULT_CONTEXTUAL,
};
use crate::earley::lexerspec::LexemeClass;
use crate::{lark_to_llguidance, loginfo, JsonCompileOptions, Logger};
use anyhow::{bail, ensure, Result};
use derivre::{ExprRef, JsonQuoteOptions, RegexAst};
use instant::Instant;
use toktrie::TokEnv;

fn resolve_rx(rx_refs: &[ExprRef], node: &RegexSpec) -> Result<RegexAst> {
    match node {
        RegexSpec::Regex(rx) => Ok(RegexAst::Regex(rx.clone())),
        RegexSpec::RegexId(id) => {
            if id.0 >= rx_refs.len() {
                bail!("invalid regex id: {}", id.0)
            } else {
                Ok(RegexAst::ExprRef(rx_refs[id.0]))
            }
        }
    }
}

fn map_rx_ref(rx_refs: &[ExprRef], id: RegexId) -> Result<RegexAst> {
    if id.0 >= rx_refs.len() {
        bail!("invalid regex id when building nodes: {}", id.0)
    } else {
        Ok(RegexAst::ExprRef(rx_refs[id.0]))
    }
}

fn map_rx_refs(rx_refs: &[ExprRef], ids: Vec<RegexId>) -> Result<Vec<RegexAst>> {
    ids.into_iter().map(|id| map_rx_ref(rx_refs, id)).collect()
}

fn map_rx_nodes(
    spec: &mut LexerSpec,
    limits: &ParserLimits,
    rx_nodes: Vec<RegexNode>,
) -> Result<Vec<ExprRef>> {
    let builder = &mut spec.regex_builder;
    let mut rx_refs = vec![];
    for node in rx_nodes {
        rx_refs.push(builder.mk(&map_node(&rx_refs, node)?)?);
        ensure!(
            builder.exprset().cost() <= limits.initial_lexer_fuel,
            "initial lexer configuration (rx_nodes) too big (limit for this grammar: {})",
            limits.initial_lexer_fuel
        );
    }
    return Ok(rx_refs);

    fn map_node(rx_refs: &[ExprRef], node: RegexNode) -> Result<RegexAst> {
        match node {
            RegexNode::Not(id) => Ok(RegexAst::Not(Box::new(map_rx_ref(rx_refs, id)?))),
            RegexNode::Repeat(id, min, max) => Ok(RegexAst::Repeat(
                Box::new(map_rx_ref(rx_refs, id)?),
                min,
                max.unwrap_or(u32::MAX),
            )),
            RegexNode::EmptyString => Ok(RegexAst::EmptyString),
            RegexNode::NoMatch => Ok(RegexAst::NoMatch),
            RegexNode::Regex(rx) => Ok(RegexAst::Regex(rx)),
            RegexNode::Literal(lit) => Ok(RegexAst::Literal(lit)),
            RegexNode::Byte(b) => Ok(RegexAst::Byte(b)),
            RegexNode::ByteSet(bs) => Ok(RegexAst::ByteSet(bs)),
            RegexNode::ByteLiteral(bs) => Ok(RegexAst::ByteLiteral(bs)),
            RegexNode::And(lst) => Ok(RegexAst::And(map_rx_refs(rx_refs, lst)?)),
            RegexNode::Concat(lst) => Ok(RegexAst::Concat(map_rx_refs(rx_refs, lst)?)),
            RegexNode::Or(lst) => Ok(RegexAst::Or(map_rx_refs(rx_refs, lst)?)),
            RegexNode::LookAhead(id) => Ok(RegexAst::LookAhead(Box::new(map_rx_ref(rx_refs, id)?))),
        }
    }
}

fn grammar_from_json(
    tok_env: &TokEnv,
    limits: &mut ParserLimits,
    grm: &mut Grammar,
    lexer_spec: &mut LexerSpec,
    mut input: GrammarWithLexer,
) -> Result<(SymIdx, LexemeClass)> {
    if input.json_schema.is_some() || input.lark_grammar.is_some() {
        ensure!(
            input.nodes.is_empty() && input.rx_nodes.is_empty(),
            "cannot have both json_schema/lark_grammar and nodes/rx_nodes"
        );

        let mut new_grm = if let Some(json_schema) = input.json_schema.take() {
            ensure!(
                input.lark_grammar.is_none(),
                "cannot have both json_schema and lark_grammar"
            );
            let opts: JsonCompileOptions = JsonCompileOptions::default();
            opts.json_to_llg(json_schema)?
        } else {
            lark_to_llguidance(input.lark_grammar.as_ref().unwrap())?
        };

        let g = new_grm.grammars.pop().unwrap();

        input.greedy_skip_rx = g.greedy_skip_rx;
        input.nodes = g.nodes;
        input.rx_nodes = g.rx_nodes;
        input.contextual = g.contextual;

        input.lark_grammar = None;
        input.json_schema = None;
    }

    ensure!(input.nodes.len() > 0, "empty grammar");

    let utf8 = !input.allow_invalid_utf8;
    lexer_spec.regex_builder.utf8(utf8);
    lexer_spec.regex_builder.unicode(utf8);

    let rx_nodes = map_rx_nodes(lexer_spec, limits, input.rx_nodes)?;
    let skip = match input.greedy_skip_rx {
        Some(rx) => resolve_rx(&rx_nodes, &rx)?,
        _ => RegexAst::NoMatch,
    };

    let class = lexer_spec.new_lexeme_class(skip)?;

    if input.no_forcing {
        lexer_spec.no_forcing = true;
    }
    if input.allow_initial_skip && class == LexemeClass::ROOT {
        // TODO: what about sub-grammars?
        lexer_spec.allow_initial_skip = true;
    }

    let node_map = input
        .nodes
        .iter()
        .enumerate()
        .map(|(idx, n)| {
            let props = n.node_props();
            let name = match props.name.as_ref() {
                Some(n) => n.clone(),
                None if props.capture_name.is_some() => {
                    props.capture_name.as_ref().unwrap().clone()
                }
                None => format!("n{}", idx),
            };
            let symprops = SymbolProps {
                commit_point: false,
                hidden: false,
                max_tokens: props.max_tokens.unwrap_or(usize::MAX),
                capture_name: props.capture_name.clone(),
                temperature: 0.0,
                stop_capture_name: None,
            };
            grm.fresh_symbol_ext(&name, symprops)
        })
        .collect::<Vec<_>>();

    let mut size = input.nodes.len();

    for (n, sym) in input.nodes.iter().zip(node_map.iter()) {
        let max_tokens = grm.sym_props(*sym).max_tokens;
        let lhs = *sym;
        match &n {
            Node::Select { among, .. } => {
                // TODO add some optimization to throw these away?
                // ensure!(among.len() > 0, "empty select");
                for v in among {
                    grm.add_rule(lhs, vec![node_map[v.0]])?;
                    size += 2;
                }
            }
            Node::Join { sequence, .. } => {
                let rhs = sequence.iter().map(|idx| node_map[idx.0]).collect();
                size += 1 + sequence.len();
                grm.add_rule(lhs, rhs)?;
            }
            Node::Gen { data, .. } => {
                // parser backtracking relies on only lazy lexers having hidden lexemes
                let body_rx = if data.body_rx.is_missing() {
                    RegexAst::Regex(".*".to_string())
                } else {
                    resolve_rx(&rx_nodes, &data.body_rx)?
                };
                let lazy = data.lazy.unwrap_or(!data.stop_rx.is_missing());
                let stop_rx = if data.stop_rx.is_missing() {
                    RegexAst::EmptyString
                } else {
                    resolve_rx(&rx_nodes, &data.stop_rx)?
                };
                let idx = lexer_spec.add_rx_and_stop(
                    format!("gen_{}", grm.sym_name(lhs)),
                    body_rx,
                    stop_rx,
                    lazy,
                    max_tokens,
                )?;

                let symprops = grm.sym_props_mut(lhs);
                if let Some(t) = data.temperature {
                    symprops.temperature = t;
                }
                if data.stop_capture_name.is_some() {
                    symprops.stop_capture_name = data.stop_capture_name.clone();
                    let wrap_props = symprops.for_wrapper();
                    let wrap_name = format!("stop_wrap_{}", grm.sym_name(lhs));
                    let wrap_sym = grm.fresh_symbol_ext(&wrap_name, wrap_props);
                    grm.make_terminal(wrap_sym, idx, &lexer_spec)?;
                    grm.add_rule(lhs, vec![wrap_sym])?;
                } else {
                    grm.make_terminal(lhs, idx, &lexer_spec)?;
                }
            }
            Node::Lexeme {
                rx,
                contextual,
                temperature,
                json_allowed_escapes,
                json_raw,
                json_string,
                ..
            } => {
                let json_options = if json_string.unwrap_or(false) {
                    Some(JsonQuoteOptions {
                        allowed_escapes: json_allowed_escapes
                            .as_ref()
                            .map_or("nrbtf\\\"u", |e| e.as_str())
                            .to_string(),
                        raw_mode: json_raw.unwrap_or(false),
                    })
                } else {
                    ensure!(
                        json_allowed_escapes.is_none(),
                        "json_allowed_escapes is only valid for json_string"
                    );
                    ensure!(json_raw.is_none(), "json_raw is only valid for json_string");
                    None
                };
                let idx = lexer_spec.add_greedy_lexeme(
                    format!("lex_{}", grm.sym_name(lhs)),
                    resolve_rx(&rx_nodes, rx)?,
                    contextual.unwrap_or(input.contextual.unwrap_or(DEFAULT_CONTEXTUAL)),
                    json_options,
                    max_tokens,
                )?;
                if let Some(t) = temperature {
                    let symprops = grm.sym_props_mut(lhs);
                    symprops.temperature = *t;
                }
                grm.make_terminal(lhs, idx, &lexer_spec)?;
            }
            Node::String { literal, .. } => {
                if literal.is_empty() {
                    grm.add_rule(lhs, vec![])?;
                } else {
                    let idx = lexer_spec.add_simple_literal(
                        format!("str_{}", grm.sym_name(lhs)),
                        &literal,
                        input.contextual.unwrap_or(DEFAULT_CONTEXTUAL),
                    )?;
                    grm.make_terminal(lhs, idx, &lexer_spec)?;
                }
            }
            Node::GenGrammar { data, .. } => {
                grm.make_gen_grammar(lhs, data.clone())?;
            }
            Node::SpecialToken { token, .. } => {
                let trie = tok_env.tok_trie();
                if trie.get_special_token(token).is_none() {
                    let spec = trie.get_special_tokens();
                    bail!(
                        "unknown special token: {:?}; following special tokens are available: {}",
                        token,
                        trie.tokens_dbg(&spec)
                    );
                }
                let idx = lexer_spec.add_special_token(token.clone())?;
                grm.make_terminal(lhs, idx, &lexer_spec)?;
            }
        }

        ensure!(
            lexer_spec.cost() <= limits.initial_lexer_fuel,
            "initial lexer configuration (grammar) too big (limit for this grammar: {})",
            limits.initial_lexer_fuel
        );

        ensure!(
            size <= limits.max_grammar_size,
            "grammar size (number of symbols) too big (limit for this grammar: {})",
            limits.max_grammar_size,
        );
    }

    limits.initial_lexer_fuel = limits.initial_lexer_fuel.saturating_sub(lexer_spec.cost());
    limits.max_grammar_size = limits.max_grammar_size.saturating_sub(size);

    Ok((node_map[0], class))
}

pub fn grammars_from_json(
    input: TopLevelGrammar,
    tok_env: &TokEnv,
    logger: &mut Logger,
    mut limits: ParserLimits,
    extra_lexemes: Vec<String>,
) -> Result<Arc<CGrammar>> {
    let t0 = Instant::now();

    // eprintln!("grammars_from_json: {}", serde_json::to_string(&input).unwrap());

    ensure!(input.grammars.len() > 0, "empty grammars array");

    let mut lexer_spec = LexerSpec::new()?;
    let mut grammar = Grammar::new(None);

    let mut grammar_by_idx = HashMap::new();
    for (idx, grm) in input.grammars.iter().enumerate() {
        if let Some(n) = &grm.name {
            let n = GrammarId::Name(n.to_string());
            if grammar_by_idx.contains_key(&n) {
                bail!("duplicate grammar name: {}", n);
            }
            grammar_by_idx.insert(n, idx);
        }
        grammar_by_idx.insert(GrammarId::Index(idx), idx);
    }

    let root_syms = input
        .grammars
        .into_iter()
        .map(|g| grammar_from_json(tok_env, &mut limits, &mut grammar, &mut lexer_spec, g))
        .collect::<Result<Vec<_>>>()?;

    let grammar_by_idx: HashMap<GrammarId, (SymIdx, LexemeClass)> = grammar_by_idx
        .into_iter()
        .map(|(k, v)| (k, root_syms[v]))
        .collect();

    grammar.resolve_grammar_refs(&mut lexer_spec, &grammar_by_idx)?;

    lexer_spec.add_extra_lexemes(&extra_lexemes);

    let log_grammar = logger.level_enabled(3) || (logger.level_enabled(2) && grammar.is_small());
    if log_grammar {
        writeln!(logger.info_logger(), "{:?}\n{:?}\n", lexer_spec, grammar).unwrap();
    } else if logger.level_enabled(2) {
        writeln!(
            logger.info_logger(),
            "Grammar: (skipping body; log_level=3 will print it); {}",
            grammar.stats()
        )
        .unwrap();
    }

    let t1 = Instant::now();
    grammar = grammar.optimize();

    if log_grammar {
        write!(logger.info_logger(), "  == Optimize ==>\n{:?}", grammar).unwrap();
    } else if logger.level_enabled(2) {
        writeln!(logger.info_logger(), "  ==> {}", grammar.stats()).unwrap();
    }

    let grammars = Arc::new(grammar.compile(lexer_spec));

    loginfo!(
        logger,
        "build grammar: {:?}; optimize: {:?}",
        t1 - t0,
        t1.elapsed()
    );

    Ok(grammars)
}
