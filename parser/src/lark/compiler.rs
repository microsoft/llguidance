use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{anyhow, bail, ensure, Result};

use crate::{
    api::{GrammarId, GrammarWithLexer, RegexId, RegexSpec, TopLevelGrammar},
    GrammarBuilder, NodeRef,
};

use super::{ast::*, common::lookup_common_regex, lexer::Location, parser::parse_lark};

#[derive(Debug, Default)]
struct Grammar {
    rules: HashMap<String, Rule>,
    tokens: HashMap<String, TokenDef>,
    ignore: Vec<Expansions>,
}

struct Compiler {
    test_rx: derivre::RegexBuilder,
    builder: GrammarBuilder,
    items: Vec<Item>,
    grammar: Arc<Grammar>,
    node_ids: HashMap<String, NodeRef>,
    regex_ids: HashMap<String, RegexId>,
    in_progress: HashSet<String>,
}

pub fn compile_lark(items: Vec<Item>) -> Result<TopLevelGrammar> {
    let mut c = Compiler {
        builder: GrammarBuilder::new(),
        test_rx: derivre::RegexBuilder::new(),
        items,
        grammar: Arc::new(Grammar::default()),
        node_ids: HashMap::new(),
        regex_ids: HashMap::new(),
        in_progress: HashSet::new(),
    };
    c.execute()?;
    c.builder.finalize()
}

pub fn lark_to_llguidance(lark: &str) -> Result<TopLevelGrammar> {
    parse_lark(lark).and_then(compile_lark)
}

impl Compiler {
    fn grammar(&self) -> Arc<Grammar> {
        Arc::clone(&self.grammar)
    }

    fn do_token(&mut self, name: &str) -> Result<RegexId> {
        if let Some(id) = self.regex_ids.get(name) {
            return Ok(*id);
        }
        if self.in_progress.contains(name) {
            bail!("circular reference in token {:?} definition", name);
        }
        self.in_progress.insert(name.to_string());
        let g = self.grammar();
        let token = g
            .tokens
            .get(name)
            .ok_or_else(|| anyhow!("token {:?} not found", name))?;
        let id = self.do_token_expansions(&token.expansions)?;
        self.regex_ids.insert(name.to_string(), id);
        self.in_progress.remove(name);
        Ok(id)
    }

    fn mk_regex(&mut self, info: &str, rx: String) -> Result<RegexId> {
        self.test_rx
            .mk_regex(&rx)
            .map_err(|e| anyhow!("invalid regex {rx:?} (in {info}): {e}"))?;
        Ok(self.builder.regex.regex(rx))
    }

    fn do_token_atom(&mut self, atom: &Atom) -> Result<RegexId> {
        match atom {
            Atom::Group(expansions) => self.do_token_expansions(expansions),
            Atom::Maybe(expansions) => {
                let id = self.do_token_expansions(expansions)?;
                Ok(self.builder.regex.optional(id))
            }
            Atom::Value(value) => match value {
                Value::LiteralRange(a, b) => {
                    ensure!(
                        a.chars().count() == 1,
                        "range start must be a single character"
                    );
                    ensure!(
                        b.chars().count() == 1,
                        "range end must be a single character"
                    );
                    let a = a.chars().next().unwrap();
                    let b = b.chars().next().unwrap();
                    if a <= b {
                        self.mk_regex(
                            "range",
                            format!(
                                "[{}-{}]",
                                regex_syntax::escape(&a.to_string()),
                                regex_syntax::escape(&b.to_string())
                            ),
                        )
                    } else {
                        bail!("invalid range order: {:?}..{:?}", a, b);
                    }
                }
                Value::Name(n) => self.do_token(n),
                Value::LiteralString(val, flags) => {
                    if flags.contains("i") {
                        self.mk_regex(
                            "string with i-flag",
                            format!("(?i){}", regex_syntax::escape(val)),
                        )
                    } else {
                        Ok(self.builder.regex.literal(val.clone()))
                    }
                }
                Value::LiteralRegex(val, flags) => {
                    ensure!(!flags.contains("l"), "l-flag is not supported in regexes");
                    let rx = if flags.is_empty() {
                        val.clone()
                    } else {
                        format!("(?{}){}", flags, val)
                    };
                    self.mk_regex("regex", rx)
                }
                Value::SpecialToken(s) => {
                    bail!("special tokens (like {:?}) cannot be used as terminals", s);
                }
                Value::GrammarRef(g) => {
                    bail!(
                        "grammar references (like {:?}) cannot be used as terminals",
                        g
                    );
                }
                Value::TemplateUsage { .. } => bail!("template usage not supported yet"),
            },
        }
    }

    fn do_token_expr(&mut self, expr: &Expr) -> Result<RegexId> {
        let atom = self.do_token_atom(&expr.atom)?;
        if let Some(range) = &expr.range {
            ensure!(expr.op.is_none(), "ranges not supported with operators");
            ensure!(range.0 >= 0, "range start must be >= 0");
            ensure!(range.1 >= range.0, "range end must be >= start");
            Ok(self
                .builder
                .regex
                .repeat(atom, range.0 as u32, Some(range.1 as u32)))
        } else {
            match &expr.op {
                Some(op) => match op.0.as_str() {
                    "*" => Ok(self.builder.regex.zero_or_more(atom)),
                    "+" => Ok(self.builder.regex.one_or_more(atom)),
                    "?" => Ok(self.builder.regex.optional(atom)),
                    _ => {
                        bail!("unsupported operator: {:?}", op.0);
                    }
                },
                None => Ok(atom),
            }
        }
    }

    fn do_token_expansions(&mut self, expansions: &Expansions) -> Result<RegexId> {
        let options = expansions
            .1
            .iter()
            .map(|alias| {
                let args = alias
                    .expansion
                    .0
                    .iter()
                    .map(|e| self.do_token_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(self.builder.regex.concat(args))
            })
            .collect::<Result<Vec<_>>>()
            .map_err(|e| expansions.0.augment(e))?;
        Ok(self.builder.regex.select(options))
    }

    fn lift_regex(&mut self, rx_id: RegexId) -> Result<NodeRef> {
        Ok(self.builder.lexeme(RegexSpec::RegexId(rx_id), false))
    }

    fn do_atom(&mut self, expr: &Atom) -> Result<NodeRef> {
        match expr {
            Atom::Group(expansions) => self.do_expansions(expansions),
            Atom::Maybe(expansions) => {
                let id = self.do_expansions(expansions)?;
                Ok(self.builder.optional(id))
            }
            Atom::Value(value) => {
                match value {
                    Value::Name(n) => {
                        if self.grammar.rules.contains_key(n) {
                            return self.do_rule(n);
                        } else if self.grammar.tokens.contains_key(n) {
                            // OK -> treat as token
                        } else {
                            bail!("unknown name: {:?}", n);
                        }
                    }
                    Value::SpecialToken(s) => return Ok(self.builder.special_token(s)),
                    Value::GrammarRef(g) => {
                        assert!(g.starts_with("@"));
                        // see if g[1..] is an integer
                        let id = if let Ok(id) = g[1..].parse::<usize>() {
                            GrammarId::Index(id)
                        } else {
                            GrammarId::Name(g[1..].to_string())
                        };
                        return Ok(self.builder.gen_grammar(id));
                    }
                    Value::LiteralRange(_, _)
                    | Value::LiteralString(_, _)
                    | Value::LiteralRegex(_, _) => {
                        // treat as token
                    }
                    Value::TemplateUsage { .. } => {
                        bail!("template usage not supported yet");
                    }
                };
                let rx = self.do_token_atom(expr)?;
                Ok(self.lift_regex(rx)?)
            }
        }
    }

    fn do_expr(&mut self, expr: &Expr) -> Result<NodeRef> {
        let atom = self.do_atom(&expr.atom)?;

        if let Some((a, b)) = expr.range {
            ensure!(expr.op.is_none(), "ranges not supported with operators");
            ensure!(a <= b, "range end must be >= start");
            ensure!(a >= 0, "range start must be >= 0");
            Ok(self.builder.repeat(atom, a as usize, Some(b as usize)))
        } else {
            match &expr.op {
                Some(op) => match op.0.as_str() {
                    "*" => Ok(self.builder.zero_or_more(atom)),
                    "+" => Ok(self.builder.one_or_more(atom)),
                    "?" => Ok(self.builder.optional(atom)),
                    _ => {
                        bail!("unsupported operator: {}", op.0);
                    }
                },
                None => Ok(atom),
            }
        }
    }

    fn do_expansions(&mut self, expansions: &Expansions) -> Result<NodeRef> {
        let options = expansions
            .1
            .iter()
            .map(|alias| {
                let args = alias
                    .expansion
                    .0
                    .iter()
                    .map(|e| self.do_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(self.builder.join(&args))
            })
            .collect::<Result<Vec<_>>>()
            .map_err(|e| expansions.0.augment(e))?;
        Ok(self.builder.select(&options))
    }

    fn do_rule(&mut self, name: &str) -> Result<NodeRef> {
        if let Some(id) = self.node_ids.get(name) {
            return Ok(*id);
        }
        if self.in_progress.contains(name) {
            let id = self.builder.placeholder();
            self.node_ids.insert(name.to_string(), id);
            return Ok(id);
        }
        self.in_progress.insert(name.to_string());
        let g = self.grammar();
        let rule = g
            .rules
            .get(name)
            .ok_or_else(|| anyhow!("rule {:?} not found", name))?;
        let id = self.do_expansions(&rule.expansions)?;
        if let Some(placeholder) = self.node_ids.get(name) {
            self.builder.set_placeholder(*placeholder, id);
        }
        self.node_ids.insert(name.to_string(), id);
        self.in_progress.remove(name);
        Ok(id)
    }

    fn execute(&mut self) -> Result<()> {
        let mut grm = Grammar::default();
        for item in std::mem::take(&mut self.items) {
            let loc = item.location().clone();
            grm.process_item(item).map_err(|e| loc.augment(e))?;
        }
        ensure!(grm.rules.contains_key("start"), "no start rule found");
        let ignore = std::mem::take(&mut grm.ignore);
        self.grammar = Arc::new(grm);
        self.builder.add_grammar(GrammarWithLexer::default());
        let ignore = ignore
            .iter()
            .map(|exp| self.do_token_expansions(exp))
            .collect::<Result<Vec<_>>>()?;
        let start = self.do_rule("start")?;
        self.builder.set_start_node(start);
        if ignore.len() > 0 {
            let ignore_rx = self.builder.regex.select(ignore);
            self.builder.top_grammar.grammars[0].greedy_skip_rx =
                Some(RegexSpec::RegexId(ignore_rx));
        }
        Ok(())
    }
}

impl Grammar {
    fn add_token_def(&mut self, loc: &Location, local_name: String, regex: &str) -> Result<()> {
        ensure!(
            !self.tokens.contains_key(&local_name),
            "duplicate token (in import): {:?}",
            local_name
        );

        let t = TokenDef {
            name: local_name,
            params: None,
            priority: None,
            expansions: Expansions(
                loc.clone(),
                vec![Alias {
                    expansion: Expansion(vec![Expr {
                        atom: Atom::Value(Value::LiteralRegex(regex.to_string(), "".to_string())),
                        op: None,
                        range: None,
                    }]),
                    alias: None,
                }],
            ),
        };
        self.tokens.insert(t.name.clone(), t.clone());
        Ok(())
    }

    fn do_statement(&mut self, loc: &Location, statement: Statement) -> Result<()> {
        match statement {
            Statement::Ignore(exp) => {
                self.ignore.push(exp);
            }
            Statement::Import { path, alias } => {
                let regex = lookup_common_regex(&path)?;
                let local_name =
                    alias.unwrap_or_else(|| path.split('.').last().unwrap().to_string());
                self.add_token_def(loc, local_name, regex)?;
            }
            Statement::MultiImport { path, names } => {
                for n in names {
                    let qname = format!("{}.{}", path, n);
                    let regex = lookup_common_regex(&qname)?;
                    self.add_token_def(loc, n.to_string(), regex)?;
                }
            }
            Statement::OverrideRule(_) => {
                bail!("override statement not supported yet");
            }
            Statement::Declare(_) => {
                bail!("declare statement not supported yet");
            }
        }
        Ok(())
    }

    fn process_item(&mut self, item: Item) -> Result<()> {
        match item {
            Item::Rule(rule) => {
                ensure!(rule.params.is_none(), "params not supported yet");
                ensure!(rule.priority.is_none(), "priority not supported yet");
                ensure!(
                    !self.rules.contains_key(&rule.name),
                    "duplicate rule: {:?}",
                    rule.name
                );
                self.rules.insert(rule.name.clone(), rule);
            }
            Item::Token(token_def) => {
                ensure!(token_def.params.is_none(), "params not supported yet");
                ensure!(token_def.priority.is_none(), "priority not supported yet");
                ensure!(
                    !self.tokens.contains_key(&token_def.name),
                    "duplicate token: {:?}",
                    token_def.name
                );
                self.tokens.insert(token_def.name.clone(), token_def);
            }
            Item::Statement(loc, statement) => {
                self.do_statement(&loc, statement)?;
            }
        }
        Ok(())
    }
}
