use super::{
    ast::*,
    lexer::{lex_lark, Lexeme, Location, Token},
};
use anyhow::{anyhow, bail, ensure, Result};

/// The parser struct that holds the tokens and current position.
pub struct Parser {
    tokens: Vec<Lexeme>,
    pos: usize,
}

impl Parser {
    /// Creates a new parser instance.
    pub fn new(tokens: Vec<Lexeme>) -> Self {
        Parser { tokens, pos: 0 }
    }

    /// Parses the start symbol of the grammar.
    pub fn parse_start(&mut self) -> Result<Vec<Item>> {
        let mut items = Vec::new();
        while !self.is_at_end() {
            self.consume_newlines();
            if self.is_at_end() {
                break;
            }
            items.push(self.parse_item()?);
            self.consume_newlines();
        }
        Ok(items)
    }

    /// Parses an item (rule, token, or statement).
    fn parse_item(&mut self) -> Result<Item> {
        if self.has_token(Token::Rule) {
            Ok(Item::Rule(self.parse_rule()?))
        } else if self.has_token(Token::Token) {
            Ok(Item::Token(self.parse_token_def()?))
        } else {
            let loc = self.location();
            Ok(Item::Statement(loc, self.parse_statement()?))
        }
    }

    fn location(&self) -> Location {
        if let Some(t) = self.peek_token() {
            Location {
                line: t.line,
                column: t.column,
            }
        } else {
            Location { line: 0, column: 0 }
        }
    }

    /// Parses a rule definition.
    fn parse_rule(&mut self) -> Result<Rule> {
        let name = self.expect_token(Token::Rule)?.value;
        let params = if self.has_token(Token::LBrace) {
            Some(self.parse_rule_params()?)
        } else {
            None
        };
        let priority = if self.has_token(Token::Dot) {
            Some(self.parse_priority()?)
        } else {
            None
        };
        let (name, pin_terminals) = if name.starts_with("!") {
            (name[1..].to_string(), true)
        } else {
            (name, false)
        };
        let (name, cond_inline) = if name.starts_with("?") {
            (name[1..].to_string(), true)
        } else {
            (name, false)
        };

        let mut rule = Rule {
            name,
            pin_terminals,
            cond_inline,
            params,
            priority,
            expansions: Expansions(self.location(), Vec::new()),
            stop: None,
            max_tokens: None,
        };

        if self.has_token(Token::LBracket) {
            self.parse_attributes(&mut rule)?;
        }

        self.expect_token(Token::Colon)?;
        rule.expansions = self.parse_expansions()?;
        Ok(rule)
    }

    /// Parses a token definition.
    fn parse_token_def(&mut self) -> Result<TokenDef> {
        let name = self.expect_token(Token::Token)?.value;
        let params = if self.has_token(Token::LBrace) {
            Some(self.parse_token_params()?)
        } else {
            None
        };
        let priority = if self.has_token(Token::Dot) {
            Some(self.parse_priority()?)
        } else {
            None
        };

        let mut token_def = TokenDef {
            name,
            params,
            priority,
            expansions: Expansions(self.location(), Vec::new()),
        };

        self.expect_token(Token::Colon)?;
        token_def.expansions = self.parse_expansions()?;
        Ok(token_def)
    }

    /// Parses attributes inside square brackets.
    fn parse_attributes(&mut self, rule: &mut Rule) -> Result<()> {
        self.expect_token(Token::LBracket)?;
        while !self.has_token(Token::RBracket) {
            let key = self.expect_token(Token::Rule)?.value;
            self.expect_token(Token::Equals)?;
            match key.as_str() {
                "stop" => {
                    let value = self.parse_value()?;
                    rule.stop = Some(value);
                }
                "max_tokens" => {
                    let value = self.expect_token(Token::Number)?.value.parse::<usize>()?;
                    rule.max_tokens = Some(value);
                }
                _ => bail!("Unknown attribute: {}", key),
            }
            if self.has_token(Token::Comma) {
                self.expect_token(Token::Comma)?;
            } else {
                break;
            }
        }
        self.expect_token(Token::RBracket)?;
        Ok(())
    }

    /// Parses a statement.
    fn parse_statement(&mut self) -> Result<Statement> {
        if self.match_token(Token::KwIgnore) {
            let expansions = self.parse_expansions()?;
            Ok(Statement::Ignore(expansions))
        } else if self.match_token(Token::KwImport) {
            let import_path = self.parse_import_path()?;
            if self.match_token(Token::Arrow) {
                let name = self.parse_name()?;
                Ok(Statement::Import {
                    path: import_path,
                    alias: Some(name),
                })
            } else if self.has_token(Token::LParen) {
                Ok(Statement::MultiImport {
                    path: import_path,
                    names: self.parse_name_list()?,
                })
            } else {
                Ok(Statement::Import {
                    path: import_path,
                    alias: None,
                })
            }
        } else if self.match_token(Token::KwOverride) {
            let rule = self.parse_rule()?;
            Ok(Statement::OverrideRule(Box::new(rule)))
        } else if self.match_token(Token::KwDeclare) {
            let mut names = Vec::new();
            while let Ok(name) = self.parse_name() {
                names.push(name);
            }
            if names.is_empty() {
                bail!("Expected at least one name after %declare")
            }
            Ok(Statement::Declare(names))
        } else {
            bail!("expecting rule, token or statement")
        }
    }

    /// Parses rule parameters.
    fn parse_rule_params(&mut self) -> Result<RuleParams> {
        if !self.match_token(Token::LBrace) {
            bail!("Expected '{{' in rule parameters")
        }
        let mut params = Vec::new();
        let name = self.expect_token(Token::Rule)?.value;
        params.push(name);
        while self.match_token(Token::Comma) {
            let name = self.expect_token(Token::Rule)?.value;
            params.push(name);
        }
        self.expect_token(Token::RBrace)?;
        Ok(RuleParams(params))
    }

    /// Parses token parameters.
    fn parse_token_params(&mut self) -> Result<TokenParams> {
        if !self.match_token(Token::LBrace) {
            bail!("Expected '{{' in token parameters")
        }
        let mut params = Vec::new();
        let name = self.expect_token(Token::Token)?.value;
        params.push(name);
        while self.match_token(Token::Comma) {
            let name = self.expect_token(Token::Token)?.value;
            params.push(name);
        }
        self.expect_token(Token::RBrace)?;
        Ok(TokenParams(params))
    }

    /// Parses priority.
    fn parse_priority(&mut self) -> Result<i32> {
        if !self.match_token(Token::Dot) {
            bail!("Expected '.' in priority")
        }
        let number = self.expect_token(Token::Number)?.value.parse::<i32>()?;
        Ok(number)
    }

    /// Parses expansions.
    fn parse_expansions(&mut self) -> Result<Expansions> {
        let loc = self.location();
        let mut aliases = Vec::new();
        aliases.push(self.parse_alias()?);
        while self.match_vbar() {
            aliases.push(self.parse_alias()?);
        }
        Ok(Expansions(loc, aliases))
    }

    fn match_vbar(&mut self) -> bool {
        if self.match_token(Token::VBar) {
            return true;
        }
        let p0 = self.pos;
        if self.match_token(Token::Newline) {
            if self.match_token(Token::VBar) {
                return true;
            }
        }
        self.pos = p0;
        false
    }

    /// Parses an alias.
    fn parse_alias(&mut self) -> Result<Alias> {
        let expansion = self.parse_expansion()?;
        let alias = if self.match_token(Token::Arrow) {
            Some(self.expect_token(Token::Rule)?.value)
        } else {
            None
        };
        Ok(Alias { expansion, alias })
    }

    /// Parses an expansion.
    fn parse_expansion(&mut self) -> Result<Expansion> {
        let mut exprs = Vec::new();
        loop {
            if self.has_token(Token::Newline)
                || self.has_token(Token::VBar)
                || self.has_token(Token::Arrow)
                || self.has_token(Token::RBrace)
                || self.has_token(Token::RParen)
                || self.has_token(Token::RBracket)
            {
                break;
            }
            exprs.push(self.parse_expr()?);
        }
        Ok(Expansion(exprs))
    }

    /// Parses an expression.
    fn parse_expr(&mut self) -> Result<Expr> {
        let atom = self.parse_atom()?;
        let mut op = None;
        let mut range = None;
        if let Some(op_token) = self.match_token_with_value(Token::Op) {
            op = Some(Op(op_token.value.clone()));
        } else if self.match_token(Token::Tilde) {
            let start_num = self.expect_token(Token::Number)?.value.parse::<i32>()?;
            let end_num = if self.match_token(Token::DotDot) {
                Some(self.expect_token(Token::Number)?.value.parse::<i32>()?)
            } else {
                None
            };
            range = Some((start_num, end_num.unwrap_or(start_num)));
        }
        Ok(Expr { atom, op, range })
    }

    /// Parses an atom.
    fn parse_atom(&mut self) -> Result<Atom> {
        if self.match_token(Token::LParen) {
            let expansions = self.parse_expansions()?;
            self.expect_token(Token::RParen)?;
            Ok(Atom::Group(expansions))
        } else if self.match_token(Token::LBracket) {
            let expansions = self.parse_expansions()?;
            self.expect_token(Token::RBracket)?;
            Ok(Atom::Maybe(expansions))
        } else {
            Ok(Atom::Value(self.parse_value()?))
        }
    }

    fn parse_string(&self, string1: &Lexeme) -> Result<(String, String)> {
        let inner = string1.value.clone();
        let (inner, flags) = if inner.ends_with('i') {
            (inner[..inner.len() - 1].to_string(), "i".to_string())
        } else {
            (inner, "".to_string())
        };
        let inner =
            serde_json::from_str(&inner).map_err(|e| anyhow!("error parsing string: {e}"))?;
        Ok((inner, flags))
    }

    fn parse_simple_string(&self, string1: &Lexeme) -> Result<String> {
        let (inner, flags) = self.parse_string(string1)?;
        ensure!(flags.is_empty(), "flags not allowed in this context");
        Ok(inner)
    }

    /// Parses a value.
    fn parse_value(&mut self) -> Result<Value> {
        if let Some(string1) = self.match_token_with_value(Token::String) {
            if self.match_token(Token::DotDot) {
                let string2 = self.expect_token(Token::String)?;
                Ok(Value::LiteralRange(
                    self.parse_simple_string(&string1)?,
                    self.parse_simple_string(&string2)?,
                ))
            } else {
                let (inner, flags) = self.parse_string(&string1)?;
                Ok(Value::LiteralString(inner, flags))
            }
        } else if let Some(regexp_token) = self.match_token_with_value(Token::Regexp) {
            let inner = regexp_token.value;
            let last_slash_idx = inner.rfind('/').unwrap();
            let flags = inner[last_slash_idx + 1..].to_string();
            let regex = inner[1..last_slash_idx].to_string();
            Ok(Value::LiteralRegex(regex, flags))
        } else if let Some(grammar_ref) = self.match_token_with_value(Token::GrammarRef) {
            Ok(Value::GrammarRef(grammar_ref.value.clone()))
        } else if let Some(special_token) = self.match_token_with_value(Token::SpecialToken) {
            Ok(Value::SpecialToken(special_token.value.clone()))
        } else if let Some(name_token) = self
            .match_token_with_value(Token::Rule)
            .or_else(|| self.match_token_with_value(Token::Token))
        {
            if self.match_token(Token::LBrace) {
                let mut values = Vec::new();
                values.push(self.parse_value()?);
                while self.match_token(Token::Comma) {
                    values.push(self.parse_value()?);
                }
                self.expect_token(Token::RBrace)?;
                Ok(Value::TemplateUsage {
                    name: name_token.value.clone(),
                    values,
                })
            } else {
                Ok(Value::Name(name_token.value.clone()))
            }
        } else {
            bail!("Expected value")
        }
    }

    /// Parses an import path.
    fn parse_import_path(&mut self) -> Result<String> {
        let mut names = String::new();
        if self.match_token(Token::Dot) {
            names.push('.');
        }
        names.push_str(&self.parse_name()?);
        while self.match_token(Token::Dot) {
            names.push('.');
            names.push_str(&self.parse_name()?);
        }
        Ok(names)
    }

    /// Parses a name (RULE or TOKEN).
    fn parse_name(&mut self) -> Result<String> {
        if let Some(token) = self.match_token_with_value(Token::Rule) {
            Ok(token.value.clone())
        } else if let Some(token) = self.match_token_with_value(Token::Token) {
            Ok(token.value.clone())
        } else {
            bail!("Expected name (RULE or TOKEN)")
        }
    }

    /// Parses a list of names.
    fn parse_name_list(&mut self) -> Result<Vec<String>> {
        if !self.match_token(Token::LParen) {
            bail!("Expected '(' in name list")
        }
        let mut names = Vec::new();
        names.push(self.parse_name()?);
        while self.match_token(Token::Comma) {
            names.push(self.parse_name()?);
        }
        self.expect_token(Token::RParen)?;
        Ok(names)
    }

    fn has_token(&self, token: Token) -> bool {
        if let Some(lexeme) = self.peek_token() {
            lexeme.token == token
        } else {
            false
        }
    }

    /// Matches a specific token.
    fn match_token(&mut self, expected: Token) -> bool {
        if let Some(token) = self.peek_token() {
            if token.token == expected {
                self.advance();
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Expects a specific token, or returns an error.
    fn expect_token(&mut self, expected: Token) -> Result<Lexeme> {
        if let Some(token) = self.peek_token() {
            if token.token == expected {
                let r = token.clone();
                self.advance();
                Ok(r)
            } else {
                bail!("Expected token {:?}, found {:?}", expected, token.token)
            }
        } else {
            bail!("Expected token {:?}, found end of input", expected)
        }
    }

    /// Matches a token and returns it if it matches the expected token.
    fn match_token_with_value(&mut self, expected: Token) -> Option<Lexeme> {
        if let Some(token) = self.peek_token() {
            if token.token == expected {
                let r = token.clone();
                self.advance();
                Some(r)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Consumes any newlines.
    fn consume_newlines(&mut self) {
        while let Some(token) = self.peek_token() {
            if token.token == Token::Newline {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Checks if the parser has reached the end of the tokens.
    fn is_at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// Peeks at the next token without advancing.
    fn peek_token(&self) -> Option<&Lexeme> {
        self.tokens.get(self.pos)
    }

    /// Advances to the next token.
    fn advance(&mut self) {
        if !self.is_at_end() {
            self.pos += 1;
        }
    }
}

pub fn parse_lark(input: &str) -> Result<Vec<Item>> {
    let tokens = lex_lark(input)?;
    let mut parser = Parser::new(tokens);
    parser.parse_start().map_err(|e| {
        if let Some(tok) = parser.peek_token() {
            anyhow!(
                "{}({}): {} (at {:?} ({:?}))",
                tok.line,
                tok.column,
                e,
                tok.value,
                tok.token
            )
        } else {
            anyhow!("at EOF: {}", e)
        }
    })
}
