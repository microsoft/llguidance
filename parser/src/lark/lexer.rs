use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use derivre::{RegexAst, RegexBuilder};

use crate::earley::{
    lexer::{Lexer, LexerResult},
    lexerspec::{LexemeIdx, LexerSpec},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    KwIgnore,
    KwImport,
    KwOverride,
    KwDeclare,
    Colon,
    Comma,
    Dot,
    DotDot,
    Arrow,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Tilde,
    // regexps
    Op,
    String,
    Regexp,
    Rule,
    Token,
    Number,
    Newline,
    VBar,
    // special
    SKIP,
    EOF,
}

/// Represents a lexeme with its token type, value, and position.
#[derive(Debug, Clone)]
pub struct Lexeme {
    token: Token,
    value: String,
    line: usize,
    column: usize,
}

/// Represents an item in the grammar (rule, token, or statement).
#[derive(Debug, Clone)]
pub enum Item {
    Rule(Rule),
    Token(TokenDef),
    Statement(Statement),
}

/// Represents a grammar rule.
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub params: Option<RuleParams>,
    pub priority: Option<i32>,
    pub expansions: Expansions,
}

/// Represents a token definition.
#[derive(Debug, Clone)]
pub struct TokenDef {
    pub name: String,
    pub params: Option<TokenParams>,
    pub priority: Option<i32>,
    pub expansions: Expansions,
}

/// Represents different types of statements.
#[derive(Debug, Clone)]
pub enum Statement {
    Ignore(Expansions),
    Import {
        path: ImportPath,
        alias: Option<String>,
    },
    MultiImport {
        path: ImportPath,
        names: Vec<String>,
    },
    OverrideRule(Box<Rule>),
    Declare(Vec<String>),
}

/// Represents an import path.
#[derive(Debug, Clone)]
pub struct ImportPath(pub Vec<String>);

/// Represents parameters for a rule.
#[derive(Debug, Clone)]
pub struct RuleParams(pub Vec<String>);

/// Represents parameters for a token.
#[derive(Debug, Clone)]
pub struct TokenParams(pub Vec<String>);

/// Represents a list of expansions.
#[derive(Debug, Clone)]
pub struct Expansions(pub Vec<Alias>);

/// Represents an alias in the grammar.
#[derive(Debug, Clone)]
pub struct Alias {
    pub expansion: Expansion,
    pub alias: Option<String>,
}

/// Represents an expansion consisting of expressions.
#[derive(Debug, Clone)]
pub struct Expansion(pub Vec<Expr>);

/// Represents an expression.
#[derive(Debug, Clone)]
pub struct Expr {
    pub atom: Atom,
    pub op: Option<Op>,
    pub range: Option<(i32, i32)>,
}

/// Represents an atom in the grammar.
#[derive(Debug, Clone)]
pub enum Atom {
    Group(Expansions),
    Maybe(Expansions),
    Value(Value),
}

/// Represents different values in the grammar.
#[derive(Debug, Clone)]
pub enum Value {
    LiteralRange(String, String),
    Name(String),
    Literal(String),
    TemplateUsage { name: String, values: Vec<Value> },
}

/// Represents an operator.
#[derive(Debug, Clone)]
pub struct Op(pub String);

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
            Ok(Item::Statement(self.parse_statement()?))
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
        self.expect_token(Token::Colon)?;
        let expansions = self.parse_expansions()?;
        Ok(Rule {
            name,
            params,
            priority,
            expansions,
        })
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
        self.expect_token(Token::Colon)?;
        let expansions = self.parse_expansions()?;
        Ok(TokenDef {
            name,
            params,
            priority,
            expansions,
        })
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
            } else {
                Ok(Statement::MultiImport {
                    path: import_path,
                    names: self.parse_name_list()?,
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
        let mut aliases = Vec::new();
        aliases.push(self.parse_alias()?);
        while self.match_vbar() {
            aliases.push(self.parse_alias()?);
        }
        Ok(Expansions(aliases))
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

    /// Parses a value.
    fn parse_value(&mut self) -> Result<Value> {
        if let Some(string1) = self.match_token_with_value(Token::String) {
            if self.match_token(Token::DotDot) {
                let string2 = self.expect_token(Token::String)?.value;
                Ok(Value::LiteralRange(string1.value.clone(), string2))
            } else {
                Ok(Value::Literal(string1.value.clone()))
            }
        } else if let Some(regexp_token) = self.match_token_with_value(Token::Regexp) {
            Ok(Value::Literal(regexp_token.value.clone()))
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
    fn parse_import_path(&mut self) -> Result<ImportPath> {
        let mut names = Vec::new();
        if self.match_token(Token::Dot) {
            names.push(".".to_string());
        }
        names.push(self.parse_name()?);
        while self.match_token(Token::Dot) {
            names.push(self.parse_name()?);
        }
        Ok(ImportPath(names))
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

impl Token {
    const LITERAL_TOKENS: &'static [(Token, &'static str)] = &[
        (Token::Arrow, "->"),
        (Token::Colon, ":"),
        (Token::Comma, ","),
        (Token::Dot, "."),
        (Token::DotDot, ".."),
        (Token::KwDeclare, "%declare"),
        (Token::KwIgnore, "%ignore"),
        (Token::KwImport, "%import"),
        (Token::KwOverride, "%override"),
        (Token::LParen, "("),
        (Token::RParen, ")"),
        (Token::LBrace, "{"),
        (Token::RBrace, "}"),
        (Token::LBracket, "["),
        (Token::RBracket, "]"),
        (Token::Tilde, "~"),
        (Token::VBar, "|"),
    ];

    const REGEX_TOKENS: &'static [(Token, &'static str)] = &[
        (Token::Op, r"[+*?]"),
        (Token::Rule, r"!?[_?]?[a-z][_a-z0-9]*"),
        (Token::Token, r"_?[A-Z][_A-Z0-9]*"),
        // use JSON string syntax
        (
            Token::String,
            r#""(\\([\"\\\/bfnrt]|u[a-fA-F0-9]{4})|[^\"\\\x00-\x1F\x7F])*"(i|)"#,
        ),
        (Token::Regexp, r#"/(\\.|[^/\\])+/[imslux]*"#),
        (Token::Number, r#"[+-]?[0-9]+"#),
        (Token::Newline, r"(\r?\n)+[ \t]*"),
    ];
}

pub fn lex_lark(input: &str) -> Result<Vec<Lexeme>> {
    let builder = RegexBuilder::new();
    let comment_or_ws = r"((#|//)[^\n]*)|[ \t]+".to_string();
    let mut spec = LexerSpec::new(builder, RegexAst::Regex(comment_or_ws)).unwrap();
    let mut lexeme_idx_to_token = HashMap::new();
    lexeme_idx_to_token.insert(LexemeIdx::SKIP, Token::SKIP);
    for (token, literal) in Token::LITERAL_TOKENS {
        let l = spec
            .add_simple_literal(format!("{:?}", token), *literal, false)
            .unwrap();
        lexeme_idx_to_token.insert(l, *token);
    }
    for (token, regexp) in Token::REGEX_TOKENS {
        let l = spec
            .add_greedy_lexeme(
                format!("{:?}", token),
                RegexAst::Regex(regexp.to_string()),
                false,
                None,
            )
            .unwrap();
        lexeme_idx_to_token.insert(l, *token);
    }
    let mut lexer = Lexer::from(&spec, &mut Default::default()).unwrap();
    let all_lexemes = spec.all_lexemes();
    let state0 = lexer.start_state(&all_lexemes, None);
    let mut line_no = 1;
    let mut column_no = 1;
    let mut curr_lexeme = Lexeme {
        token: Token::EOF,
        value: String::new(),
        line: 1,
        column: 1,
    };
    let mut state = state0;
    let mut lexemes = Vec::new();
    let mut start_idx = 0;

    let input = format!("{}\n", input);
    let input_bytes = input.as_bytes();
    for idx in 0..=input_bytes.len() {
        let mut b = b'\n';
        let res = if idx == input_bytes.len() {
            lexer.force_lexeme_end(state)
        } else {
            b = input_bytes[idx];
            lexer.advance(state, b, false)
        };

        match res {
            LexerResult::Error => {
                bail!("{}({}): lexer error", line_no, column_no);
            }
            LexerResult::State(s, _) => {
                state = s;
            }
            LexerResult::Lexeme(p) => {
                let transition_byte = if p.byte_next_row { p.byte } else { None };

                let token = lexeme_idx_to_token[&p.idx];
                curr_lexeme.token = token;
                let end_idx = if p.byte_next_row || p.byte.is_none() {
                    idx
                } else {
                    idx + 1
                };
                curr_lexeme.value = input[start_idx..end_idx].to_string();
                start_idx = end_idx;

                // println!("lex: {:?}", curr_lexeme);

                if curr_lexeme.token != Token::SKIP {
                    lexemes.push(curr_lexeme.clone());
                }

                state = lexer.start_state(&all_lexemes, transition_byte);

                curr_lexeme.value.clear();
                curr_lexeme.line = line_no;
                curr_lexeme.column = column_no;
            }
        }

        if b == b'\n' {
            line_no += 1;
            column_no = 1;
        } else {
            column_no += 1;
        }
    }

    Ok(lexemes)
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

#[allow(dead_code)]
pub fn test_lex_lark() {
    lex_lark(LARK_GRAMMAR).unwrap();
}

const LARK_GRAMMAR: &str = r#"
# Lark grammar of Lark's syntax
# Note: Lark is not bootstrapped, its parser is implemented in load_grammar.py

start: (_item? _NL)* _item?

_item: rule
     | token
     | statement

rule: RULE rule_params priority? ":" expansions
token: TOKEN token_params priority? ":" expansions

rule_params: ["{" RULE ("," RULE)* "}"]
token_params: ["{" TOKEN ("," TOKEN)* "}"]

priority: "." NUMBER

statement: "%ignore" expansions                    -> ignore
         | "%import" import_path ["->" name]       -> import
         | "%import" import_path name_list         -> multi_import
         | "%override" rule                        -> override_rule
         | "%declare" name+                        -> declare

!import_path: "."? name ("." name)*
name_list: "(" name ("," name)* ")"

?expansions: alias (_VBAR alias)*

?alias: expansion ["->" RULE]

?expansion: expr*

?expr: atom [OP | "~" NUMBER [".." NUMBER]]

?atom: "(" expansions ")"
     | "[" expansions "]" -> maybe
     | value

?value: STRING ".." STRING -> literal_range
      | name
      | (REGEXP | STRING) -> literal
      | name "{" value ("," value)* "}" -> template_usage

name: RULE
    | TOKEN

_VBAR: _NL? "|"
OP: /[+*]|[?](?![a-z])/
RULE: /!?[_?]?[a-z][_a-z0-9]*/
TOKEN: /_?[A-Z][_A-Z0-9]*/
STRING: _STRING "i"?
REGEXP: /\/(?!\/)(\\\/|\\\\|[^\/])*?\/[imslux]*/
_NL: /(\r?\n)+\s*/

_STRING_INNER: /.*?/
_STRING_ESC_INNER: _STRING_INNER /(?<!\\)(\\\\)*?/

_STRING : "\"" _STRING_ESC_INNER "\""

WS_INLINE: (" "|/\t/)+
DIGIT: "0".."9"
INT: DIGIT+
NUMBER: ["+"|"-"] INT

COMMENT: /\s*/ "//" /[^\n]/* | /\s*/ "@" /[^\n]/*

%ignore WS_INLINE
%ignore COMMENT
"#;
