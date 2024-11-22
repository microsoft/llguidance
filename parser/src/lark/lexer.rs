use std::{collections::HashMap, fmt::Display};

use anyhow::{bail, Result};
use derivre::{RegexAst, RegexBuilder};

use crate::earley::{
    lexer::{Lexer, LexerResult},
    lexerspec::{LexemeIdx, LexerSpec},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    KwIgnore,
    KwImport,
    KwOverride,
    KwDeclare,
    Colon,
    Equals,
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
    Op, // + * ?
    String,
    Regexp,
    Rule,
    Token,
    Number,
    Newline,
    VBar,
    SpecialToken, // <something>
    GrammarRef, // @grammar_id or @7
    // special
    SKIP,
    EOF,
}

/// Represents a lexeme with its token type, value, and position.
#[derive(Debug, Clone)]
pub struct Lexeme {
    pub token: Token,
    pub value: String,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone)]
pub struct Location {
    pub line: usize,
    pub column: usize,
}

impl Location {
    pub fn augment(&self, err: impl Display) -> anyhow::Error {
        let err = err.to_string();
        if err.starts_with("at ") {
            // don't add more location info
            anyhow::anyhow!("{err}")
        } else {
            anyhow::anyhow!("at {}({}): {}", self.line, self.column, err)
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
        (Token::Equals, "="),
    ];

    const REGEX_TOKENS: &'static [(Token, &'static str)] = &[
        (Token::Op, r"[+*?]"),
        (Token::Rule, r"!?[_?]?[a-z][_a-z0-9\-]*"),
        (Token::Token, r"_?[A-Z][_A-Z0-9]*"),
        // use JSON string syntax
        (
            Token::String,
            r#""(\\([\"\\\/bfnrt]|u[a-fA-F0-9]{4})|[^\"\\\x00-\x1F\x7F])*"(i|)"#,
        ),
        (Token::Regexp, r#"/(\\.|[^/\\])+/[imslux]*"#),
        (Token::Number, r#"[+-]?[0-9]+"#),
        (Token::Newline, r"(\r?\n)+[ \t]*"),
        (Token::SpecialToken, r"<[^<>\s]+>"),
        (Token::GrammarRef, r"@[a-zA-Z0-9_\-]+"),
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
    let mut lexer = Lexer::from_internal(&spec).unwrap();
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
