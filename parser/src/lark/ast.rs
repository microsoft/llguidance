use super::lexer::Location;

/// Represents an item in the grammar (rule, token, or statement).
#[derive(Debug, Clone)]
pub enum Item {
    Rule(Rule),
    Token(TokenDef),
    Statement(Location, Statement),
}

impl Item {
    pub fn location(&self) -> &Location {
        match self {
            Item::Rule(rule) => &rule.expansions.0,
            Item::Token(token) => &token.expansions.0,
            Item::Statement(loc, _) => loc,
        }
    }
}

/// Represents a grammar rule.
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    #[allow(dead_code)]
    pub cond_inline: bool,
    #[allow(dead_code)]
    pub pin_terminals: bool,
    pub params: Option<RuleParams>,
    pub priority: Option<i32>,
    pub expansions: Expansions,

    pub stop: Option<Value>,
    pub max_tokens: Option<usize>,
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
        path: String,
        alias: Option<String>,
    },
    MultiImport {
        path: String,
        names: Vec<String>,
    },
    #[allow(dead_code)]
    OverrideRule(Box<Rule>),
    #[allow(dead_code)]
    Declare(Vec<String>),
}

/// Represents parameters for a rule.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RuleParams(pub Vec<String>);

/// Represents parameters for a token.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TokenParams(pub Vec<String>);

/// Represents a list of expansions.
#[derive(Debug, Clone)]
pub struct Expansions(pub Location, pub Vec<Alias>);

/// Represents an alias in the grammar.
#[derive(Debug, Clone)]
pub struct Alias {
    pub expansion: Expansion,
    #[allow(dead_code)]
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
    LiteralString(String, String),
    LiteralRegex(String, String),
    GrammarRef(String),
    SpecialToken(String),
    #[allow(dead_code)]
    TemplateUsage {
        name: String,
        values: Vec<Value>,
    },
}

/// Represents an operator.
#[derive(Debug, Clone)]
pub struct Op(pub String);
