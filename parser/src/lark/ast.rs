use super::lexer::Location;

/// Represents an item in the grammar (rule, token, or statement).
#[derive(Debug, Clone)]
pub enum Item {
    Rule(Rule),
    Token(TokenDef),
    Statement(Location, Statement),
}

/// Represents a grammar rule.
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub cond_inline: bool,
    pub pin_terminals: bool,
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
pub struct Expansions(pub Location, pub Vec<Alias>);

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
    LiteralString(String, String),
    LiteralRegex(String, String),
    TemplateUsage { name: String, values: Vec<Value> },
}

/// Represents an operator.
#[derive(Debug, Clone)]
pub struct Op(pub String);
