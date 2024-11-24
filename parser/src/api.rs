use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// This represents a collection of grammars, with a designated
/// "start" grammar at first position.
/// Grammars can refer to each other via GenGrammar nodes.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TopLevelGrammar {
    pub grammars: Vec<GrammarWithLexer>,
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub test_trace: bool,
}

/// cbindgen:ignore
pub const DEFAULT_CONTEXTUAL: bool = true;

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct GrammarWithLexer {
    /// The name of this grammar, can be used in GenGrammar nodes.
    pub name: Option<String>,

    /// The start symbol is at nodes[0]
    /// When nodes is empty, then one of json_schema or lark_grammar must be set.
    #[serde(default)]
    pub nodes: Vec<Node>,

    /// The JSON schema that the grammar should generate.
    /// When this is set, nodes and rx_nodes must be empty.
    pub json_schema: Option<Value>,

    /// The Lark grammar that the grammar should generate.
    /// When this is set, nodes and rx_nodes must be empty.
    pub lark_grammar: Option<String>,

    /// This is no longer used.
    /// When enabled, the grammar can use `Lexeme` but not `Gen`.
    /// When disabled, the grammar can use `Gen` but not `Lexeme`.
    /// `String` is allowed in either case as a shorthand for either `Lexeme` or `Gen`.
    #[serde(default)]
    pub greedy_lexer: bool,

    /// Only applies to greedy_lexer grammars.
    /// This adds a new lexeme that will be ignored when parsing.
    pub greedy_skip_rx: Option<RegexSpec>,

    /// The default value for 'contextual' in Lexeme nodes.
    pub contextual: Option<bool>,

    /// When set, the regexps can be referenced by their id (position in this list).
    #[serde(default)]
    pub rx_nodes: Vec<RegexNode>,

    /// If set, the grammar will allow skip_rx as the first lexeme.
    #[serde(default)]
    pub allow_initial_skip: bool,

    /// Normally, when a sequence of bytes is forced by grammar, it is tokenized
    /// canonically and forced as tokens.
    /// With `no_forcing`, we let the model decide on tokenization.
    /// This generally reduces both quality and speed, so should not be used
    /// outside of testing.
    #[serde(default)]
    pub no_forcing: bool,

    /// If set, the grammar will allow invalid utf8 byte sequences.
    /// Any Unicode regex will cause an error.
    #[serde(default)]
    pub allow_invalid_utf8: bool,
}

impl Debug for GrammarWithLexer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GrammarWithLexer [{} nodes]", self.nodes.len())
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub enum Node {
    // Terminals:
    /// Force generation of the specific string.
    String {
        literal: String,

        #[serde(flatten)]
        props: NodeProps,
    },
    /// Generate according to regex.
    Gen {
        #[serde(flatten)]
        data: GenOptions,

        #[serde(flatten)]
        props: NodeProps,
    },
    /// Lexeme in a greedy grammar.
    Lexeme {
        /// The regular expression that will greedily match the input.
        rx: RegexSpec,

        /// If false, all other lexemes are excluded when this lexeme is recognized.
        /// This is normal behavior for keywords in programming languages.
        /// Set to true for eg. a JSON schema with both `/"type"/` and `/"[^"]*"/` as lexemes,
        /// or for "get"/"set" contextual keywords in C#.
        /// Default value set in GrammarWithLexer.
        contextual: Option<bool>,

        /// Override sampling temperature.
        temperature: Option<f32>,

        /// When set, the lexeme will be quoted as a JSON string.
        /// For example, /[a-z"]+/ will be quoted as /([a-z]|\\")+/
        json_string: Option<bool>,

        /// It lists the allowed escape sequences, typically one of:
        /// "nrbtf\\\"u" - to allow all JSON escapes, including \u00XX for control characters
        ///     this is the default
        /// "nrbtf\\\"" - to disallow \u00XX control characters
        /// "nrt\\\"" - to also disallow unusual escapes (\f and \b)
        /// "" - to disallow all escapes
        /// Note that \uXXXX for non-control characters (code points above U+001F) are never allowed,
        /// as they never have to be quoted in JSON.
        json_allowed_escapes: Option<String>,

        /// When set and json_string is also set, "..." will not be added around the regular expression.
        json_raw: Option<bool>,

        #[serde(flatten)]
        props: NodeProps,
    },
    /// Generate according to specified grammar.
    GenGrammar {
        #[serde(flatten)]
        data: GenGrammarOptions,

        #[serde(flatten)]
        props: NodeProps,
    },
    /// Used for special tokens.
    SpecialToken {
        token: String,

        #[serde(flatten)]
        props: NodeProps,
    },

    // Non-terminals:
    /// Generate one of the options.
    Select {
        among: Vec<NodeId>,

        #[serde(flatten)]
        props: NodeProps,
    },
    /// Generate all of the nodes in sequence.
    Join {
        sequence: Vec<NodeId>,

        #[serde(flatten)]
        props: NodeProps,
    },
}

impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| std::fmt::Error)?;
        f.write_str(&s)
    }
}

/// Optional fields allowed on any Node
#[derive(Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
pub struct NodeProps {
    pub max_tokens: Option<usize>,
    pub name: Option<String>,
    pub capture_name: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct GenOptions {
    /// Regular expression matching the body of generation.
    pub body_rx: RegexSpec,

    /// The whole generation must match `body_rx + stop_rx`.
    /// Whatever matched `stop_rx` is discarded.
    /// If `stop_rx` is empty, it's assumed to be EOS.
    pub stop_rx: RegexSpec,

    /// When set, the string matching `stop_rx` will be output as a capture
    /// with the given name.
    pub stop_capture_name: Option<String>,

    /// Lazy gen()s take the shortest match. Non-lazy take the longest.
    /// If not specified, the gen() is lazy if stop_rx is non-empty.
    pub lazy: Option<bool>,

    /// Override sampling temperature.
    pub temperature: Option<f32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GenGrammarOptions {
    pub grammar: GrammarId,

    /// Override sampling temperature.
    pub temperature: Option<f32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Hash, PartialEq, Eq)]
pub enum RegexNode {
    /// Intersection of the regexes
    And(Vec<RegexId>),
    /// Union of the regexes
    Or(Vec<RegexId>),
    /// Concatenation of the regexes
    Concat(Vec<RegexId>),
    /// Matches the regex; should be at the end of the main regex.
    /// The length of the lookahead can be recovered from the engine.
    LookAhead(RegexId),
    /// Matches everything the regex doesn't match.
    /// Can lead to invalid utf8.
    Not(RegexId),
    /// Repeat the regex at least min times, at most max times
    Repeat(RegexId, u32, Option<u32>),
    /// Matches the empty string. Same as Concat([]).
    EmptyString,
    /// Matches nothing. Same as Or([]).
    NoMatch,
    /// Compile the regex using the regex_syntax crate
    Regex(String),
    /// Matches this string only
    Literal(String),
    /// Matches this string of bytes only. Can lead to invalid utf8.
    ByteLiteral(Vec<u8>),
    /// Matches this byte only. If byte is not in 0..127, it may lead to invalid utf8
    Byte(u8),
    /// Matches any byte in the set, expressed as bitset.
    /// Can lead to invalid utf8 if the set is not a subset of 0..127
    ByteSet(Vec<u32>),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(untagged)]
pub enum RegexSpec {
    RegexId(RegexId),
    Regex(String),
}

impl RegexSpec {
    pub fn is_missing(&self) -> bool {
        match self {
            RegexSpec::RegexId(_) => false,
            RegexSpec::Regex(s) => s.is_empty(),
        }
    }
}

#[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone, Debug)]
#[serde(untagged)]
pub enum GrammarId {
    Index(usize),
    Name(String),
}

impl GrammarId {
    pub fn to_index(&self) -> Option<usize> {
        match self {
            GrammarId::Index(i) => Some(*i),
            GrammarId::Name(_) => None,
        }
    }
}

impl Display for GrammarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrammarId::Index(i) => write!(f, "@#{}", i),
            GrammarId::Name(s) => write!(f, "@{}", s),
        }
    }
}

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone, Copy, Debug)]
        #[serde(transparent)]
        pub struct $name(pub usize);
    };
}

id_type!(NodeId);
id_type!(RegexId);

impl Node {
    pub fn node_props(&self) -> &NodeProps {
        match self {
            Node::String { props, .. } => props,
            Node::Gen { props, .. } => props,
            Node::Lexeme { props, .. } => props,
            Node::GenGrammar { props, .. } => props,
            Node::Select { props, .. } => props,
            Node::Join { props, .. } => props,
            Node::SpecialToken { props, .. } => props,
        }
    }
}

impl Default for GenGrammarOptions {
    fn default() -> Self {
        GenGrammarOptions {
            grammar: GrammarId::Index(0),
            temperature: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Parser has not emitted stop() yet.
    NotStopped,
    /// max_tokens limit on the total number of tokens has been reached.
    MaxTokensTotal,
    /// max_tokens limit on the number of tokens in the top-level parser has been reached. (no longer used)
    MaxTokensParser,
    /// Top-level parser indicates that no more bytes can be added.
    NoExtension,
    /// Top-level parser indicates that no more bytes can be added, however it was recognized late.
    NoExtensionBias,
    /// Top-level parser allowed EOS (as it was in an accepting state), and EOS was generated.
    EndOfSentence,
    /// Something went wrong with creating a nested parser.
    InternalError,
    /// The lexer is too complex
    LexerTooComplex,
    /// The parser is too complex
    ParserTooComplex,
}

impl StopReason {
    pub fn to_string(&self) -> String {
        serde_json::to_value(self)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    }

    pub fn is_ok(&self) -> bool {
        matches!(
            self,
            StopReason::NotStopped
                | StopReason::EndOfSentence
                | StopReason::NoExtension
                | StopReason::NoExtensionBias
        )
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[repr(C)]
pub struct ParserLimits {
    /// For non-ambiguous grammars, this is the maximum "branching factor" of the grammar.
    /// For ambiguous grammars, this might get hit much quicker.
    /// Default: 200
    pub max_items_in_row: usize,

    /// How much "fuel" are we willing to spend to build initial lexer regex AST nodes.
    /// Default: 1_000_000 (~20ms)
    pub initial_lexer_fuel: u64,

    /// Maximum lexer fuel for computation of the whole token mask.
    /// Default: 500_000 (~10ms)
    pub step_lexer_fuel: u64,

    /// Number of Earley items created for the whole token mask.
    /// Default: 100_000 (~3ms)
    pub step_max_items: usize,

    /// Maximum number of lexer states.
    /// Default: 10_000
    pub max_lexer_states: usize,

    /// Maximum size of the grammar (symbols in productions)
    /// Default: 500_000 (a few megabytes of JSON)
    pub max_grammar_size: usize,
}

impl Default for ParserLimits {
    fn default() -> Self {
        Self {
            max_items_in_row: 200,
            initial_lexer_fuel: 1_000_000, // fhir schema => 500k
            step_lexer_fuel: 500_000,      // 500k => 10ms
            max_lexer_states: 10_000,      // ?
            max_grammar_size: 500_000,     // fhir schema => 200k
            step_max_items: 100_000,       //
        }
    }
}

impl TopLevelGrammar {
    pub fn from_regex(rx: RegexNode) -> Self {
        Self::from_grammar(GrammarWithLexer::from_regex(rx))
    }

    pub fn from_lark(lark_grammar: String) -> Self {
        Self::from_grammar(GrammarWithLexer::from_lark(lark_grammar))
    }

    pub fn from_grammar(grammar: GrammarWithLexer) -> Self {
        TopLevelGrammar {
            grammars: vec![grammar],
            max_tokens: None,
            test_trace: false,
        }
    }
}

impl GrammarWithLexer {
    pub fn from_lark(lark_grammar: String) -> Self {
        GrammarWithLexer {
            name: Some("lark_grammar".to_string()),
            lark_grammar: Some(lark_grammar),
            greedy_lexer: true,
            ..GrammarWithLexer::default()
        }
    }

    pub fn from_regex(rx: RegexNode) -> Self {
        GrammarWithLexer {
            name: Some("regex_grammar".to_string()),
            nodes: vec![Node::Lexeme {
                rx: RegexSpec::RegexId(RegexId(0)),
                contextual: None,
                temperature: None,
                props: NodeProps::default(),
                json_string: None,
                json_allowed_escapes: None,
                json_raw: None,
            }],
            greedy_lexer: true,
            rx_nodes: vec![rx],
            ..Default::default()
        }
    }
}
