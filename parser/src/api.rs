use serde::{Deserialize, Serialize};

/// This represents a collection of grammars, with a designated
/// "start" grammar at first position.
/// Grammars can refer to each other via GrammarRef nodes.
#[derive(Serialize, Deserialize, Clone)]
pub struct TopLevelGrammar {
    pub grammars: Vec<GrammarWithLexer>,
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub test_trace: bool,
}

/// cbindgen:ignore
pub const DEFAULT_CONTEXTUAL: bool = true;

#[derive(Serialize, Deserialize, Clone)]
pub struct GrammarWithLexer {
    /// The start symbol is at nodes[0]
    pub nodes: Vec<Node>,

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

#[derive(Serialize, Deserialize, Clone)]
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

/// Optional fields allowed on any Node
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct NodeProps {
    pub max_tokens: Option<usize>,
    pub name: Option<String>,
    pub capture_name: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GenGrammarOptions {
    pub grammar: GrammarId,

    /// Override sampling temperature.
    pub temperature: Option<f32>,

    #[serde(skip)]
    pub max_tokens_grm: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
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

#[derive(Serialize, Deserialize, Clone, Debug)]
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

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone, Copy, Debug)]
        #[serde(transparent)]
        pub struct $name(pub usize);
    };
}

id_type!(GrammarId);
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
        }
    }
}

impl Default for GenGrammarOptions {
    fn default() -> Self {
        GenGrammarOptions {
            grammar: GrammarId(0),
            temperature: None,
            max_tokens_grm: usize::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Parser has not emitted stop() yet.
    NotStopped,
    /// max_tokens limit on the total number of tokens has been reached.
    MaxTokensTotal,
    /// max_tokens limit on the number of tokens in the top-level parser has been reached.
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
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[repr(C)]
pub struct ParserLimits {
    pub max_items_in_row: usize,
    pub initial_lexer_fuel: u64,
    pub step_lexer_fuel: u64,
    pub max_lexer_states: usize,
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
        }
    }
}
