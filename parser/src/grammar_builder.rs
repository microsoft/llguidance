use anyhow::{bail, ensure, Result};
use derivre::RegexAst;
use hashbrown::HashMap;
use std::sync::atomic::AtomicU32;

use crate::api::{
    GenGrammarOptions, GenOptions, GrammarWithLexer, Node, NodeId, NodeProps, RegexId, RegexNode,
    RegexSpec, TopLevelGrammar,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct NodeRef {
    idx: usize,
    grammar_id: u32,
}

const K: usize = 4;

pub struct GrammarBuilder {
    pub top_grammar: TopLevelGrammar,
    placeholder: Node,
    strings: HashMap<String, NodeRef>,
    curr_grammar_id: u32,
    node_refs: HashMap<String, NodeRef>,
    nodes: Vec<Node>,
    pub regex: RegexBuilder,
    at_most_cache: HashMap<(NodeRef, usize), NodeRef>,
    repeat_exact_cache: HashMap<(NodeRef, usize), NodeRef>,
}

pub struct RegexBuilder {
    node_ids: HashMap<RegexNode, RegexId>,
    nodes: Vec<RegexNode>,
}

impl RegexBuilder {
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            node_ids: HashMap::new(),
        }
    }

    pub fn add_ast(&mut self, ast: RegexAst) -> Result<RegexId> {
        let id = match ast {
            RegexAst::And(asts) => {
                let ids = self.add_asts(asts)?;
                self.and(ids)
            }
            RegexAst::Or(asts) => {
                let ids = self.add_asts(asts)?;
                self.add_node(RegexNode::Or(ids))
            }
            RegexAst::Concat(asts) => {
                let ids = self.add_asts(asts)?;
                self.concat(ids)
            }
            RegexAst::LookAhead(ast) => {
                let id = self.add_ast(*ast)?;
                self.add_node(RegexNode::LookAhead(id))
            }
            RegexAst::Not(ast) => {
                let id = self.add_ast(*ast)?;
                self.not(id)
            }
            RegexAst::Repeat(ast, min, max) => {
                let id = self.add_ast(*ast)?;
                self.repeat(id, min, Some(max))
            }
            RegexAst::EmptyString => self.add_node(RegexNode::EmptyString),
            RegexAst::NoMatch => self.add_node(RegexNode::NoMatch),
            RegexAst::Regex(rx) => self.regex(rx),
            RegexAst::Literal(s) => self.literal(s),
            RegexAst::ByteLiteral(bytes) => self.add_node(RegexNode::ByteLiteral(bytes)),
            RegexAst::Byte(b) => self.add_node(RegexNode::Byte(b)),
            RegexAst::ByteSet(bs) => self.add_node(RegexNode::ByteSet(bs)),
            RegexAst::ExprRef(_) => {
                bail!("ExprRef not supported")
            }
        };
        Ok(id)
    }

    fn add_asts(&mut self, asts: Vec<RegexAst>) -> Result<Vec<RegexId>> {
        asts.into_iter().map(|ast| self.add_ast(ast)).collect()
    }

    pub fn add_node(&mut self, node: RegexNode) -> RegexId {
        if let Some(id) = self.node_ids.get(&node) {
            return *id;
        }
        let id = RegexId(self.nodes.len());
        self.nodes.push(node.clone());
        self.node_ids.insert(node, id);
        id
    }

    pub fn regex(&mut self, rx: String) -> RegexId {
        self.add_node(RegexNode::Regex(rx))
    }

    pub fn literal(&mut self, s: String) -> RegexId {
        self.add_node(RegexNode::Literal(s))
    }

    pub fn concat(&mut self, nodes: Vec<RegexId>) -> RegexId {
        if nodes.len() == 1 {
            return nodes[0];
        }
        if nodes.len() == 0 {
            return self.add_node(RegexNode::NoMatch);
        }
        self.add_node(RegexNode::Concat(nodes))
    }

    pub fn select(&mut self, nodes: Vec<RegexId>) -> RegexId {
        if nodes.len() == 1 {
            return nodes[0];
        }
        if nodes.len() == 0 {
            return self.add_node(RegexNode::NoMatch);
        }
        self.add_node(RegexNode::Or(nodes))
    }

    pub fn zero_or_more(&mut self, node: RegexId) -> RegexId {
        self.repeat(node, 0, None)
    }

    pub fn one_or_more(&mut self, node: RegexId) -> RegexId {
        self.repeat(node, 1, None)
    }

    pub fn optional(&mut self, node: RegexId) -> RegexId {
        self.repeat(node, 0, Some(1))
    }

    pub fn repeat(&mut self, node: RegexId, min: u32, max: Option<u32>) -> RegexId {
        self.add_node(RegexNode::Repeat(node, min, max))
    }

    pub fn not(&mut self, node: RegexId) -> RegexId {
        self.add_node(RegexNode::Not(node))
    }

    pub fn and(&mut self, nodes: Vec<RegexId>) -> RegexId {
        self.add_node(RegexNode::And(nodes))
    }

    fn finalize(&mut self) -> Vec<RegexNode> {
        let r = std::mem::take(&mut self.nodes);
        *self = Self::new();
        r
    }
}

impl GrammarBuilder {
    pub fn new() -> Self {
        Self {
            top_grammar: TopLevelGrammar {
                grammars: vec![],
                max_tokens: None,
                test_trace: false,
            },
            placeholder: Node::String {
                literal: "__placeholder__: do not use this string in grammars".to_string(),
                props: NodeProps {
                    max_tokens: Some(usize::MAX - 108),
                    capture_name: Some("$$$placeholder$$$".to_string()),
                    ..NodeProps::default()
                },
            },
            strings: HashMap::new(),
            curr_grammar_id: 0,
            node_refs: HashMap::new(),
            nodes: vec![],
            regex: RegexBuilder::new(),
            at_most_cache: HashMap::new(),
            repeat_exact_cache: HashMap::new(),
        }
    }

    fn shift_nodes(&mut self) {
        if self.top_grammar.grammars.len() == 0 {
            assert!(self.nodes.is_empty(), "nodes added before add_grammar()");
        } else {
            let nodes = std::mem::take(&mut self.nodes);
            assert!(
                nodes.len() > 0,
                "no nodes added before add_grammar() or finalize()"
            );
            self.top_grammar.grammars.last_mut().unwrap().nodes = nodes;
            self.top_grammar.grammars.last_mut().unwrap().rx_nodes = self.regex.finalize();
        }
    }

    fn next_grammar_id(&mut self) {
        static COUNTER: AtomicU32 = AtomicU32::new(1);
        self.curr_grammar_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn add_grammar(&mut self, grammar: GrammarWithLexer) {
        assert!(grammar.nodes.is_empty(), "Grammar already has nodes");
        self.shift_nodes();

        self.next_grammar_id();
        self.top_grammar.grammars.push(grammar);
        self.strings.clear();

        // add root node
        let id = self.placeholder();
        assert!(id.idx == 0);
    }

    pub fn add_node(&mut self, node: Node) -> NodeRef {
        // Generate a key for the node from its serialized form if it is not the placeholder
        let key = (node != self.placeholder)
            .then(|| serde_json::to_string(&node).ok())
            .flatten();

        // Return the node reference if it already exists
        if let Some(ref key) = key {
            if let Some(node_ref) = self.node_refs.get(key) {
                return *node_ref;
            }
        }

        // Create new node reference
        let r = NodeRef {
            idx: self.nodes.len(),
            grammar_id: self.curr_grammar_id,
        };

        // Add the node and store the reference (if it's not the placeholder)
        self.nodes.push(node);
        if let Some(key) = key {
            self.node_refs.insert(key, r);
        }
        r
    }

    pub fn string(&mut self, s: &str) -> NodeRef {
        if let Some(r) = self.strings.get(s) {
            return *r;
        }
        let r = self.add_node(Node::String {
            literal: s.to_string(),
            props: NodeProps::default(),
        });
        self.strings.insert(s.to_string(), r);
        r
    }

    pub fn special_token(&mut self, name: &str) -> NodeRef {
        self.add_node(Node::SpecialToken {
            token: name.to_string(),
            props: NodeProps::default(),
        })
    }

    pub fn gen_grammar(&mut self, data: GenGrammarOptions, props: NodeProps) -> NodeRef {
        self.add_node(Node::GenGrammar { data, props })
    }

    pub fn gen_rx(&mut self, regex: &str, stop_regex: &str) -> NodeRef {
        self.gen(
            GenOptions {
                body_rx: RegexSpec::Regex(regex.to_string()),
                stop_rx: RegexSpec::Regex(stop_regex.to_string()),
                stop_capture_name: None,
                lazy: None,
                temperature: None,
            },
            NodeProps::default(),
        )
    }

    pub fn gen(&mut self, data: GenOptions, props: NodeProps) -> NodeRef {
        self.add_node(Node::Gen { data, props })
    }

    pub fn lexeme(&mut self, rx: RegexSpec, json_quoted: bool) -> NodeRef {
        self.add_node(Node::Lexeme {
            rx,
            contextual: None,
            temperature: None,
            json_string: Some(json_quoted),
            json_raw: None,
            json_allowed_escapes: None,
            props: NodeProps::default(),
        })
    }

    fn child_nodes(&mut self, options: &[NodeRef]) -> Vec<NodeId> {
        options
            .iter()
            .map(|e| {
                assert!(e.grammar_id == self.curr_grammar_id);
                NodeId(e.idx)
            })
            .collect()
    }

    pub fn select(&mut self, options: &[NodeRef]) -> NodeRef {
        let ch = self.child_nodes(&options);
        self.add_node(Node::Select {
            among: ch,
            props: NodeProps::default(),
        })
    }

    pub fn max_tokens(&mut self, node: NodeRef, max_tokens: usize) -> NodeRef {
        self.join_props(
            &[node],
            NodeProps {
                max_tokens: Some(max_tokens),
                ..Default::default()
            },
        )
    }

    pub fn join(&mut self, values: &[NodeRef]) -> NodeRef {
        self.join_props(values, NodeProps::default())
    }

    pub fn join_props(&mut self, values: &[NodeRef], props: NodeProps) -> NodeRef {
        let mut ch = self.child_nodes(&values);
        let empty = NodeId(self.empty().idx);
        ch.retain(|&n| n != empty);
        if ch.len() == 0 {
            return self.empty();
        }
        if ch.len() == 1 && props == NodeProps::default() {
            return NodeRef {
                idx: ch[0].0,
                grammar_id: self.curr_grammar_id,
            };
        }
        self.add_node(Node::Join {
            sequence: ch,
            props,
        })
    }

    pub fn empty(&mut self) -> NodeRef {
        self.string("")
    }

    pub fn optional(&mut self, value: NodeRef) -> NodeRef {
        let empty = self.empty();
        self.select(&[value, empty])
    }

    pub fn one_or_more(&mut self, elt: NodeRef) -> NodeRef {
        let p = self.placeholder();
        let p_elt = self.join(&[p, elt]);
        let inner = self.select(&[elt, p_elt]);
        self.set_placeholder(p, inner);
        p
    }

    pub fn zero_or_more(&mut self, elt: NodeRef) -> NodeRef {
        let p = self.placeholder();
        let empty = self.empty();
        let p_elt = self.join(&[p, elt]);
        let inner = self.select(&[empty, p_elt]);
        self.set_placeholder(p, inner);
        p
    }

    // at_most() creates a rule which accepts at most 'n' copies
    // of element 'elt'.

    // The first-time reader of at_most() might want to consult
    // the comments for repeat_exact(), where similar logic is
    // used in a simpler form.
    //
    // at_most() recursively factors the sequence into K-size pieces,
    // in an attempt to keep grammar size O(log(n)).
    fn at_most(&mut self, elt: NodeRef, n: usize) -> NodeRef {
        if let Some(r) = self.at_most_cache.get(&(elt, n)) {
            return *r;
        }
        let r = if n == 0 {
            // If the max ('n') is 0, an empty rule
            self.empty()
        } else if n == 1 {
            // If 'n' is 1, an optional rule of length 1
            self.optional(elt)
        } else if n < 3 * K {
            // If 'n' is below a fixed number (currently 12),
            // the rule is a choice of all the rules of fixed length
            // from 0 to 'n'.
            let options = (0..=n)
                .map(|k| self.simple_repeat(elt, k))
                .collect::<Vec<_>>();
            self.select(&options)
        } else {
            // Above a fixed number (again, currently 12),
            // we "factor" the sequence into K-sized pieces.
            // Let 'elt_k' be a k-element --- the repetition
            // of 'k' copies of the element ('elt').
            let elt_k = self.simple_repeat(elt, K);

            // First we deal with the sequences of length less than
            // (n/K)*K.
            // 'elt_max_nk' is all the sequences of k-elements
            // of length less than n/K.
            let elt_max_nk = self.at_most(elt_k, (n / K) - 1);
            // The may be up to K-1 elements not accounted by the sequences
            // of k-elements in 'elt_max_k'.  The choices in 'elt_max_k'
            // account for these "remainders".
            let elt_max_k = self.at_most(elt, K - 1);
            let elt_max_nk = self.join(&[elt_max_nk, elt_max_k]);

            // Next we deal with the sequences of length between
            // (n/K)*K and 'n', inclusive. It is integer arithmetic, so there
            // will be n%K of these.
            // Here we call n/K the quotient and n%K the remainder.
            // 'elt_nk' repeats the k-element exactly the quotient
            // number of times, to ensure all our sequences are of
            // length at least (n/K)*K.
            let elt_nk = self.repeat_exact(elt_k, n / K);
            // 'left' repeats 'elt' at most the remainder number
            // of times.  The remainder is always less than K.
            let left = self.at_most(elt, n % K);
            // Join 'elt_nk' and 'left' into 'elt_n'.
            // 'elt_nk' is a constant-sized piece,
            // which ensures all the sequences of 'elt' in 'elt_n',
            // will be of length at least (n/K)*K.
            // 'left' will be a choice of rules which
            // produce at most K-1 copies of 'elt'.
            let elt_n = self.join(&[elt_nk, left]);

            // We have accounted for all the sequences of less than
            // (n/K)*K elements in 'elt_max_nk'.  We have accounted
            // for all the sequences of length between (n/K)*K elements and n elements
            // (inclusive) in 'elt_n'.  Clearly, the sequences of length at most 'n'
            // are the alternation of 'elt_max_nk' and 'elt_n'.
            self.select(&[elt_n, elt_max_nk])
        };
        self.at_most_cache.insert((elt, n), r);
        r
    }

    // simple_repeat() "simply" repeats the element ('elt') 'n' times.
    // Here "simple" means we do not factor into K-size pieces, so that
    // time will be O(n).  The intent is that simple_repeat() only be
    // called for small 'n'.
    fn simple_repeat(&mut self, elt: NodeRef, n: usize) -> NodeRef {
        let elt_n = (0..n).map(|_| elt).collect::<Vec<_>>();
        self.join(&elt_n)
    }

    // Repeat element 'elt' exactly 'n' times, using factoring
    // in an attempt to keep grammar size O(log(n)).
    fn repeat_exact(&mut self, elt: NodeRef, n: usize) -> NodeRef {
        if let Some(r) = self.repeat_exact_cache.get(&(elt, n)) {
            return *r;
        }
        let r = if n > 2 * K {
            // For large 'n', try to keep the number of rules O(log(n))
            // by "factoring" the sequence into K-sized pieces

            // Create a K-element -- 'elt' repeated 'K' times.
            let elt_k = self.simple_repeat(elt, K);

            // Repeat the K-element n/K times.  The repetition
            // is itself factored, so that the process is
            // recursive.
            let inner = self.repeat_exact(elt_k, n / K);

            // 'inner' will contain ((n/K)K) be an 'elt'-sequence
            // of length ((n/K)K), which is n-((n/K)K), or n%K,
            // short of what we want.  We create 'elt_left' to contain
            // the n%K additional items we need, and concatenate it
            // with 'inner' to form our result.
            let left = n % K;
            let mut elt_left = (0..left).map(|_| elt).collect::<Vec<_>>();
            elt_left.push(inner);
            self.join(&elt_left)
        } else {
            // For small 'n' (currently, 8 or less), simply
            // repeat 'elt' 'n' times.
            self.simple_repeat(elt, n)
        };
        self.repeat_exact_cache.insert((elt, n), r);
        r
    }

    // at_least() accepts a sequence of at least 'n' copies of
    // element 'elt'.
    fn at_least(&mut self, elt: NodeRef, n: usize) -> NodeRef {
        let z = self.zero_or_more(elt);
        if n == 0 {
            // If n==0, atleast() is equivalent to zero_or_more().
            z
        } else {
            // If n>0, first sequence is a factored repetition of
            // exactly 'n' copies of 'elt', ...
            let r = self.repeat_exact(elt, n);
            // ... followed by zero or more copies of 'elt'
            self.join(&[r, z])
        }
    }

    // Create a rule which accepts from 'min' to 'max' copies of element
    // 'elt', inclusive.
    pub fn repeat(&mut self, elt: NodeRef, min: usize, max: Option<usize>) -> NodeRef {
        if max.is_none() {
            // If no 'max', what we want is equivalent to a rule accepting at least
            // 'min' elements.
            return self.at_least(elt, min);
        }
        let max = max.unwrap();
        assert!(min <= max);
        if min == max {
            // Where 'min' is equal to 'max', what we want is equivalent to a rule
            // repeating element 'elt' exactly 'min' times.
            self.repeat_exact(elt, min)
        } else if min == 0 {
            // If 'min' is zero, what we want is equivalent to a rule accepting at least
            // 'min' elements.
            self.at_most(elt, max)
        } else {
            // In the general case, what we want is equivalent to
            // a rule accepting a fixed-size block of length 'min',
            // followed by a rule accepting at most 'd' elements,
            // where 'd' is the difference between 'min' and 'max'
            let d = max - min;
            let common = self.repeat_exact(elt, min);
            let extra = self.at_most(elt, d);
            self.join(&[common, extra])
        }
    }

    pub fn placeholder(&mut self) -> NodeRef {
        self.add_node(self.placeholder.clone())
    }

    pub fn is_placeholder(&self, node: NodeRef) -> bool {
        assert!(node.grammar_id == self.curr_grammar_id);
        self.nodes[node.idx] == self.placeholder
    }

    pub fn set_placeholder(&mut self, placeholder: NodeRef, node: NodeRef) {
        let ch = self.child_nodes(&[placeholder, node]); // validate
        if !self.is_placeholder(placeholder) {
            panic!(
                "placeholder already set at {} to {:?}",
                placeholder.idx, self.nodes[placeholder.idx]
            );
        }
        self.nodes[placeholder.idx] = Node::Join {
            sequence: vec![ch[1]],
            props: NodeProps::default(),
        };
    }

    pub fn set_start_node(&mut self, node: NodeRef) {
        self.set_placeholder(
            NodeRef {
                idx: 0,
                grammar_id: self.curr_grammar_id,
            },
            node,
        );
    }

    pub fn finalize(mut self) -> Result<TopLevelGrammar> {
        ensure!(
            self.top_grammar.grammars.len() > 0,
            "No grammars added to the top level grammar"
        );
        self.shift_nodes();
        for grammar in &self.top_grammar.grammars {
            for node in &grammar.nodes {
                ensure!(node != &self.placeholder, "Unresolved placeholder");
            }
        }
        Ok(self.top_grammar.clone())
    }
}
