use std::{collections::HashMap, sync::atomic::AtomicU32};

use anyhow::{ensure, Result};

use crate::api::{
    GrammarWithLexer, Node, NodeId, NodeProps, RegexId, RegexNode, RegexSpec, TopLevelGrammar,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
    nodes: Vec<Node>,
    pub regex: RegexBuilder,
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

    fn finalize(&mut self) -> Vec<RegexNode> {
        let r = std::mem::take(&mut self.nodes);
        *self = Self::new();
        r
    }
}

impl GrammarBuilder {
    pub fn from_grammar(grammar: TopLevelGrammar) -> (Self, NodeRef) {
        assert!(grammar.grammars.len() == 1);
        let mut builder = Self::new();
        builder.top_grammar = grammar;
        builder.nodes = std::mem::take(&mut builder.top_grammar.grammars[0].nodes);
        builder.next_grammar_id();
        let prev_root = builder.nodes[0].clone();
        builder.nodes[0] = builder.placeholder.clone();
        let prev_root = builder.add_node(prev_root);
        (builder, prev_root)
    }

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
            nodes: vec![],
            regex: RegexBuilder::new(),
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

    fn add_node(&mut self, node: Node) -> NodeRef {
        let r = NodeRef {
            idx: self.nodes.len(),
            grammar_id: self.curr_grammar_id,
        };
        self.nodes.push(node);
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

    pub fn join(&mut self, values: &[NodeRef]) -> NodeRef {
        let mut ch = self.child_nodes(&values);
        let empty = NodeId(self.empty().idx);
        ch.retain(|&n| n != empty);
        if ch.len() == 0 {
            return self.empty();
        }
        if ch.len() == 1 {
            return NodeRef {
                idx: ch[0].0,
                grammar_id: self.curr_grammar_id,
            };
        }
        self.add_node(Node::Join {
            sequence: ch,
            props: NodeProps::default(),
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

    // this tries to keep grammar size O(log(n))
    fn at_most(&mut self, elt: NodeRef, n: usize) -> NodeRef {
        if n == 0 {
            self.empty()
        } else if n == 1 {
            self.optional(elt)
        } else if n < 3 * K {
            let n_1 = self.at_most(elt, n - 1);
            let inner = self.join(&[elt, n_1]);
            self.optional(inner)
        } else {
            let elt_k = self.simple_repeat(elt, K);

            let elt_max_nk = self.at_most(elt_k, (n / K) - 1);
            let elt_max_k = self.at_most(elt, K - 1);
            let elt_max_nk = self.join(&[elt_max_nk, elt_max_k]);

            let elt_nk = self.repeat_exact(elt_k, n / K);
            let left = self.at_most(elt, n % K);
            let elt_n = self.join(&[elt_nk, left]);

            self.select(&[elt_n, elt_max_nk])
        }
    }

    fn simple_repeat(&mut self, elt: NodeRef, n: usize) -> NodeRef {
        if n == 0 {
            self.empty()
        } else if n == 1 {
            elt
        } else {
            let elt_n = (0..n).map(|_| elt).collect::<Vec<_>>();
            self.join(&elt_n)
        }
    }

    // this tries to keep grammar size O(log(n))
    fn repeat_exact(&mut self, elt: NodeRef, n: usize) -> NodeRef {
        if n > 2 * K {
            let elt_k = self.simple_repeat(elt, K);
            let inner = self.repeat_exact(elt_k, n / K);
            let left = n % K;
            let mut elt_left = (0..left).map(|_| elt).collect::<Vec<_>>();
            elt_left.push(inner);
            self.join(&elt_left)
        } else {
            self.simple_repeat(elt, n)
        }
    }

    fn at_least(&mut self, elt: NodeRef, n: usize) -> NodeRef {
        let z = self.zero_or_more(elt);
        if n == 0 {
            z
        } else {
            let r = self.repeat_exact(elt, n);
            self.join(&[r, z])
        }
    }

    pub fn repeat(&mut self, elt: NodeRef, min: usize, max: Option<usize>) -> NodeRef {
        if max.is_none() {
            return self.at_least(elt, min);
        }
        let max = max.unwrap();
        assert!(min <= max);
        if min == max {
            self.repeat_exact(elt, min)
        } else if min == 0 {
            self.at_most(elt, max)
        } else {
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
