use std::collections::HashMap;

use anyhow::Result;

use crate::{
    api::{RegexSpec, TopLevelGrammar},
    GrammarBuilder, NodeRef,
};

use super::ast::*;

struct Compiler {
    builder: GrammarBuilder,
    items: Vec<Item>,
    nodes: HashMap<String, NodeInfo>,
}

struct NodeInfo {
    id: NodeRef,
    is_terminal: bool,
    regex: Option<RegexSpec>,
}

pub fn lark_to_llguidance(items: Vec<Item>) -> Result<TopLevelGrammar> {
    let mut c = Compiler {
        builder: GrammarBuilder::new(),
        items,
        nodes: HashMap::new(),
    };
    c.execute()?;
    c.builder.finalize()
}

impl Compiler {
    fn execute(&mut self) -> Result<()> {
        for item in self.items.iter() {
            match item {
                Item::Rule(rule) => {
                    let id = self.builder.placeholder();
                    self.nodes.insert(
                        rule.name.clone(),
                        NodeInfo {
                            id,
                            is_terminal: false,
                            regex: None,
                        },
                    );
                }
                Item::Token(token_def) => {
                    let id = self.builder.placeholder();
                    self.nodes.insert(
                        token_def.name.clone(),
                        NodeInfo {
                            id,
                            is_terminal: true,
                            regex: None,
                        },
                    );
                }
                Item::Statement(statement) => todo!(),
            }
        }
        Ok(())
    }
}
