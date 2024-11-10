use std::{collections::HashMap, vec};
use derivre::{ExprRef, Regex, RegexAst, RegexBuilder};

#[derive(Debug)]
struct State {
    len: usize,
    link: Option<usize>,
    next: HashMap<char, usize>,
}

/// For details see https://en.wikipedia.org/wiki/Suffix_automaton.
/// Implementation is based on https://cp-algorithms.com/string/suffix-automaton.html
struct SuffixAutomaton {
    states: Vec<State>,
    last: usize,
}

impl SuffixAutomaton {
    fn new() -> Self {
        let init_state = State {
            len: 0,
            link: None,
            next: HashMap::new(),
        };
        SuffixAutomaton {
            states: vec![init_state],
            last: 0,
        }
    }

    fn from_string(s: &str) -> Self {
        let mut sa = SuffixAutomaton::new();
        for c in s.chars() {
            sa.extend(c);
        }
        sa
    }

    fn extend(&mut self, c: char) {
        let cur_index = self.states.len();
        self.states.push(
            State {
                len: self.states[self.last].len + 1,
                link: None,
                next: HashMap::new(),
            }
        );

        let mut p = Some(self.last);
        while let Some(pp) = p{
            if self.states[pp].next.contains_key(&c) {
                break;
            }
            self.states[pp].next.insert(c, cur_index);
            p = self.states[pp].link;
        }

        if let Some(pp) = p {
            let q = self.states[pp].next[&c];
            if self.states[pp].len + 1 == self.states[q].len {
                self.states[cur_index].link = Some(q);
            } else {
                let clone_index = self.states.len();
                self.states.push(
                    State {
                        len: self.states[pp].len + 1,
                        link: self.states[q].link,
                        next: self.states[q].next.clone(),
                    }
                );
                while let Some(ppp) = p {
                    if self.states[ppp].next[&c] == q {
                        self.states[ppp].next.insert(c, clone_index);
                    } else {
                        break;
                    }
                    p = self.states[ppp].link;
                }
                self.states[q].link = Some(clone_index);
                self.states[cur_index].link = Some(clone_index);
            }
        } else {
            self.states[cur_index].link = Some(0);
        }
        self.last = cur_index;
    }
}

pub fn substring(string: &str) -> Result<Regex, anyhow::Error> {
    let sa = SuffixAutomaton::from_string(string);
    let mut builder = RegexBuilder::new();
    let mut state_stack = vec![0];
    let mut node_cache: HashMap<usize, ExprRef> = HashMap::new();

    while let Some(state_index) = state_stack.last() {
        let state = &sa.states[*state_index];
        if node_cache.contains_key(state_index) {
            state_stack.pop();
            continue;
        }

        if state.next.is_empty() {
            let expr = builder.mk(&RegexAst::EmptyString)?;
            node_cache.insert(*state_index, expr);
            state_stack.pop();
            continue;
        }

        let unprocessed_children = state.next
            .values()
            .filter(|child_index| !node_cache.contains_key(child_index))
            .collect::<Vec<_>>();
        if !unprocessed_children.is_empty() {
            state_stack.extend(unprocessed_children);
            continue;
        }

        let options = state.next
            .keys()
            .map(|c| {
                RegexAst::Concat(vec![
                    RegexAst::Literal(c.to_string()),
                    RegexAst::ExprRef(node_cache[&state.next[c]]),
                ])
            })
            .collect::<Vec<_>>();
        let node = RegexAst::Or([options, vec![RegexAst::EmptyString]].concat());
        let expr = builder.mk(&node)?;
        node_cache.insert(*state_index, expr);
        state_stack.pop();
    }
    Ok(builder.to_regex(node_cache[&0]))
}

mod tests {
    use super::{substring, SuffixAutomaton};

    #[test]
    fn test_substring() {
        let regex = substring("abacaba").unwrap();
        assert_eq!(regex.clone().is_match("acab"), true);
        assert_eq!(regex.clone().is_match("abaca"), true);
        assert_eq!(regex.clone().is_match("acaba"), true);
        assert_eq!(regex.clone().is_match("abacaba"), true);
        assert_eq!(regex.clone().is_match("acabab"), false);
        assert_eq!(regex.clone().is_match("cabaca"), false);
    }
}
