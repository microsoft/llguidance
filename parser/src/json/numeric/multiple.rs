use std::collections::HashSet;

use anyhow::{bail, Result};
use derivre::RegexAst;
use indexmap::IndexMap;
use lazy_static::lazy_static;

lazy_static! {
    /// Regexes for multiples of common (and easy to regex) numbers
    // Notes:
    // 1. descending order so we use the largest factors first
    // 2. not too careful about leading zeroes, as we can restrict the regex later
    static ref RX_MULTIPLE_OF: IndexMap<u32, String> = IndexMap::from_iter([
        (25, "[0-9]*(00|25|50|75)".to_string()),
        (5, "[0-9]*[05]".to_string()),
        // Last two digits are divisible by 4
        (4, "[048]|[0-9]*([02468][048]|[13579][26])".to_string()),
        // From quaxio.com/triple
        (3, format!("({0}|{2}{0}*{1}|({1}|{2}{0}*{2})({0}|{1}{0}*{2})*({2}|{1}{0}*{1}))*", "[0369]", "[147]", "[258]")),
        (2, "[0-9]*[02468]".to_string()),
    ]);
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a.clone()
    } else {
        gcd(b, a % b)
    }
}

fn rx_pos_int_multiple_of(n: u32) -> Result<RegexAst> {
    if n == 0 {
        bail!("Multiple of 0 not supported");
    }

    let mut current = n;
    let mut nodes: Vec<RegexAst> = Vec::new();

    // Handle multiples of powers of 10
    let n_trailing_zeroes = current
        .to_string() 
        .chars()
        .rev()
        .take_while(|&c| c == '0')
        .count();

    if n_trailing_zeroes > 0 {
        current /= 10u32.pow(n_trailing_zeroes as u32);
    } 

    // Nonzero and no leading zeroes!
    nodes.push(RegexAst::Regex("[1-9][0-9]*".to_string()));

    // Factorize the number if we can (must have multiplicity <= 1)
    let mut used = HashSet::<u32>::new();
    if current != 1 {
        for (&factor, rx) in RX_MULTIPLE_OF.iter() {
            if current % factor == 0 {
                if used.iter().find(|&&x| gcd(x, factor) != 1).is_some() {
                    // Not coprime with all used factors, so we don't know how to factorize
                    // E.g. 8 = 4 * 2, but 2 and 4 are not coprime, which means that the intersection
                    // of their regexes doesn't imply the regex for 8
                    bail!("Multiple of {} not supported", n);
                }
                current /= factor;
                nodes.push(RegexAst::Regex(rx.clone()));
                used.insert(factor.clone());
            }
        }
        if current != 1 {
            // We couldn't factorize the number
            bail!("Multiple of {} not supported", n);
        }
    }

    let mut ast = match nodes.len() {
        0 => unreachable!(),
        1 => nodes.pop().unwrap(),
        _ => RegexAst::And(nodes)
    };
    if n_trailing_zeroes > 0 {
        ast = RegexAst::Concat(vec![
            ast,
            RegexAst::Regex(format!("0{{{}}}", n_trailing_zeroes))
        ]);
    }
    Ok(ast)
}

pub fn rx_int_multiple_of(n: u32) -> Result<RegexAst> {
    let pos_rx = rx_pos_int_multiple_of(n)?;
    Ok(RegexAst::Or(vec![
        RegexAst::Literal("0".to_string()),
        RegexAst::Concat(vec![
            RegexAst::Regex("-?".to_string()),
            pos_rx
        ])
    ]))
}

#[cfg(test)]
mod tests {
    use derivre::RegexBuilder;
    use super::rx_int_multiple_of;

    #[test]
    fn test_rx_int_multiple_of() {
        let mut builder = RegexBuilder::new();
        for n in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 25, 30, 50, 60, 75, 100, 120, 150, 200, 250, 300, 500, 600, 750] {
            let node = rx_int_multiple_of(n).unwrap();
            let expr = builder.mk(&node).unwrap();
            let rx = builder.to_regex(expr);
            for i in -1000i32..1000 {
                assert_eq!(rx.clone().is_match(&i.to_string()), i.abs() as u32 % n == 0, "n = {}, i = {}, rx = {:#?}", n, i, node);
            }
        }
    }
}