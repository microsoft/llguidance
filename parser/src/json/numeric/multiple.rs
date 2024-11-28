use anyhow::{bail, Result};
use derivre::RegexAst;

fn rx_pos_int_multiple_of(n: u64) -> Result<RegexAst> {
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
        current /= 10u64.pow(n_trailing_zeroes as u32);
    } 

    // Nonzero and no leading zeroes!
    nodes.push(RegexAst::Regex("[1-9][0-9]*".to_string()));

    // Factorize the number if we can (must have multiplicity <= 1)
    if current != 1 {
        let basis = [
            (2, "[0-9]*[02468]".to_string()),
            // From quaxio.com/triple
            (3, format!("({0}|{2}{0}*{1}|({1}|{2}{0}*{2})({0}|{1}{0}*{2})*({2}|{1}{0}*{1}))*", "[0369]", "[147]", "[258]")),
            (5, "[0-9]*[05]".to_string()),
        ];
        for (factor, rx) in basis.into_iter() {
            if current % factor == 0 {
                current /= factor;
                nodes.push(RegexAst::Regex(rx));
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

pub fn rx_int_multiple_of(n: u64) -> Result<RegexAst> {
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
        for n in [1, 2, 3, 5, 6, 10, 15, 20, 30, 50, 100, 200, 300, 600] {
            let node = rx_int_multiple_of(n).unwrap();
            let expr = builder.mk(&node).unwrap();
            let rx = builder.to_regex(expr);
            for i in -1000i64..1000 {
                assert_eq!(rx.clone().is_match(&i.to_string()), i.abs() as u64 % n == 0, "n = {}, i = {}, rx = {:#?}", n, i, node);
            }
        }
    }
}