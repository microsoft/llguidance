use anyhow::{anyhow, Result};
use regex_syntax::escape;

fn mk_or(parts: Vec<String>) -> String {
    if parts.len() == 1 {
        parts[0].clone()
    } else {
        format!("({})", parts.join("|"))
    }
}

fn num_digits(n: i64) -> usize {
    n.abs().to_string().len()
}

pub fn rx_int_range(left: Option<i64>, right: Option<i64>) -> Result<String> {
    match (left, right) {
        (None, None) => Ok("-?(0|[1-9][0-9]*)".to_string()),
        (Some(left), None) => {
            if left < 0 {
                Ok(mk_or(vec![
                    rx_int_range(Some(left), Some(-1))?,
                    rx_int_range(Some(0), None)?,
                ]))
            } else {
                let max_value = "9"
                    .repeat(num_digits(left))
                    .parse::<i64>()
                    .map_err(|e| anyhow!("Failed to parse max value for left {}: {}", left, e))?;
                Ok(mk_or(vec![
                    rx_int_range(Some(left), Some(max_value))?,
                    format!("[1-9][0-9]{{{},}}", num_digits(left)),
                ]))
            }
        }
        (None, Some(right)) => {
            if right >= 0 {
                Ok(mk_or(vec![
                    rx_int_range(Some(0), Some(right))?,
                    rx_int_range(None, Some(-1))?,
                ]))
            } else {
                Ok(format!("-{}", rx_int_range(Some(-right), None)?))
            }
        }
        (Some(left), Some(right)) => {
            if left > right {
                return Err(anyhow!(
                    "Invalid range: left ({}) cannot be greater than right ({})",
                    left,
                    right
                ));
            }
            if left < 0 {
                if right < 0 {
                    Ok(format!("(-{})", rx_int_range(Some(-right), Some(-left))?))
                } else {
                    Ok(format!(
                        "(-{}|{})",
                        rx_int_range(Some(0), Some(-left))?,
                        rx_int_range(Some(0), Some(right))?
                    ))
                }
            } else {
                if num_digits(left) == num_digits(right) {
                    let l = left.to_string();
                    let r = right.to_string();
                    if left == right {
                        return Ok(format!("({})", l));
                    }

                    let lpref = &l[..l.len() - 1];
                    let lx = &l[l.len() - 1..];
                    let rpref = &r[..r.len() - 1];
                    let rx = &r[r.len() - 1..];

                    if lpref == rpref {
                        return Ok(format!("({}[{}-{}])", lpref, lx, rx));
                    }

                    let mut left_rec = lpref.parse::<i64>().unwrap_or(0);
                    let mut right_rec = rpref.parse::<i64>().unwrap_or(0);
                    if left_rec >= right_rec {
                        return Err(anyhow!(
                            "Invalid recursive range: left_rec ({}) must be less than right_rec ({})",
                            left_rec,
                            right_rec
                        ));
                    }

                    let mut parts = Vec::new();

                    if lx != "0" {
                        left_rec += 1;
                        parts.push(format!("{}[{}-9]", lpref, lx));
                    }

                    if rx != "9" {
                        right_rec -= 1;
                        parts.push(format!("{}[0-{}]", rpref, rx));
                    }

                    if left_rec <= right_rec {
                        let inner = rx_int_range(Some(left_rec), Some(right_rec))?;
                        parts.push(format!("{}[0-9]", inner));
                    }

                    Ok(mk_or(parts))
                } else {
                    let break_point = 10_i64
                        .checked_pow(num_digits(left) as u32)
                        .ok_or_else(|| anyhow!("Overflow when calculating break point"))?
                        - 1;
                    Ok(mk_or(vec![
                        rx_int_range(Some(left), Some(break_point))?,
                        rx_int_range(Some(break_point + 1), Some(right))?,
                    ]))
                }
            }
        }
    }
}

fn lexi_x_to_9(x: &str, incl: bool) -> Result<String> {
    if incl {
        if x.is_empty() {
            Ok("[0-9]*".to_string())
        } else if x.len() == 1 {
            Ok(format!("[{}-9][0-9]*", x))
        } else {
            let x0 = x
                .chars()
                .next()
                .ok_or_else(|| anyhow!("String x is unexpectedly empty"))?
                .to_digit(10)
                .ok_or_else(|| anyhow!("Failed to parse character as digit"))?;
            let x_rest = &x[1..];
            let mut parts = vec![format!(
                "{}{}",
                x.chars()
                    .next()
                    .ok_or_else(|| anyhow!("String x is unexpectedly empty"))?,
                lexi_x_to_9(x_rest, incl)?
            )];
            if x0 < 9 {
                parts.push(format!("[{}-9][0-9]*", x0 + 1));
            }
            Ok(mk_or(parts))
        }
    } else {
        if x.is_empty() {
            Ok("[0-9]*[1-9]".to_string())
        } else {
            let x0 = x
                .chars()
                .next()
                .ok_or_else(|| anyhow!("String x is unexpectedly empty"))?
                .to_digit(10)
                .ok_or_else(|| anyhow!("Failed to parse character as digit"))?;
            let x_rest = &x[1..];
            let mut parts = vec![format!(
                "{}{}",
                x.chars()
                    .next()
                    .ok_or_else(|| anyhow!("String x is unexpectedly empty"))?,
                lexi_x_to_9(x_rest, incl)?
            )];
            if x0 < 9 {
                parts.push(format!("[{}-9][0-9]*", x0 + 1));
            }
            Ok(mk_or(parts))
        }
    }
}

fn lexi_0_to_x(x: &str, incl: bool) -> Result<String> {
    if x.is_empty() {
        if incl {
            Ok("".to_string())
        } else {
            Err(anyhow!("Inclusive flag must be true for an empty string"))
        }
    } else {
        let x0 = x
            .chars()
            .next()
            .ok_or_else(|| anyhow!("String x is unexpectedly empty"))?
            .to_digit(10)
            .ok_or_else(|| anyhow!("Failed to parse character as digit"))?;
        let x_rest = &x[1..];

        if !incl && x.len() == 1 {
            if x0 == 0 {
                return Err(anyhow!(
                    "x0 must be greater than 0 for non-inclusive single character"
                ));
            }
            return Ok(format!("[0-{}][0-9]*", x0 - 1));
        }

        let mut parts = vec![format!(
            "{}{}",
            x.chars()
                .next()
                .ok_or_else(|| anyhow!("String x is unexpectedly empty"))?,
            lexi_0_to_x(x_rest, incl)?
        )];
        if x0 > 0 {
            parts.push(format!("[0-{}][0-9]*", x0 - 1));
        }
        Ok(mk_or(parts))
    }
}

fn lexi_range(ld: &str, rd: &str, ld_incl: bool, rd_incl: bool) -> Result<String> {
    if ld.len() != rd.len() {
        return Err(anyhow!("ld and rd must have the same length"));
    }
    if ld == rd {
        if ld_incl && rd_incl {
            Ok(ld.to_string())
        } else {
            Err(anyhow!(
                "Empty range when ld equals rd and not both inclusive"
            ))
        }
    } else {
        let l0 = ld
            .chars()
            .next()
            .ok_or_else(|| anyhow!("ld is unexpectedly empty"))?
            .to_digit(10)
            .ok_or_else(|| anyhow!("Failed to parse character as digit"))?;
        let r0 = rd
            .chars()
            .next()
            .ok_or_else(|| anyhow!("rd is unexpectedly empty"))?
            .to_digit(10)
            .ok_or_else(|| anyhow!("Failed to parse character as digit"))?;
        if l0 == r0 {
            let ld_rest = &ld[1..];
            let rd_rest = &rd[1..];
            Ok(format!(
                "{}{}",
                ld.chars()
                    .next()
                    .ok_or_else(|| anyhow!("ld is unexpectedly empty"))?,
                lexi_range(ld_rest, rd_rest, ld_incl, rd_incl)?
            ))
        } else {
            if l0 >= r0 {
                return Err(anyhow!("l0 must be less than r0"));
            }
            let ld_rest = ld[1..].trim_end_matches('0');
            let mut parts = vec![format!(
                "{}{}",
                ld.chars()
                    .next()
                    .ok_or_else(|| anyhow!("ld is unexpectedly empty"))?,
                lexi_x_to_9(ld_rest, ld_incl)?
            )];
            if l0 + 1 < r0 {
                parts.push(format!("[{}-{}][0-9]*", l0 + 1, r0 - 1));
            }
            let rd_rest = rd[1..].trim_end_matches('0');
            if !rd_rest.is_empty() || rd_incl {
                parts.push(format!(
                    "{}{}",
                    rd.chars()
                        .next()
                        .ok_or_else(|| anyhow!("rd is unexpectedly empty"))?,
                    lexi_0_to_x(rd_rest, rd_incl)?
                ));
            }
            Ok(mk_or(parts))
        }
    }
}

fn float_to_str(f: f64) -> String {
    format!("{}", f)
}

pub fn rx_float_range(
    left: Option<f64>,
    right: Option<f64>,
    left_inclusive: bool,
    right_inclusive: bool,
) -> Result<String> {
    match (left, right) {
        (None, None) => Ok("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?".to_string()),
        (Some(left), None) => {
            if left < 0.0 {
                Ok(mk_or(vec![
                    rx_float_range(Some(left), Some(0.0), left_inclusive, false)?,
                    rx_float_range(Some(0.0), None, true, false)?,
                ]))
            } else {
                let left_int_part = left as i64;
                Ok(mk_or(vec![
                    rx_float_range(
                        Some(left),
                        Some(10f64.powi(num_digits(left_int_part) as i32)),
                        left_inclusive,
                        false,
                    )?,
                    format!("[1-9][0-9]{{{},}}(\\.[0-9]+)?", num_digits(left_int_part)),
                ]))
            }
        }
        (None, Some(right)) => {
            if right == 0.0 {
                let r = format!("-{}", rx_float_range(Some(0.0), None, false, false)?);
                if right_inclusive {
                    Ok(mk_or(vec![r, "0".to_string()]))
                } else {
                    Ok(r)
                }
            } else if right > 0.0 {
                Ok(mk_or(vec![
                    format!("-{}", rx_float_range(Some(0.0), None, false, false)?),
                    rx_float_range(Some(0.0), Some(right), true, right_inclusive)?,
                ]))
            } else {
                Ok(format!(
                    "-{}",
                    rx_float_range(Some(-right), None, right_inclusive, false)?
                ))
            }
        }
        (Some(left), Some(right)) => {
            if left > right {
                return Err(anyhow!(
                    "Invalid range: left ({}) cannot be greater than right ({})",
                    left,
                    right
                ));
            }
            if left == right {
                if left_inclusive && right_inclusive {
                    Ok(format!("({})", escape(&float_to_str(left))))
                } else {
                    Err(anyhow!(
                        "Empty range when left equals right and not both inclusive"
                    ))
                }
            } else if left < 0.0 {
                if right < 0.0 {
                    Ok(format!(
                        "(-{})",
                        rx_float_range(Some(-right), Some(-left), right_inclusive, left_inclusive)?
                    ))
                } else {
                    let mut parts = vec![];
                    let neg_part = rx_float_range(Some(0.0), Some(-left), false, left_inclusive)?;
                    parts.push(format!("(-{})", neg_part));

                    if right > 0.0 || right_inclusive {
                        let pos_part =
                            rx_float_range(Some(0.0), Some(right), true, right_inclusive)?;
                        parts.push(pos_part);
                    }
                    Ok(mk_or(parts))
                }
            } else {
                let l = float_to_str(left);
                let r = float_to_str(right);
                if l == r {
                    return Err(anyhow!(
                        "Unexpected equality of left and right string representations"
                    ));
                }
                if !left.is_finite() || !right.is_finite() {
                    return Err(anyhow!("Infinite numbers not supported"));
                }

                let mut left_rec: i64 = l
                    .split('.')
                    .next()
                    .ok_or_else(|| anyhow!("Failed to split left integer part"))?
                    .parse()
                    .map_err(|e| anyhow!("Failed to parse left integer part: {}", e))?;
                let right_rec: i64 = r
                    .split('.')
                    .next()
                    .ok_or_else(|| anyhow!("Failed to split right integer part"))?
                    .parse()
                    .map_err(|e| anyhow!("Failed to parse right integer part: {}", e))?;

                let mut ld = l.split('.').nth(1).unwrap_or("").to_string();
                let mut rd = r.split('.').nth(1).unwrap_or("").to_string();

                if left_rec == right_rec {
                    while ld.len() < rd.len() {
                        ld.push('0');
                    }
                    while rd.len() < ld.len() {
                        rd.push('0');
                    }
                    let suff = format!(
                        "\\.{}",
                        lexi_range(&ld, &rd, left_inclusive, right_inclusive)?
                    );
                    if ld.parse::<i64>().unwrap_or(0) == 0 {
                        Ok(format!("({}({})?)", left_rec, suff))
                    } else {
                        Ok(format!("({}{})", left_rec, suff))
                    }
                } else {
                    let mut parts = vec![];
                    if !ld.is_empty() || !left_inclusive {
                        parts.push(format!(
                            "({}\\.{})",
                            left_rec,
                            lexi_x_to_9(&ld, left_inclusive)?
                        ));
                        left_rec += 1;
                    }

                    if right_rec - 1 >= left_rec {
                        let inner = rx_int_range(Some(left_rec), Some(right_rec - 1))?;
                        parts.push(format!("({}(\\.[0-9]+)?)", inner));
                    }

                    if !rd.is_empty() {
                        parts.push(format!(
                            "({}(\\.{})?)",
                            right_rec,
                            lexi_0_to_x(&rd, right_inclusive)?
                        ));
                    } else if right_inclusive {
                        parts.push(format!("{}(\\.0+)?", right_rec));
                    }

                    Ok(mk_or(parts))
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::{rx_float_range, rx_int_range};
    use regex::Regex;

    fn do_test_int_range(rx: &str, left: Option<i64>, right: Option<i64>) {
        let re = Regex::new(&format!("^{}$", rx)).unwrap();
        for n in (left.unwrap_or(0) - 1000)..=(right.unwrap_or(0) + 1000) {
            let matches = re.is_match(&n.to_string());
            let expected =
                (left.is_none() || left.unwrap() <= n) && (right.is_none() || n <= right.unwrap());
            if expected != matches {
                let range_str = match (left, right) {
                    (Some(l), Some(r)) => format!("[{}, {}]", l, r),
                    (Some(l), None) => format!("[{}, ∞)", l),
                    (None, Some(r)) => format!("(-∞, {}]", r),
                    (None, None) => "(-∞, ∞)".to_string(),
                };
                if matches {
                    panic!("{} not in range {} but matches {:?}", n, range_str, rx);
                } else {
                    panic!("{} in range {} but does not match {:?}", n, range_str, rx);
                }
            }
        }
    }

    #[test]
    fn test_int_range() {
        let cases = vec![
            (Some(0), Some(9)),
            (Some(1), Some(7)),
            (Some(0), Some(99)),
            (Some(13), Some(170)),
            (Some(13), Some(17)),
            (Some(13), Some(27)),
            (Some(13), Some(57)),
            (Some(72), Some(91)),
            (Some(723), Some(915)),
            (Some(23), Some(915)),
            (Some(-1), Some(915)),
            (Some(-9), Some(9)),
            (Some(-3), Some(3)),
            (Some(-3), Some(0)),
            (Some(-72), Some(13)),
            (None, Some(0)),
            (None, Some(7)),
            (None, Some(23)),
            (None, Some(725)),
            (None, Some(-1)),
            (None, Some(-17)),
            (None, Some(-283)),
            (Some(0), None),
            (Some(2), None),
            (Some(33), None),
            (Some(234), None),
            (Some(-1), None),
            (Some(-87), None),
            (Some(-329), None),
            (None, None),
            (Some(-13), Some(-13)),
            (Some(-1), Some(-1)),
            (Some(0), Some(0)),
            (Some(1), Some(1)),
            (Some(13), Some(13)),
        ];

        for (left, right) in cases {
            let rx = rx_int_range(left, right).unwrap();
            do_test_int_range(&rx, left, right);
        }
    }

    fn do_test_float_range(
        rx: &str,
        left: Option<f64>,
        right: Option<f64>,
        left_inclusive: bool,
        right_inclusive: bool,
    ) {
        let re = Regex::new(&format!("^{}$", rx)).unwrap();
        let left_int = left.map(|x| {
            let left_int = x.ceil() as i64;
            if !left_inclusive && x == left_int as f64 {
                left_int + 1
            } else {
                left_int
            }
        });
        let right_int = right.map(|x| {
            let right_int = x.floor() as i64;
            if !right_inclusive && x == right_int as f64 {
                right_int - 1
            } else {
                right_int
            }
        });
        do_test_int_range(rx, left_int, right_int);

        let eps1 = 0.0000001;
        let eps2 = 0.01;
        let test_cases = vec![
            left.unwrap_or(-1000.0),
            right.unwrap_or(1000.0),
            0.0,
            left_int.unwrap_or(-1000) as f64,
            right_int.unwrap_or(1000) as f64,
        ];
        for x in test_cases {
            for offset in [0.0, -eps1, eps1, -eps2, eps2, 1.0, -1.0].iter() {
                let n = x + offset;
                let matches = re.is_match(&n.to_string());
                let left_cond =
                    left.is_none() || left.unwrap() < n || (left.unwrap() == n && left_inclusive);
                let right_cond = right.is_none()
                    || right.unwrap() > n
                    || (right.unwrap() == n && right_inclusive);
                let expected = left_cond && right_cond;
                if expected != matches {
                    let lket = if left_inclusive { "[" } else { "(" };
                    let rket = if right_inclusive { "]" } else { ")" };
                    let range_str = match (left, right) {
                        (Some(l), Some(r)) => format!("{}{}, {}{}", lket, l, r, rket),
                        (Some(l), None) => format!("{}{}, ∞)", lket, l),
                        (None, Some(r)) => format!("(-∞, {}{}", r, rket),
                        (None, None) => "(-∞, ∞)".to_string(),
                    };
                    if matches {
                        panic!("{} not in range {} but matches {:?}", n, range_str, rx);
                    } else {
                        panic!("{} in range {} but does not match {:?}", n, range_str, rx);
                    }
                }
            }
        }
    }

    #[test]
    fn test_float_range() {
        let cases = vec![
            (Some(0.0), Some(10.0)),
            (Some(-10.0), Some(0.0)),
            (Some(0.5), Some(0.72)),
            (Some(0.5), Some(1.72)),
            (Some(0.5), Some(1.32)),
            (Some(0.45), Some(0.5)),
            (Some(0.3245), Some(0.325)),
            (Some(0.443245), Some(0.44325)),
            (Some(1.0), Some(2.34)),
            (Some(1.33), Some(2.0)),
            (Some(1.0), Some(10.34)),
            (Some(1.33), Some(10.0)),
            (Some(-1.33), Some(10.0)),
            (Some(-17.23), Some(-1.33)),
            (Some(-1.23), Some(-1.221)),
            (Some(-10.2), Some(45293.9)),
            (None, Some(0.0)),
            (None, Some(1.0)),
            (None, Some(1.5)),
            (None, Some(1.55)),
            (None, Some(-17.23)),
            (None, Some(-1.33)),
            (None, Some(-1.23)),
            (None, Some(103.74)),
            (None, Some(100.0)),
            (Some(0.0), None),
            (Some(1.0), None),
            (Some(1.5), None),
            (Some(1.55), None),
            (Some(-17.23), None),
            (Some(-1.33), None),
            (Some(-1.23), None),
            (Some(103.74), None),
            (Some(100.0), None),
            (None, None),
            (Some(-103.4), Some(-103.4)),
            (Some(-27.0), Some(-27.0)),
            (Some(-1.5), Some(-1.5)),
            (Some(-1.0), Some(-1.0)),
            (Some(0.0), Some(0.0)),
            (Some(1.0), Some(1.0)),
            (Some(1.5), Some(1.5)),
            (Some(27.0), Some(27.0)),
            (Some(103.4), Some(103.4)),
        ];

        for (left, right) in cases {
            for left_inclusive in [true, false].iter() {
                for right_inclusive in [true, false].iter() {
                    match (left, right) {
                        (Some(left), Some(right))
                            if left == right && !(*left_inclusive && *right_inclusive) =>
                        {
                            assert!(rx_float_range(
                                Some(left),
                                Some(right),
                                *left_inclusive,
                                *right_inclusive
                            )
                            .is_err());
                        }
                        _ => {
                            let rx = rx_float_range(left, right, *left_inclusive, *right_inclusive)
                                .unwrap();
                            do_test_float_range(
                                &rx,
                                left,
                                right,
                                *left_inclusive,
                                *right_inclusive,
                            );
                        }
                    }
                }
            }
        }
    }
}
