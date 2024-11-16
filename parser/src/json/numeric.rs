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

pub fn rx_int_range(left: Option<i64>, right: Option<i64>) -> String {
    match (left, right) {
        (None, None) => "-?(0|[1-9][0-9]*)".to_string(),
        (Some(left), None) => {
            if left < 0 {
                mk_or(vec![
                    rx_int_range(Some(left), Some(-1)),
                    rx_int_range(Some(0), None),
                ])
            } else {
                mk_or(vec![
                    rx_int_range(
                        Some(left),
                        Some("9".repeat(num_digits(left)).parse::<i64>().unwrap()),
                    ),
                    format!("[1-9][0-9]{{{},}}", num_digits(left)),
                ])
            }
        }
        (None, Some(right)) => {
            if right >= 0 {
                mk_or(vec![
                    rx_int_range(Some(0), Some(right)),
                    rx_int_range(None, Some(-1)),
                ])
            } else {
                format!("-{}", rx_int_range(Some(-right), None))
            }
        }
        (Some(left), Some(right)) => {
            assert!(left <= right);
            if left < 0 {
                if right < 0 {
                    format!("(-{})", rx_int_range(Some(-right), Some(-left)))
                } else {
                    format!(
                        "(-{}|{})",
                        rx_int_range(Some(0), Some(-left)),
                        rx_int_range(Some(0), Some(right))
                    )
                }
            } else {
                if num_digits(left) == num_digits(right) {
                    let l = left.to_string();
                    let r = right.to_string();
                    if left == right {
                        return format!("({})", l);
                    }

                    let lpref = &l[..l.len() - 1];
                    let lx = &l[l.len() - 1..];
                    let rpref = &r[..r.len() - 1];
                    let rx = &r[r.len() - 1..];

                    if lpref == rpref {
                        return format!("({}{}[{}-{}])", lpref, "", lx, rx);
                    }

                    let left_rec: i64 = if lpref.is_empty() {
                        0
                    } else {
                        lpref.parse().unwrap()
                    };
                    let right_rec: i64 = if rpref.is_empty() {
                        0
                    } else {
                        rpref.parse().unwrap()
                    };
                    assert!(left_rec < right_rec);
                    let mut parts = Vec::new();

                    if lx != "0" {
                        parts.push(format!("{}[{}-9]", lpref, lx));
                    }

                    if rx != "9" {
                        parts.push(format!("{}[0-{}]", rpref, rx));
                    }

                    if left_rec <= right_rec {
                        let inner = rx_int_range(Some(left_rec), Some(right_rec));
                        parts.push(format!("{}[0-9]", inner));
                    }

                    mk_or(parts)
                } else {
                    let break_point = 10_i64.pow(num_digits(left) as u32) - 1;
                    mk_or(vec![
                        rx_int_range(Some(left), Some(break_point)),
                        rx_int_range(Some(break_point + 1), Some(right)),
                    ])
                }
            }
        }
    }
}

fn lexi_x_to_9(x: &str, incl: bool) -> String {
    if incl {
        if x.is_empty() {
            "[0-9]*".to_string()
        } else if x.len() == 1 {
            format!("[{}-9][0-9]*", x)
        } else {
            let x0 = x.chars().next().unwrap().to_digit(10).unwrap();
            let x_rest = &x[1..];
            let mut parts = vec![format!(
                "{}{}",
                x.chars().next().unwrap(),
                lexi_x_to_9(x_rest, incl)
            )];
            if x0 < 9 {
                parts.push(format!("[{}-9][0-9]*", x0 + 1));
            }
            mk_or(parts)
        }
    } else {
        if x.is_empty() {
            "[0-9]*[1-9]".to_string()
        } else {
            let x0 = x.chars().next().unwrap().to_digit(10).unwrap();
            let x_rest = &x[1..];
            let mut parts = vec![format!(
                "{}{}",
                x.chars().next().unwrap(),
                lexi_x_to_9(x_rest, incl)
            )];
            if x0 < 9 {
                parts.push(format!("[{}-9][0-9]*", x0 + 1));
            }
            mk_or(parts)
        }
    }
}

fn lexi_0_to_x(x: &str, incl: bool) -> String {
    if x.is_empty() {
        assert!(incl);
        "".to_string()
    } else {
        let x0 = x.chars().next().unwrap().to_digit(10).unwrap();
        let x_rest = &x[1..];

        if !incl && x.len() == 1 {
            assert!(x0 > 0);
            return format!("[0-{}][0-9]*", x0 - 1);
        }

        let mut parts = vec![format!(
            "{}{}",
            x.chars().next().unwrap(),
            lexi_0_to_x(x_rest, incl)
        )];
        if x0 > 0 {
            parts.push(format!("[0-{}][0-9]*", x0 - 1));
        }
        mk_or(parts)
    }
}

fn lexi_range(ld: &str, rd: &str, ld_incl: bool, rd_incl: bool) -> String {
    assert_eq!(ld.len(), rd.len());
    if ld == rd {
        assert!(ld_incl && rd_incl);
        ld.to_string()
    } else {
        let l0 = ld.chars().next().unwrap().to_digit(10).unwrap();
        let r0 = rd.chars().next().unwrap().to_digit(10).unwrap();
        if l0 == r0 {
            let ld_rest = &ld[1..];
            let rd_rest = &rd[1..];
            format!(
                "{}{}",
                ld.chars().next().unwrap(),
                lexi_range(ld_rest, rd_rest, ld_incl, rd_incl)
            )
        } else {
            assert!(l0 < r0);
            let ld_rest = ld[1..].trim_end_matches('0');
            let mut parts = vec![format!(
                "{}{}",
                ld.chars().next().unwrap(),
                lexi_x_to_9(ld_rest, ld_incl)
            )];
            if l0 + 1 < r0 {
                parts.push(format!("[{}-{}][0-9]*", l0 + 1, r0 - 1));
            }
            let rd_rest = rd[1..].trim_end_matches('0');
            if !rd_rest.is_empty() || rd_incl {
                parts.push(format!(
                    "{}{}",
                    rd.chars().next().unwrap(),
                    lexi_0_to_x(rd_rest, rd_incl)
                ));
            }
            mk_or(parts)
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
) -> String {
    match (left, right) {
        (None, None) => "-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?".to_string(),
        (Some(left), None) => {
            if left < 0.0 {
                mk_or(vec![
                    rx_float_range(Some(left), Some(0.0), left_inclusive, false),
                    rx_float_range(Some(0.0), None, true, false),
                ])
            } else {
                let left_int_part = left as i64;
                mk_or(vec![
                    rx_float_range(
                        Some(left),
                        Some(10f64.powi(num_digits(left_int_part) as i32)),
                        left_inclusive,
                        false,
                    ),
                    format!("[1-9][0-9]{{{},}}(\\.[0-9]+)?", num_digits(left_int_part)),
                ])
            }
        }
        (None, Some(right)) => {
            if right == 0.0 {
                let r = format!("-{}", rx_float_range(Some(0.0), None, false, false));
                if right_inclusive {
                    mk_or(vec![r, "0".to_string()])
                } else {
                    r
                }
            } else if right > 0.0 {
                mk_or(vec![
                    format!("-{}", rx_float_range(Some(0.0), None, false, false)),
                    rx_float_range(Some(0.0), Some(right), true, right_inclusive),
                ])
            } else {
                format!(
                    "-{}",
                    rx_float_range(Some(-right), None, right_inclusive, false)
                )
            }
        }
        (Some(left), Some(right)) => {
            assert!(left <= right);
            if left == right {
                if left_inclusive && right_inclusive {
                    format!("({})", float_to_str(left))
                } else {
                    panic!("Empty range");
                }
            } else if left < 0.0 {
                if right < 0.0 {
                    format!(
                        "(-{})",
                        rx_float_range(Some(-right), Some(-left), right_inclusive, left_inclusive)
                    )
                } else {
                    let mut parts = vec![];
                    let neg_part = rx_float_range(
                        Some(0.0),
                        Some(-left),
                        false, // we don't allow -0
                        left_inclusive,
                    );
                    parts.push(format!("(-{})", neg_part));

                    if right > 0.0 || right_inclusive {
                        let pos_part = rx_float_range(
                            Some(0.0),
                            Some(right),
                            true, // always include 0
                            right_inclusive,
                        );
                        parts.push(pos_part);
                    }
                    mk_or(parts)
                }
            } else {
                let l = float_to_str(left);
                let r = float_to_str(right);
                assert!(l != r);
                assert!(!l.contains('e') && !r.contains('e'));

                if !left.is_finite() || !right.is_finite() {
                    panic!("Infinite numbers not supported");
                }

                let mut left_rec: i64 = l.split('.').next().unwrap().parse().unwrap();
                let right_rec: i64 = r.split('.').next().unwrap().parse().unwrap();

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
                        lexi_range(&ld, &rd, left_inclusive, right_inclusive)
                    );
                    if ld.parse::<i64>().unwrap() == 0 {
                        format!("({}({})?)", left_rec, suff)
                    } else {
                        format!("({}{})", left_rec, suff)
                    }
                } else {
                    let mut parts = vec![];
                    if !ld.is_empty() || !left_inclusive {
                        parts.push(format!(
                            "({}\\.{})",
                            left_rec,
                            lexi_x_to_9(&ld, left_inclusive)
                        ));
                        left_rec += 1;
                    }

                    if right_rec - 1 >= left_rec {
                        let inner = rx_int_range(Some(left_rec), Some(right_rec - 1));
                        parts.push(format!("({}(\\.[0-9]+)?)", inner));
                    }

                    if !rd.is_empty() {
                        parts.push(format!(
                            "({}(\\.{})?)",
                            right_rec,
                            lexi_0_to_x(&rd, right_inclusive)
                        ));
                    } else if right_inclusive {
                        parts.push(format!("{}(\\.0+)?", right_rec));
                    }

                    mk_or(parts)
                }
            }
        }
    }
}
