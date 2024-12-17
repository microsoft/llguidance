use sample_parser::*;
use serde_json::json;

#[test]
fn test_ll_skip() {
    check_lark_grammar(
        r#"start: "A" "!"
           %ignore /[ \t]+/"#,
        &["A", " ‧ ‧!"],
    );

    check_lark_grammar(
        r#"
            start: "A: " NUMBER
            NUMBER: /[0-9]+/
            %ignore /[ \t]+/
        "#,
        &["A‧:", " ‧ ‧5‧6‧≺EOS≻"],
    );

    check_lark_grammar_nested(
        r#"start: "." @sub"#,
        r#"start: "A" "!"
           %ignore /[ \t]+/"#,
        &[".‧A", " ‧ ‧!"],
    );
}

#[test]
fn test_ll_format() {
    check_lark_json(
        r#"start: "JSON" @sub
        "#,
        json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "string",
                    "format": "date-time"
                }
            }
        }),
        &[
            "JSON",
            "{\"‧a‧\":‧ ‧\"‧2‧0‧2‧0",
            "-",
            "0‧2",
            "-",
            "2‧9‧T‧1‧0",
            ":",
            "3‧3",
            ":",
            "2‧2‧Z‧\"‧}",
        ],
    );

    check_lark_json(
        r#"start: "JSON" @sub
        "#,
        json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "string",
                    "format": "date"
                }
            }
        }),
        &[
            "JSON",
            "{\"‧a‧\":‧ ‧\"‧2‧0‧2‧0",
            "-",
            "0‧2",
            "-",
            "2‧9‧\"‧}",
        ],
    );
}

#[test]
fn test_ll_json() {
    // basic JSON parsing
    check_lark_json(
        r#"start: "JSON" @sub
        "#,
        json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "number"
                }
            }
        }),
        &["JSON", "{\"‧a‧\":‧ ‧5‧}"],
    );

    // check for forcing the field name
    check_lark_json(
        r#"start: "JSON" @sub
        "#,
        json!({
            "type": "object",
            "properties": {
                "a_long_prop_name": {
                    "type": "number"
                }
            },
            "required": ["a_long_prop_name"]
        }),
        &["JSON", "{\"", "a‧_‧long‧_‧prop‧_‧name", "\":‧ ‧5‧}"],
    );

    check_lark_json(
        r#"start: "JSON" @sub "END"
        "#,
        json!({
            "type": "array"
        }),
        &["JSON", "[‧1‧,‧2‧,‧3‧,‧4‧,‧5‧,‧6‧,‧7‧,‧8‧]", "END"],
    );

    // again, off by one
    let c = check_lark_json(
        r#"start: "JSON" j "END"
               j[max_tokens=3]: @sub
            "#,
        json!({
            "type": "array"
        }),
        &["JSON", "[‧1‧,‧2", "END"],
    );
    check_capture(&c, "j", "[1,2");

    let c = check_lark_json(
        r#"start: "JSON" j
               j[max_tokens=3]: @sub
            "#,
        json!({
            "type": "array"
        }),
        &["JSON", "[‧1‧,‧2"],
    );
    check_capture(&c, "j", "[1,2");
}

#[test]
fn test_ll_subgrammar_max_tokens() {
    // TODO test this - should return an error from prompt processing
    // check_lark_grammar(
    //     r#"start: " x" aa " y"
    //        aa: " a" aa
    //        "#,
    //     &[" x", " a‧ a‧ a‧ a‧ b", " y"],
    // );

    // voluntary stop of the subgrammar
    for max_tokens in &[3, 4, 5] {
        let c = check_lark_grammar_nested(
            &format!(
                r#"start: " x x x" (" q")* " x" ab " y"
                   ab[max_tokens={}]: @sub
                "#,
                max_tokens,
            ),
            r#"start: (" a")* " b""#,
            &[" x‧ x‧ x", " q‧ q‧ q‧ q‧ x‧ a‧ a‧ b", " y"],
        );
        check_capture(&c, "ab", " a a b");

        // no unique start marker
        let c = check_lark_grammar_nested(
            &format!(
                r#"start: " x x x" (" q")* ab " y"
                   ab[max_tokens={}]: @sub
                "#,
                max_tokens,
            ),
            r#"start: (" a")* " b""#,
            &[" x‧ x‧ x", " q‧ q‧ q‧ q‧ a‧ a‧ b", " y"],
        );
        check_capture(&c, "ab", " a a b");
    }

    // forced stop of the subgrammar
    let c = check_lark_grammar_nested(
        r#"start: " x x x" (" q")* " x" ab " y"
           ab[max_tokens=3]: @sub
        "#,
        r#"start: (" a")* " b""#,
        &[" x‧ x‧ x", " q‧ q‧ q‧ q‧ x‧ a‧ a‧ a", " y"],
    );
    check_capture(&c, "ab", " a a a");
    // and with no unique start marker
    let c = check_lark_grammar_nested(
        r#"start: " x x x" (" q")* ab " y"
           ab[max_tokens=3]: @sub
        "#,
        r#"start: (" a")* " b""#,
        &[" x‧ x‧ x", " q‧ q‧ q‧ q‧ a‧ a‧ a", " y"],
    );
    check_capture(&c, "ab", " a a a");

    // TODO we're off by one here
    let c = check_lark_grammar_nested(
        r#"start: " x x x" ab " y"
           ab[max_tokens=2]: @sub
        "#,
        r#"start: (" a")* " b""#,
        &[" x‧ x‧ x", " a‧ a‧ a", " y"],
    );
    check_capture(&c, "ab", " a a a");

    // TODO we're off by one here
    let c = check_lark_grammar_nested(
        r#"start: ab " y"
           ab[max_tokens=2]: @sub
        "#,
        r#"start: (" a")* " b""#,
        &["", " a‧ a‧ a", " y"],
    );
    check_capture(&c, "ab", " a a a");
}

#[test]
fn test_ll_lexeme_subgrammar_max_tokens() {
    check_lark_grammar_nested(
        r#"start: " x" ab " y"
           ab[max_tokens=3]: @sub
        "#,
        r#"start: TEXT
           TEXT: (" a")* " b"
        "#,
        &[" x", " a‧ a‧ a", " y"],
    );

    // TODO check_tokens() should increment token_idx and we should somehow test it
}

#[test]
fn test_ll_temperature() {
    check_lark_grammar_nested(
        r#"start: /[xy]/ sub_temp
           sub_temp[temperature=0.5]: @sub
        "#,
        r#"start: "[" ("A")* "]"
           %ignore /[ \t]+/"#,
        &["", "x‧[‧]"],
    );

    check_lark_grammar_nested(
        r#"start: sub_temp
           sub_temp[temperature=0.5]: @sub
        "#,
        r#"start: "[" ("A")* "]"
           %ignore /[ \t]+/"#,
        &["", "[‧]"],
    );

    check_lark_grammar_nested(
        r#"start: sub_temp
           sub_temp[temperature=0.5]: @sub
        "#,
        r#"start: "[" ("A")* "]"
           %ignore /[ \t]+/"#,
        &["", "[]"],
    );

    check_lark_grammar_nested(
        r#"start: sub_temp
           sub_temp[temperature=0.5]: @sub
        "#,
        r#"start: "[" ("A")* "]"
        "#,
        &["", "[‧]"],
    );
}

#[test]
fn test_ll_backtrack_stop() {
    check_lark_grammar(
        r#"
            start: "Count to 10: 1, 2, 3, 4, 5, 6, 7, " text "\nNot quite."
            text[stop=","]: /.+/
        "#,
        &[
            "Count‧ to‧ ‧1‧0‧:‧ ‧1‧,‧ ‧2‧,‧ ‧3‧,‧ ‧4‧,‧ ‧5‧,‧ ‧6‧,‧ ‧7‧,",
            " ‧8‧,",
            "1↶\n‧Not‧ quite‧.",
        ],
    );

    check_lark_grammar(
        r#"
            start: "Name: " name "\nName: " name
            name[stop=STOP]: /E[a-z]+/
            STOP: /[a-b]/ | /[x-z]/
        "#,
        &["Name‧:", " Em‧ily", "1↶il‧\n‧Name‧:", " Emil‧ie‧a", "1↶"],
    );
}

#[test]
fn test_llparser() {
    check_lark_grammar_prompt(
        r#"
            start: gen
            gen[stop=""]: /.*/
        "#,
        "2 + 2 =",
        &["2‧ +‧ ‧2", " =>‧ ‧4‧≺EOS≻"],
    );

    check_lark_grammar(
        r#"
            start: "Power frequency is " num "Hz; voltage is " num "V"
            num[stop="", max_tokens=5]: /[0-9]+/
        "#,
        &[
            "Power‧ frequency‧ is‧ ",
            "5‧0‧Hz", // no EoS needed on 50Hz
            ";‧ voltage‧ is‧ ",
            "2‧2‧0‧V",
        ],
    );

    check_lark_grammar(
        r#"
            start: "Power frequency is " num "Hz; voltage is " num "V"
            num[stop="", max_tokens=3]: /[0-9]+/
        "#,
        &[
            "Power‧ frequency‧ is‧ ",
            "5‧0‧Hz", // no EoS needed on 50Hz
            ";‧ voltage‧ is‧ ",
            "2‧2‧0",
            "V", // V is forced since max_tokens=3
        ],
    );

    check_lark_grammar(
        r#"
            start: "Q: Are dolphins fish?\nA: " ANSWER "\nQ: Are sharks fish?\nA: " ANSWER
            ANSWER: "Yes" | "No"
        "#,
        &[
            "Q‧:‧ Are‧ dol‧ph‧ins‧ fish‧?‧\n‧A‧:",
            " No", // note the prefix space - moved by token healing
            "\n‧Q‧:‧ Are‧ sh‧arks‧ fish‧?‧\n‧A‧:",
            " Yes",
        ],
    );

    check_lark_grammar(
        r#"
            start: "Q: 7 * 8\nA: " NUMBER
            NUMBER: /[0-9]+/
        "#,
        &["Q‧:‧ ‧7‧ *‧ ‧8‧\n‧A‧:‧ ", "5‧6‧≺EOS≻"],
    );
}

#[test]
fn test_ll_nullable_lexeme() {
    // make sure 'a' is not forced
    check_lark_grammar(
        r#"start: gen
           gen[stop=""]: /a*/"#,
        &["", "a‧≺EOS≻"],
    );

    // this one doesn't work - no lexeme was scanned by EOS, so we allow more lexemes...
    check_lark_grammar(
        r#"start: gen
           gen[stop=""]: /a*/"#,
        &["", "≺EOS≻"],
    );

    // see that we can skip 5*
    check_lark_grammar(
        r#"start: "6 * 7 = " five_seq num "\n"
           five_seq[stop=""]: /5*/
           num[stop=""]: /[1-4][0-9]/"#,
        &["6‧ *‧ ‧7‧ =‧ ", "4‧2", "\n"],
    );

    check_lark_grammar_nested(
        r#"start: "Here: 2 + 2 = " @sub"#,
        r#"start: /[0-9]+/"#,
        &["Here‧:‧ ‧2‧ +‧ ‧2‧ =‧ ", "4‧≺EOS≻"],
    );

    // make sure it stops at EOS
    check_lark_grammar_nested(
        r#"start: "Here: 2 + 2 = " @sub"#,
        r#"start: num q
           num: /[0-9]+/
           q: /Q?/
        "#,
        &["Here‧:‧ ‧2‧ +‧ ‧2‧ =‧ ", "4‧≺EOS≻"],
    );

    let float_grammar = r#"
        start: num1 | num2
        num1: /-?(?:0|[1-9][0-9]*)/
        num2: /-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)/
    "#;
    check_lark_grammar_nested(r#"start: @sub"#, &float_grammar, &["", "1‧≺EOS≻"]);
    check_lark_grammar_nested(r#"start: @sub"#, &float_grammar, &["", "0‧≺EOS≻"]);
    check_lark_grammar_nested(r#"start: @sub"#, &float_grammar, &["", "1‧.‧1‧≺EOS≻"]);
    check_lark_grammar_nested(r#"start: @sub"#, &float_grammar, &["", "0‧.‧1‧≺EOS≻"]);
}

#[test]
fn test_ll_pop_tokens() {
    // check_grammar(grm, ["6‧ *‧ ‧7‧ =‧ ", "4‧2‧\n"])
    // grm = "6 * 7 = " + subgrammar(body=lexeme("[0-9]{1,3}")) + "\n"
    check_lark_grammar(
        r#"start: "6 * 7 = " NUM "\n"
           NUM: /[0-9]{1,3}/
        "#,
        &["6‧ *‧ ‧7‧ =‧ ", "4‧2‧\n"],
    );
}

#[test]
fn test_ll_nice_man() {
    let grm = r#"start: ("a" | "ab" | "c")"#;
    let grm_d = r#"start: ("a" | "ab" | "c") ("d")"#;
    let grm_opt_d = r#"start: ("a" | "ab" | "c") ("d" | "")"#;

    check_lark_grammar(grm, &["", "a‧b"]);
    check_lark_grammar(grm, &["", "a‧≺EOS≻"]);
    check_lark_grammar(grm_d, &["", "a‧d"]);
    check_lark_grammar(grm_d, &["", "a‧b", "d"]);

    check_lark_grammar(grm_opt_d, &["", "a‧b‧d"]);
    check_lark_grammar(grm_opt_d, &["", "a‧b‧≺EOS≻"]);
    check_lark_grammar(grm_opt_d, &["", "a‧≺EOS≻"]);

    // TODO: this should also work for "abq" as a single lexeme
    // https://github.com/guidance-ai/llguidance/issues/2
    let abq = r#"start: ("a" | "a" "bq" | "c") ("bQ" | "")"#;
    check_lark_grammar(abq, &["", "a‧b‧q‧≺EOS≻"]);
    check_lark_grammar(abq, &["", "a‧b‧Q"]);
}

#[test]
fn test_ll_stop_quote_comma() {
    let grm = r#"
        start: "{ \"items\": [\"" ap "\",\n   \"" bp "\"] }"
        ap[stop="\""]: /a+/
        bp[stop="\""]: /b+/
    "#;

    // make sure we allow ", as a single token; also "]
    check_lark_grammar(
        grm,
        &["{‧ \"‧items‧\":‧ [\"", "a‧\",", "\n‧  ‧ \"", "b‧\"]", " }"],
    );

    // and as seprate tokens
    check_lark_grammar(
        grm,
        &[
            "{‧ \"‧items‧\":‧ [\"",
            "a‧\"",
            ",‧\n‧  ‧ \"",
            "b‧\"",
            "]‧ }",
        ],
    );
}

#[test]
fn test_ll_nullable_bug() {
    check_lark_grammar(
        r#"start: (maybe_a maybe_a maybe_a maybe_a | "foo")
           maybe_a: "a" | ""
        "#,
        &["", "a‧≺EOS≻"],
    );
}

#[test]
fn test_ll_max_tokens() {
    check_lark_grammar(
        r#"start: "Name: " name " Height: " height
           name[max_tokens=3, stop=""]: /.*/
           height[max_tokens=3, stop=""]: /.*/
        "#,
        &["Name‧:", " Em‧ily‧ Carter", " Height‧:", " ‧5‧'‧6"],
    );

    // here we have two gen() with the same regex (so they are the same lexeme)
    // but different max_tokens limits
    check_lark_grammar(
        r#"start: "Name: " name " Height: " height
           name[max_tokens=2, stop=""]: /.*/
           height[max_tokens=3, stop=""]: /.*/
        "#,
        &["Name‧:", " Em‧ily", " Height‧:", " ‧5‧'‧6"],
    );

    // now this is a strange case, where gen() is allowed together with the following
    // string, and gen() runs out of tokens, so the fixed string takes over
    // note how Emily is not repeated
    check_lark_grammar(
        r#"start: "Name: " name "Emily Carter is great; Height: " height
           name[max_tokens=2, stop=""]: /.*/
           height[max_tokens=3, stop=""]: /.*/
        "#,
        &[
            "Name‧:",
            " Em‧ily",
            " Carter‧ is‧ great‧;‧ Height‧:",
            " ‧5‧'‧6",
        ],
    );
}
