use std::{env, fs::File, io::Read};

use llguidance_parser::earley::lark::lex_lark;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <grammar.lark>", args[0]);
        std::process::exit(1);
    }

    let grammar_file = read_file_to_string(&args[1]);
    let _tokens = lex_lark(&grammar_file).unwrap();
}

fn read_file_to_string(filename: &str) -> String {
    let mut file = File::open(filename).expect("Unable to open file");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Unable to read file");
    content
}
