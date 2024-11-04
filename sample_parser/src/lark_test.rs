use std::{
    env,
    fs::File,
    io::{Read, Write},
};

use llguidance_parser::lark::{lark_to_llguidance, parse_lark};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <grammar.lark>", args[0]);
        std::process::exit(1);
    }

    let grammar_file = read_file_to_string(&args[1]);
    let r = parse_lark(&grammar_file).unwrap();
    for it in r.iter() {
        println!("{:?}", it);
    }

    let llguidance = lark_to_llguidance(r).unwrap();
    let json = serde_json::to_string_pretty(&llguidance).unwrap();
    // write json to file
    let mut file = File::create("tmp/llguidance.json").expect("Unable to create file");
    file.write_all(json.as_bytes())
        .expect("Unable to write data to file");
    println!("tmp/llguidance.json created");
}

fn read_file_to_string(filename: &str) -> String {
    let mut file = File::open(filename).expect("Unable to open file");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Unable to read file");
    content
}
