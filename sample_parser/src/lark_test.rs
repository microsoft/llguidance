use std::{
    env,
    fs::File,
    io::{Read, Write},
};

use anyhow::Result;
use llguidance_parser::lark_to_llguidance;

fn process_file(filename: &str) -> Result<()> {
    let grammar_file = read_file_to_string(filename);
    let llguidance = lark_to_llguidance(&grammar_file)?;
    let json = serde_json::to_string_pretty(&llguidance).unwrap();
    // write json to file
    let mut file = File::create("tmp/llguidance.json").expect("Unable to create file");
    file.write_all(json.as_bytes())
        .expect("Unable to write data to file");
    println!("{} OK", filename);
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <grammar.lark...>", args[0]);
        std::process::exit(1);
    }

    for filename in &args[1..] {
        if let Err(e) = process_file(filename) {
            eprintln!("Error: {} {}", filename, e);
        }
    }
}

fn read_file_to_string(filename: &str) -> String {
    let mut file = File::open(filename).expect("Unable to open file");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Unable to read file");
    content
}
