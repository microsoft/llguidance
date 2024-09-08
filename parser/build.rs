extern crate cbindgen;

use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let mut config = cbindgen::Config::default();
    config.language = cbindgen::Language::C;
    config.cpp_compat = true;
    config.usize_is_size_t = true; // not exposed as .with_*() method

    cbindgen::Builder::new()
        .with_config(config)
        .with_include_guard("LLGUIDANCE_H")
        .with_crate(crate_dir)
        .generate()
        .map_or_else(
            |error| match error {
                cbindgen::Error::ParseSyntaxError { .. } => {}
                e => panic!("{:?}", e),
            },
            |bindings| {
                bindings.write_to_file("llguidance.h");
            },
        );
}
