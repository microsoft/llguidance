#!/bin/sh

cargo run data/blog.schema.ll.json data/blog.sample.json
# cargo run data/blog.schema.json data/blog.sample.json
# cargo run --release --bin minimal data/blog.schema.json data/blog.sample.json
# mkdir -p tmp
# strip -o tmp/minimal ../../target/release/minimal
# ls -l ../../target/release/minimal tmp/minimal
