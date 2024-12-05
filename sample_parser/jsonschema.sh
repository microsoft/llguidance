#!/bin/sh

set -e

cargo run --release --bin json_schema_testsuite ../../../JSON-Schema-Test-Suite/tests/draft2020-12/*.json
