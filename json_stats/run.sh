#!/bin/sh

if [ -z "$1" ]; then
    DEFAULT_ARGS=$JSB_DATA/unique_tests
else
    DEFAULT_ARGS=
fi  

cargo run --release $DEFAULT_ARGS "$@"
