#!/bin/sh

if [ -z "$1" ]; then
    DEFAULT_ARGS=$JSB_DATA/unique_tests
else
    DEFAULT_ARGS=
fi

if [ -z "$PERF" ]; then
    cargo run --release $DEFAULT_ARGS "$@"
else
    PERF='perf record -F 999 -g'
    RUSTFLAGS='-C force-frame-pointers=y' cargo build --profile perf
    $PERF ../target/perf/json_stats $DEFAULT_ARGS "$@"
    echo "perf report -g graph,0.05,caller"
    echo "perf report -g graph,0.05,callee"
fi
