#!/bin/sh

if [ -z "$1" ]; then
    LARK=data/lark.lark
fi

cargo run --bin lark_test -- $LARK "$@"
