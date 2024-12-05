#!/bin/sh

if [ -z "$1" ]; then
    DEFAULT_ARGS=~/src/JSONSchemaBench/json/*
else
    DEFAULT_ARGS=
fi  

cargo run --release $DEFAULT_ARGS "$@"
