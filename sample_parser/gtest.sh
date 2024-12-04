#!/bin/sh

ARGS=
FINAL_ARGS=
while [ $# -gt 0 ]; do
  case "$1" in
    -s)
      FINAL_ARGS="-- --nocapture"
      ;;
    *)
      ARGS="$ARGS $1"
      ;;
  esac
  shift
done

RUST_BACKTRACE=1 cargo test -F llguidance/logging $ARGS $FINAL_ARGS
