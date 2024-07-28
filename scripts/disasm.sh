#!/bin/sh

TRG=`rustup show | head -1 | sed -e 's/.*: //'`
#CRATE=`grep "^name =" Cargo.toml  | head -1 | sed -e 's/.*= "//; s/"//'`
CRATE=$1
shift
RUSTFLAGS="--emit asm" cargo build --release --target $TRG
TRG_DIR=`cargo metadata --format-version 1 | jq -r '.target_directory'`
F=`echo $TRG_DIR/$TRG/release/deps/$CRATE-*.s`
# if $F has more than one file
if [ `echo $F | wc -w` -gt 1 ]; then
    echo "More than one file found: $F; removing; try again"
    rm -f $F
    exit 1
fi

mkdir -p tmp
cp $F tmp/full.s
HERE=$(dirname $0)
node $HERE/annotate_asm.js tmp/full.s "$@" | rustfilt -h > tmp/func.s
ls -l tmp/func.s
