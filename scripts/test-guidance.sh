#!/bin/sh

set -e
cd $(dirname $0)/..

cargo fmt --check

cargo build --locked
cargo test

echo "Running sample_parser"
(cd sample_parser && ./run.sh >/dev/null)

(cd c_sample && make)

pip uninstall -y llguidance || :

if test -z "$CONDA_PREFIX" -a -z "$VIRTUAL_ENV" ; then
    if [ "X$CI" = "Xtrue" ]; then
        echo "Building in CI with pip"
        pip install -v -e .
    else
        echo "No conda and no CI"
        exit 1
    fi
else
    maturin develop --release
fi

PYTEST_FLAGS=

if test -f ../guidance/tests/unit/test_ll.py ; then
    echo "Guidance side by side"
    cd ../guidance
else
    mkdir -p tmp
    cd tmp
    PYTEST_FLAGS=-v
    if test -f guidance/tests/unit/test_ll.py ; then
        echo "Guidance clone OK"
    else
        git clone -b llg_0_5_1 https://github.com/guidance-ai/guidance
    fi
    cd guidance
    echo "Branch: $(git branch --show-current), Remote URL: $(git remote get-url origin), HEAD: $(git rev-parse HEAD)"
fi

python -m pytest $PYTEST_FLAGS tests/unit/test_ll.py # main test
python -m pytest $PYTEST_FLAGS tests/unit/test_[lgmp]*.py tests/unit/library "$@"
