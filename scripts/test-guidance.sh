#!/bin/sh

set -e
cd $(dirname $0)/..

pip uninstall -y llguidance || :

if test -z "$CONDA_PREFIX" ; then
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
    if test -f guidance/tests/unit/test_ll.py ; then
    echo "Guidance clone OK"
    PYTEST_FLAGS=-v
    else
        git clone -b lazy_grammars https://github.com/hudson-ai/guidance
    fi
    cd guidance
fi

python -m pytest $PYTEST_FLAGS tests/unit/test_ll.py # main test
python -m pytest $PYTEST_FLAGS tests/unit/test_[lgmp]*.py tests/unit/library
