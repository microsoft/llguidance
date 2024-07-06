#!/bin/sh

set -e
cd $(dirname $0)/..

pip uninstall -y llguidance || :
pip install -e . -v --no-build-isolation

mkdir -p tmp
cd tmp
if test -f guidance/tests/unit/test_ll.py ; then
   echo "Guidance clone OK"
else
    git clone -b lazy_grammars https://github.com/hudson-ai/guidance
fi
cd guidance
python -m pytest -v tests/unit/test_ll.py # main test
python -m pytest -v tests/unit/test_[lgmp]*.py tests/unit/library
