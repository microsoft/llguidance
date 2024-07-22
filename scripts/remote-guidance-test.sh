#!/bin/sh
cd $(dirname $0)/..
cd ../guidance
pytest --selected_model azure_guidance -v --durations=10 tests/need_credentials/test_azure_guidance.py "$@"