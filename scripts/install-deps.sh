#!/bin/sh

# installing guidance for deps
pip install setuptools==68 setuptools-rust pytest guidance huggingface_hub tokenizers jsonschema wheel twine build
pip uninstall -y guidance

