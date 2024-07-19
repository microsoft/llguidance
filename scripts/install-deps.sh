#!/bin/sh

# installing guidance for deps
pip install pytest guidance huggingface_hub tokenizers jsonschema maturin
pip uninstall -y guidance
