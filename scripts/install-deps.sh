#!/bin/sh

# installing guidance for deps
pip install pytest guidance huggingface_hub tokenizers jsonschema maturin[zig] \
    torch transformers bitsandbytes ipython
pip uninstall -y guidance
