#!/usr/bin/env python3

import sys
import json
import glob
import os
import random
import time

import xgrammar as xgr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig

positive_base = os.environ.get("HOME") + "/src/json-data/positive"
output_base = os.environ.get("HOME") + "/src/json-data/xgr_output"


def do_process(file: str):
    with open(file) as f:
        pos_data = json.loads(f.read())

    schema = json.dumps(pos_data["schema"])
    instance = json.dumps(pos_data["tests"][0]["data"], indent=4)

    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    tokens = tokenizer.encode(instance, add_special_tokens=False)

    status = {
        "file": file,
        "ok": False,
        "num_tokens": len(tokens),
        "accepted_tokens": 0,
    }

    try:
        t0 = time.monotonic()
        compiled_grammar = compiler.compile_json_schema(schema, indent=4)
        matcher = xgr.GrammarMatcher(compiled_grammar)
    except Exception as e:
        status["compile_error"] = repr(e)
        return status

    status["compile_time"] = int((time.monotonic() - t0) * 1000)

    t1 = time.monotonic()
    for i, t in enumerate(tokens):
        matcher.fill_next_token_bitmask(token_bitmask)
        ok = matcher.accept_token(t)
        if not ok:
            break
        status["accepted_tokens"] = i + 1

    status["ok"] = status["accepted_tokens"] == len(tokens)
    return status


def process_file(file: str):
    file_base = file.split("/")[-1]
    output_name = f"{output_base}/{file_base}"
    if os.path.exists(output_name):
        return

    status = do_process(file)
    print(status, file=sys.stderr)
    with open(output_name, "w") as f:
        f.write(json.dumps(status, indent=4))


# Get tokenizer info
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
# This can be larger than tokenizer.vocab_size due to paddings
full_vocab_size = config.vocab_size
tokenizer_info = xgr.TokenizerInfo.from_huggingface(
    tokenizer, vocab_size=full_vocab_size
)
compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1)

files = []
for arg in sys.argv[1:]:
    if arg.endswith(".json"):
        files.append(arg)
    else:
        files.extend(glob.glob(arg + "/*.json"))
print(len(files), file=sys.stderr)
random.shuffle(files)

for f in files:
    process_file(f)
