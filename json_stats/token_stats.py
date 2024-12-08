#!/usr/bin/env python3

import sys
import json
import glob
import os
import random
import re
from jsonschema import Draft202012Validator, validate

class Stats:
    def __init__(self):
        self.files = 0
        self.responses = 0
        self.server_error = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        

stats = Stats()


def process_file(file_name):
    file_base = file_name.split("/")[-1]

    stats.files += 1

    with open(file_name) as f:
        inp = json.loads(f.read())
    
    pos_data = inp["pos_data"]

    for idx, resp in  enumerate(inp["responses"]):
        stats.responses += 1

        if resp.get("error", None):
            stats.server_error += 1
            print("server error", file_name, idx)
            continue

        if resp["choices"][0]["finish_reason"] == "length":
            print("length", file_name, idx)
            continue

        usage = resp["usage"]
        stats.completion_tokens += usage["completion_tokens"]
        stats.prompt_tokens += usage["prompt_tokens"]


files = []
for arg in sys.argv[1:]:
    if arg.endswith(".json"):
        files.append(arg)
    else:
        files.extend(glob.glob(arg + "/*.json"))
print(len(files))

for idx, f in enumerate( files ):
    if idx % 1000 == 0:
        print(idx, stats.__dict__)
    process_file(f)

print(stats.__dict__)
