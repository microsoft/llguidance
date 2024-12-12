#!/usr/bin/env python3

import sys
import json
import glob
import os
import random
import re
from jsonschema import Draft202012Validator, validate

prev_base = os.environ.get("HOME") + "/src/json-data/prev_tests"
output_base = os.environ.get("HOME") + "/src/json-data/new_tests"

class Stats:
    def __init__(self):
        self.files = 0
        self.responses = 0
        self.server_error = 0
        self.json_error = 0
        self.json_error_non_length = 0
        self.validation_error = 0
        self.invalidation_error = 0
        self.not_negative = 0
        self.negative_added = 0


stats = Stats()


def process_file(file_name):
    file_base = file_name.split("/")[-1]

    stats.files += 1

    with open(file_name) as f:
        inp = json.loads(f.read())
    
    pos_data = inp["pos_data"]

    prev_file = f"{prev_base}/{file_base}"
    if os.path.exists(prev_file):
        with open(f"{prev_base}/{file_base}") as f:
            pos_data = json.loads(f.read())

    schema = pos_data["schema"]
    tests = pos_data["tests"]

    Draft202012Validator.check_schema(schema)

    for idx, test in enumerate( tests ):
        try:
            validate(test["data"], schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
            if not test["valid"]:
                print("positive already there", file_name, idx)
                stats.invalidation_error += 1
        except Exception as e:
            if test["valid"]:
                stats.validation_error += 1
                print("validation error", file_name, idx, repr(e))

    for idx, resp in  enumerate(inp["responses"]):
        stats.responses += 1

        if resp.get("error", None):
            stats.server_error += 1
            continue

        rs = resp["choices"][0]["message"]["content"]
        try:
            r = json.loads(rs)
        except:
            stats.json_error += 1
            if resp["choices"][0]["finish_reason"] != "length":
                stats.json_error_non_length += 1
                if resp["choices"][0].get("llg_logs", None):
                    print("non-length-llg", file_name, idx)
                else:
                    print("non-length", file_name, idx)
            continue

        try:
            validate(r, schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
            # print("not negative", file_name, idx)
            stats.not_negative += 1
            continue
        except Exception as e:
            # good
            pass

        stats.negative_added += 1

        # f"violate a constraint introduced by {f} in the schema"
        prompt = resp["expanded_prompt"]
        description = "llama-70b generated negative"
        m = re.search(r"violate a constraint introduced by (.+?) in the schema", prompt)
        if m:
            description += "; focus on " + m.group(1)
        
        tests.append({
            "description": description,
            "valid": False,
            "data": r,
        })

    with open(f"{output_base}/{file_base}", "w") as f:
        f.write(json.dumps(pos_data, indent=4))


files = []
for arg in sys.argv[1:]:
    if arg.endswith(".json"):
        files.append(arg)
    else:
        files.extend(glob.glob(arg + "/*.json"))
print(len(files))

for idx, f in enumerate( files ):
    if idx % 500 == 0:
        print(idx, stats.__dict__)
    process_file(f)

print(stats.__dict__)
