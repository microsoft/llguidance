#!/usr/bin/env python3

import sys
import json
import glob
import os
import random
from jsonschema import Draft202012Validator, validate


output_base = os.environ.get("HOME") + "/src/json-data/output-missing"
combined_base = os.environ.get("HOME") + "/src/json-data/combined"

true = True
false = False


class Stats:

    def __init__(self):
        self.total = 0
        self.server_error = 0
        self.json_error = 0
        self.schema_error = 0
        self.validation_error = 0
        self.json_ok = 0


stats = Stats()


def process_file(schema_file):
    file_base = schema_file.split("/")[-1]
    split = schema_file.split("/")[-2]
    output_name = f"{output_base}/{split}---{file_base}"

    if not os.path.exists(output_name):
        return

    stats.total += 1

    # if stats.total % 100 == 0:
    #     print(stats.__dict__)
    with open(output_name) as f:
        resp = json.loads(f.read())
    if resp.get("error", None):
        stats.server_error += 1
        return
    rs = resp["choices"][0]["message"]["content"]
    try:
        r = json.loads(rs)
    except:
        stats.json_error += 1
        if resp["choices"][0]["finish_reason"] != "length":
            if resp["choices"][0].get("llg_logs", None):
                print("non-length-llg", output_name, schema_file)
            else:
                print("non-length", output_name, schema_file)
        return

    with open(schema_file) as f:
        schema = json.loads(f.read())

    try:
        Draft202012Validator.check_schema(schema)
    except Exception as e:
        stats.schema_error += 1
        print("schema error", output_name, schema_file)
        return

    try:
        validate(r, schema)
    except Exception as e:
        stats.validation_error += 1
        # print("validation error", output_name, schema_file)
        return

    stats.json_ok += 1
    data = {
        "description": "sample " + split + "/" + file_base,
        "_schema_file": schema_file,
        "_output_file": output_name,
        "schema": schema,
        "tests": [
            {
                "description": "llama 70b generated valid",
                "valid": True,
                "data": r,
            }
        ],
    }
    with open(f"{combined_base}/{split}---{file_base}", "w") as f:
        f.write(json.dumps(data, indent=4))


files = []
for arg in sys.argv[1:]:
    if arg.endswith(".json"):
        files.append(arg)
    else:
        files.extend(glob.glob(arg + "/*.json"))
print(len(files))

for f in files:
    process_file(f)

print(stats.__dict__)
