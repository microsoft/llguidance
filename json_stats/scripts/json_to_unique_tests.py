#!/usr/bin/env python3

import sys
import json
import glob
import os
import random
import re
from jsonschema import Draft202012Validator, validate

combined_base = os.environ.get("JSB_DATA") + "/unique_tests"


class Stats:
    def __init__(self):
        self.files = 0
        self.responses = 0
        self.server_error = 0
        self.schema_error = 0
        self.json_error = 0
        self.json_error_non_length = 0
        self.validation_error = 0
        self.invalidation_error = 0
        self.not_negative = 0
        self.negative_added = 0


stats = Stats()


def process_file(file_name):
    file_base = file_name.split("/")[-1]
    split = file_name.split("/")[-2]

    stats.files += 1

    with open(file_name) as f:
        schema = json.loads(f.read())

    combined_name = f"{combined_base}/{split}---{file_base}"
    combined = {
        "description": f"sample {split}/{file_base}",
        "schema": schema,
        "tests": [],
    }

    if os.path.exists(combined_name):
        with open(combined_name, "r") as f:
            combined = json.loads(f.read())
            prev = json.dumps(combined["schema"])
            combined["schema"] = schema
            if json.dumps(combined["schema"]) == prev:
                return  # no change

    try:
        Draft202012Validator.check_schema(schema)
    except Exception as e:
        stats.schema_error += 1
        print(f"Error in {file_name}: {repr(e)}")
        return

    print(f"Adding {file_name} to {combined_name}")

    with open(combined_name, "w") as f:
        f.write(json.dumps(combined, indent=2, ensure_ascii=False))


files = []
for arg in sys.argv[1:]:
    if arg.endswith(".json"):
        files.append(arg)
    else:
        files.extend(glob.glob(arg + "/*.json"))
print(len(files))

for idx, f in enumerate(files):
    if idx % 500 == 0:
        print(idx, stats.__dict__)
    process_file(f)

print(stats.__dict__)
