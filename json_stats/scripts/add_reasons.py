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
                print("validation error", file_name, idx, str(e))
            else:
                test["python_error"] = str(e)

    with open(file_name, "w") as f:
        f.write(json.dumps(pos_data, indent=2, ensure_ascii=False))


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
