#!/usr/bin/env python3

import sys
import json
import glob
import os
import random
import re
from jsonschema import Draft202012Validator, validate

output_base = os.environ.get("HOME") + "/src/json-data/unique_tests"


class Stats:
    def __init__(self):
        self.files = 0
        self.duplicates = 0
        self.pos_tests = 0
        self.neg_tests = 0
        self.files_without_neg = 0


stats = Stats()


def process_file(file_name):
    file_base = file_name.split("/")[-1]

    stats.files += 1

    with open(file_name) as f:
        inp = json.loads(f.read())

    pos_data = inp
    schema = pos_data["schema"]
    tests = pos_data["tests"]

    existing_tests = set()
    tests_copy = []

    num_neg = 0
    for t in tests:
        key = json.dumps(t["data"])
        if key in existing_tests:
            stats.duplicates += 1
            continue
        existing_tests.add(key)
        tests_copy.append(t)
        if t["valid"]:
            stats.pos_tests += 1
        else:
            stats.neg_tests += 1
            num_neg += 1

    if num_neg == 0:
        stats.files_without_neg += 1

    pos_data["tests"] = tests_copy
    with open(f"{output_base}/{file_base}", "w") as f:
        f.write(json.dumps(pos_data, indent=2))


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
