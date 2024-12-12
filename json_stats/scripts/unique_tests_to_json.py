#!/usr/bin/env python3

import sys
import json
import glob
import os
import random
import re
import copy
from jsonschema import Draft202012Validator, validate


def process_file(file_name):
    with open(file_name) as f:
        pos_data = json.loads(f.read())

    schema = pos_data["schema"]
    tests = pos_data["tests"]

    json_filename = file_name.replace("unique_tests", "json").replace("---", "/")

    with open(json_filename) as f:
        schema_2 = json.loads(f.read())

    if json.dumps(schema) != json.dumps(schema_2):
        with open(json_filename, "w") as f:
            f.write(json.dumps(schema, indent=4, ensure_ascii=False))


files = []
for arg in sys.argv[1:]:
    if arg.endswith(".json"):
        files.append(arg)
    else:
        files.extend(glob.glob(arg + "/*.json"))

for idx, f in enumerate(files):
    process_file(f)
