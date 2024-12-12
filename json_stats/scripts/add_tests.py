#!/usr/bin/env python3

import sys
import json
import glob
import os
import random
import re
from jsonschema import Draft202012Validator, validate

combined_base = os.environ.get("JSB_DATA") + "/unique_tests"
invalid_marker = "Please modify the instance to make it invalid according to the schema"


class Stats:
    def __init__(self):
        self.files = 0
        self.responses = 0
        self.duplicate_test = 0
        self.server_error = 0
        self.json_error = 0
        self.json_error_non_length = 0
        self.validation_error = 0
        self.invalidation_error = 0
        self.not_negative = 0
        self.tests_added = 0


stats = Stats()


def process_file(output_name):
    file_base = output_name.split("/")[-1]

    stats.files += 1

    combined_name = f"{combined_base}/{file_base}"

    if not os.path.exists(combined_name):
        print(f"File {combined_name} does not exist")
        return

    with open(output_name) as f:
        resp_file = json.loads(f.read())

    resps = [resp_file]
    if "responses" in resp_file:
        resps = resp_file["responses"]

    with open(combined_name) as f:
        try:
            combined = json.loads(f.read())
        except:
            print("Failed to load", combined_name)
            return

    existing_tests = set(json.dumps(t["data"]) for t in combined["tests"])

    schema = combined["schema"]
    # Draft202012Validator.check_schema(schema)

    num_added = 0

    for resp in resps:
        if resp.get("error", None):
            stats.server_error += 1
            continue

        rs = resp["choices"][0]["message"]["content"]
        try:
            r = json.loads(rs)
        except:
            stats.json_error += 1
            if resp["choices"][0]["finish_reason"] != "length":
                if resp["choices"][0].get("llg_logs", None):
                    print("non-length-llg", output_name, combined_name)
                else:
                    print("non-length", output_name, combined_name)
            continue

        if json.dumps(r) in existing_tests:
            stats.duplicate_test += 1
            continue

        test_is_valid = True
        prompt = resp.get("expanded_prompt", "")

        resp_meta = resp.get("meta", {})
        resp_meta_valid = resp_meta.get("valid", None)
        if resp_meta_valid is not None:
            test_is_valid = resp_meta_valid
        else:
            if invalid_marker in prompt:
                test_is_valid = False

        test = {
            "description": "llama 70b generated positive",
            "valid": test_is_valid,
            "data": r,
        }

        if not test_is_valid:
            description = "llama-70b generated negative"
            m = re.search(
                r"violate a constraint introduced by (.+?) in the schema", prompt
            )
            if m:
                description += "; focus on " + m.group(1)
            test["description"] = description

        try:
            validate(r, schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
            if not test_is_valid:
                stats.invalidation_error += 1
                # print("invalidation error", output_name, combined_name)
                continue
        except Exception as e:
            if test_is_valid:
                stats.validation_error += 1
                # print("validation error", output_name, combined_name)
                continue
            else:
                test["python_error"] = str(e)

        combined["tests"].append(test)
        stats.tests_added += 1
        num_added += 1

    if num_added > 0:
        try:
            combined_bytes = json.dumps(combined, indent=2, ensure_ascii=False).encode("utf8")
        except Exception as e:
            # this may fail due to utf8 encoding issues
            print("Failed to dump", combined_name, e)
            return
        with open(combined_name, "bw") as f:
            f.write(combined_bytes)


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
