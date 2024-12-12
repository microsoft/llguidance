#!/usr/bin/env python3

import requests
import sys
import json
import glob
import os
import random
import concurrent.futures


url = "http://localhost:3001/v1/chat/completions"

true = True
false = False

output_base = os.environ.get("HOME") + "/src/json-data/output"

def gen_example(schema):
    prompt = (
        "Please generate an example data that validates against the following schema:\n"
        + json.dumps(schema, indent=4)
    )
    if len(prompt) > 50_000:
        return { "error": f"Prompt too long, {len(prompt)}" }
    req = {
        "model": "model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "return_expanded_prompt": true,
        "include_json_schema_in_prompt": false,
        "llg_log_level": "verbose",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "strict": true,
                "schema": schema,
            },
        },
        "max_tokens": 1000,
        "temperature": 0.2,
    }

    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=req)
    # response.raise_for_status()
    return response.json()


def process_file(file):
    file_base = file.split("/")[-1]
    output_name = f"{output_base}/{file_base}"
    if os.path.exists(output_name):
        return

    with open(file) as f:
        schema = json.loads(f.read())
    resp = gen_example(schema)
    file_base = file.split("/")[-1]
    output = json.dumps(resp, indent=4)
    with open(output_name, "w") as f:
        f.write(output)
    print(f"Wrote {output_name}  {len(output)} bytes")


files = []
for arg in sys.argv[1:]:
    if arg.endswith(".json"):
        files.append(arg)
    else:
        files.extend(glob.glob(arg + "/*.json"))
print(len(files))
random.shuffle(files)

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    executor.map(process_file, files)
