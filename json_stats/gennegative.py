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


def feature_name(feature):
    if feature == "type_array":
        return "type keyword with an array of types"
    if feature == "additionalProperties:object":
        return "additionalProperties keyword with an object schema"
    return feature + " keyword"


singleton_features = [
    "type_array",
    "pattern",
    "oneOf",
    "format",
    "allOf",
    "anyOf",
    "patternProperties",
    "dependencies",
    "additionalItems",
    "additionalProperties:object",
    "multipleOf",
    "uniqueItems",
    "propertyNames",
    "contains",
]

group_features = [
    ["minProperties", "maxProperties"],
    ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"],
    ["minLength", "maxLength"],
    ["minItems", "maxItems"],
    ["if", "then", "else"],
]


# read tmp/all_stats.json
with open("tmp/all_stats.json") as f:
    all_stats = json.loads(f.read())

positive_base = os.environ.get("HOME") + "/src/json-data/positive"
output_base = os.environ.get("HOME") + "/src/json-data/positive_output"


def gen_example(schema, instance, info):
    prompt = (
        ""
        + f"You will be given a JSON schema an example instance of it. "
        + f"You will then modify the instance a little bit to violate the schema.\n"
        + info
        + "\n\nHere is the schema:\n"
        + json.dumps(schema, indent=4)
        + "\n\nHere is the valid instance:\n"
        + json.dumps(instance, indent=4)
        + "\n\nPlease modify the instance to make it invalid according to the schema."
        + info
    )
    if len(prompt) > 100_000:
        return {"error": f"Prompt too long, {len(prompt)}"}
    req = {
        "model": "model",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert in generating testcases for JSON schema validators.",
            },
            {"role": "user", "content": prompt},
        ],
        "return_expanded_prompt": true,
        "include_json_schema_in_prompt": false,
        "llg_log_level": "verbose",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                # empty schema to allow any JSON response
                "schema": {},
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


def process_file(file: str):
    file_base = file.split("/")[-1]
    output_name = f"{output_base}/{file_base}"
    if os.path.exists(output_name):
        return

    with open(file) as f:
        pos_data = json.loads(f.read())
    stats = all_stats[pos_data["_schema_file"]]
    feature_names = set()
    for k in stats["features"].keys():
        if k in singleton_features:
            feature_names.add(feature_name(k))
        else:
            for lst in group_features:
                if k in lst:
                    lst2 = [e for e in lst if stats["features"].get(e, 0) > 0]
                    name = " or ".join([feature_name(e) for e in lst2])
                    feature_names.add(name)
                    break

    feature_list = [f"Try to modify the instance to detect bugs in a validator that doesn't support {f}. In other words, try to violate a constraint introduced by {f} in the schema.\n" for f in feature_names]
    feature_list.append("")
    responses = []
    for f in feature_list:
        r = gen_example(pos_data["schema"], pos_data["tests"][0]["data"], f)
        responses.append(r)

    res = {
        "pos_data": pos_data,
        "responses": responses,
    }
    output = json.dumps(res, indent=4)
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
    futures = {executor.submit(process_file, f): f for f in files}
    for future in concurrent.futures.as_completed(futures):
        file_name = futures[future]
        try:
            future.result()  # This will raise exceptions if any occurred
        except Exception as e:
            print(f"Error processing {file_name}: {repr(e)}")
