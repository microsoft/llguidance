#!/usr/bin/env python3

import requests
import sys
import json
import glob
import os
import random
import concurrent.futures


output_base = os.environ.get("JSB_DATA") + "/work/responses"

url = "http://localhost:3001/v1/chat/completions"

gen_negative = False
gen_constrained = False


def feature_name(feature):
    if feature == "type:[]":
        return "type keyword with an array of types"
    if feature == "additionalProperties:object":
        return "additionalProperties keyword with an object schema"
    return feature + " keyword"


singleton_features = [
    "type:[]",
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


def gen_example(schema: dict, instance: dict | None, info: str, constrained: bool):
    if instance is None:
        valid = True
        prompt = (
            "Please generate an example data that validates against the following schema:\n"
            + json.dumps(schema, indent=4, ensure_ascii=False)
        )
    else:
        valid = False
        constrained = False
        prompt = (
            ""
            + f"You will be given a JSON schema an example instance of it. "
            + f"You will then modify the instance a little bit to violate the schema.\n"
            + "\n\nHere is the schema:\n"
            + json.dumps(schema, indent=4, ensure_ascii=False)
            + "\n\nHere is the valid instance:\n"
            + json.dumps(instance, indent=4, ensure_ascii=False)
            + "\n\nPlease modify the instance to make it invalid according to the schema. Focus on corner cases.\n"
            + info
        )
    if len(prompt) > 200_000:
        return {"error": f"Prompt too long, {len(prompt)}"}

    # empty schema to allow any JSON response
    format = {"schema": {}}
    if constrained:
        format = {
            "strict": True,
            "schema": schema,
        }

    req = {
        "model": "model",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert in generating testcases for JSON schema validators.",
            },
            {"role": "user", "content": prompt},
        ],
        "return_expanded_prompt": True,
        "include_json_schema_in_prompt": False,
        # "llg_log_level": "verbose",
        "response_format": {
            "type": "json_schema",
            "json_schema": format,
        },
        "max_tokens": 8000,
        "temperature": 0.7,
    }

    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=req)
    # response.raise_for_status()
    r = response.json()
    r["meta"] = {
        "valid": valid,
        "info": info,
        "prompt": prompt,
    }
    return r


def process_file(file: str):
    file_base = file.split("/")[-1]
    output_name = f"{output_base}/{file_base}"
    if os.path.exists(output_name):
        return

    with open(file) as f:
        test_file = json.loads(f.read())
    last_valid_test = None

    if gen_negative:
        for t in test_file["tests"]:
            if t["valid"]:
                last_valid_test = t
        if last_valid_test is None:
            return

    stats = test_file["meta"]
    feature_names = set()
    for k in stats["features"]:
        if k in singleton_features:
            feature_names.add(feature_name(k))
        else:
            for lst in group_features:
                if k in lst:
                    lst2 = [e for e in lst if e in stats["features"]]
                    name = " or ".join([feature_name(e) for e in lst2])
                    feature_names.add(name)
                    break

    feature_list = [
        f"Try to modify the instance to detect bugs in a validator that doesn't support {f}. In other words, try to violate a constraint introduced by {f} in the schema.\n"
        for f in feature_names
    ]
    feature_list.append("")
    responses = []

    if gen_negative:
        for f in feature_list:
            r = gen_example(
                schema=test_file["schema"],
                instance=last_valid_test["data"],
                info=f,
                constrained=False,
            )
            responses.append(r)
    else:
        # right now, not using features when generating positive examples
        r = gen_example(
            schema=test_file["schema"],
            instance=None,
            info="",
            constrained=gen_constrained,
        )
        responses.append(r)

    res = {
        "pos_data": test_file,
        "responses": responses,
    }
    output = json.dumps(res, indent=4, ensure_ascii=False)
    with open(output_name, "w") as f:
        f.write(output)
    print(f"Wrote {output_name}  {len(output)} bytes")


args = sys.argv[1:]
while len(args) > 0 and args[0].startswith("-"):
    arg = args.pop(0)
    if arg == "-n":
        gen_negative = True
    elif arg == "-c":
        gen_constrained = True
    else:
        raise ValueError(f"Unknown option {arg}")

files = []
for arg in args:
    if arg.endswith(".json"):
        files.append(arg)
    else:
        files.extend(glob.glob(arg + "/*.json"))
print(
    len(files),
    "files to process",
    "negative" if gen_negative else "positive",
    "constrained" if gen_constrained else "unconstrained",
)
random.shuffle(files)

try:
    os.makedirs(output_base)
except FileExistsError:
    pass

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    futures = {executor.submit(process_file, f): f for f in files}
    for future in concurrent.futures.as_completed(futures):
        file_name = futures[future]
        try:
            future.result()  # This will raise exceptions if any occurred
        except Exception as e:
            print(f"Error processing {file_name}: {repr(e)}")
