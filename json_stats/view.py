#!/usr/bin/env python3

import sys
import json

def process_file(output_name):
    with open(output_name) as f:
        resp = json.loads(f.read())
    rs = resp["choices"][0]["message"]["content"]
    logs = resp["choices"][0].get("llg_logs", None)

    print(logs)
    print(rs)


for arg in sys.argv[1:]:
    process_file(arg)
