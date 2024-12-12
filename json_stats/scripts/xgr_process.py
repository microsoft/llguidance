#!/usr/bin/env python3

import json
import sys

objs = []

with open(sys.argv[1]) as f:
    for l in f.readlines():
        if not l:
            continue
        j = json.loads(l)
        buf = ""
        for outline in j["output"].split("\n"):
            if outline.startswith("RESULT: "):
                parsed = json.loads(outline[len("RESULT: "):])
                parsed["warnings"] = buf
                objs.append(parsed)
                print(parsed.get("compile_time",0), parsed["file"])
                buf = ""
            else:
                buf += outline + "\n"

with open(sys.argv[2], "w") as f:
    f.write(json.dumps(objs, indent=4))