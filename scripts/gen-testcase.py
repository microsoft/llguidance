#!/usr/bin/env python

import json
import sys

"""
To use this, run Guidance with LLGUIDANCE_TEST_TRACE=1 LLGUIDANCE_LOG_LEVEL=2 and
then pass the output here.
"""

def testcase_from_logs(logs: str):
    sep = "‧"
    pairs = []
    prev_res = None
    prompt = None
    for line in logs.split("\n"):
        if line.startswith("TEST: "):
            obj = json.loads(line[6:])
            if prompt is None:
                prompt = obj["res_prompt"]
                continue
            if "res_prompt" in obj:
                print("There's a second testcase here")
                break
            if prev_res:
                pairs.append((prev_res, obj["arg"]))
            prev_res = obj["res"]
    # assert prev_res == "stop"
    testcase = [prompt]
    gen_tokens = []

    def flush_gen_tokens():
        testcase.append(sep.join(gen_tokens))
        gen_tokens.clear()

    for res, arg in pairs:
        print(res, arg)
        if res["sample_mask"]:
            gen_tokens.append(arg["tokens"])
        else:
            splice = res["splices"][0]
            t0 = splice["tokens"]
            assert t0 == arg["tokens"]
            flush_gen_tokens()
            if splice["backtrack"]:
                t0 = str(splice["backtrack"]) + "↶" + t0
            testcase.append(t0)
    if gen_tokens:
        flush_gen_tokens()

    print("Testcase:", testcase)


def main():
    with open(sys.argv[1], "r") as f:
        logs = f.read()
    testcase_from_logs(logs)

main()
