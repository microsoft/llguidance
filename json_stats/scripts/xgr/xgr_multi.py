#!/usr/bin/env python3

import sys
import json
import glob
import os
import random
import time
import concurrent.futures
import subprocess
from typing import List

positive_base = os.environ.get("HOME") + "/src/json-data/positive"
output_base = os.environ.get("HOME") + "/src/json-data/xgr_output"

def process_file(files: List[str]):
    timeout=60
    try:
        command = ["python", "xgr_test.py"] + files
        
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            timeout=timeout
        )
        
        # Collect combined output
        output = result.stdout
        
        return {
            "output": output,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired as e:
        output = e.output or ""
        # decode output if needed
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return {
            "output": output,
            "error": "TimeoutExpired",
            "message": f"The process exceeded the {timeout}-second timeout."
        }
    except Exception as e:
        return {
            "error": "ExecutionError",
            "message": str(e)
        }

files = []
for arg in sys.argv[1:]:
    if arg.endswith(".json"):
        files.append(arg)
    else:
        files.extend(glob.glob(arg + "/*.json"))
print(len(files), file=sys.stderr)
missing_files = []
for f in files:
    file_base = f.split("/")[-1]
    output_name = f"{output_base}/{file_base}"
    if not os.path.exists(output_name):
        missing_files.append(f)
print(len(missing_files), file=sys.stderr)
random.shuffle(missing_files)

if len(missing_files) < 10:
    print(missing_files)
    sys.exit(0)

chunk_size = 10
chunks = []
for i in range(0, len(missing_files), chunk_size):
    chunks.append(missing_files[i:i + chunk_size])

log_file = f"{output_base}/log.txt"
cnt = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
    futures = {executor.submit(process_file, f): f for f in chunks}
    for future in concurrent.futures.as_completed(futures):
        files = futures[future]
        try:
            r = future.result()
            cnt += len(files)
            print(cnt)
            rs = json.dumps(r)
            with open(log_file, "a") as f:
                f.write(f"FILES: {files}\n{rs}\n")
            # print(f"OK: {files}")
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"ERROR {files}: {repr(e)}")
            print(f"ERROR: {files}", repr(e))
