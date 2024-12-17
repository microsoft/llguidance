#!/usr/bin/env python3

from typing import Any, List
import sys
import json
import glob
import os
import random
import re
from jsonschema import Draft202012Validator, validate
import tokenizers
import llguidance
import copy


class PhiTokenizer:
    _ll_tokenizer = None
    _instance = None

    @staticmethod
    def instance():
        if PhiTokenizer._instance is None:
            PhiTokenizer._instance = PhiTokenizer()
        return PhiTokenizer._instance

    @staticmethod
    def ll_tokenizer():
        if PhiTokenizer._ll_tokenizer is None:
            PhiTokenizer._ll_tokenizer = llguidance.LLTokenizer(
                llguidance.TokenizerWrapper(PhiTokenizer.instance())
            )
        return PhiTokenizer._ll_tokenizer

    def tokenize_str(self, s: str) -> List[int]:
        return self.hf_tokenizer.encode(s).ids

    def __init__(self) -> None:
        self.hf_tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct"
        )
        empty = self.tokenize_str("")
        if empty:
            self.bos_token_id = empty[0]
        else:
            self.bos_token_id = None
        eos = self.tokenize_str("</s>")
        assert len(eos) == 1
        self.eos_token_id = eos[0]
        self.tokens = []
        for i in range(self.hf_tokenizer.get_vocab_size()):
            t: str = self.hf_tokenizer.id_to_token(i)
            if t.startswith("<0x"):
                self.tokens.append(bytes([int(t[3:5], 16)]))
            else:
                t = t.replace("▁", " ")
                self.tokens.append(t.encode("utf-8"))
        assert len(self.tokens) == self.hf_tokenizer.get_vocab_size()

    def __call__(self, s):
        return self.tokenize_str(s)


class Stats:
    def __init__(self):
        self.files = 0
        self.grammar_error = 0
        self.valid_ok = 0
        self.valid_error = 0
        self.invalid_ok = 0
        self.invalid_error = 0
        self.num_exn = 0
        self.num_reorder_errors = 0
        self.num_half_wrong = 0


stats = Stats()
magic_key = "$$$magic_key_for_reordering$$$"


class TokensInvalid(Exception):
    pass


class ReorderError(Exception):
    pass


def remove_magic_key(obj):
    if isinstance(obj, dict):
        if magic_key in obj:
            del obj[magic_key]
        for k, v in obj.items():
            remove_magic_key(v)
    elif isinstance(obj, list):
        for v in obj:
            remove_magic_key(v)


# this assumes we don't force the canonical tokenization on commas, quotes etc.
# we could always just use byte tokenizer
def reorder_json(test: dict, interp: llguidance.LLInterpreter):
    interp = interp.deep_copy()
    text_buf = ""
    text_committed = ""
    dbg = False
    ll_tok = PhiTokenizer.ll_tokenizer()

    def save_state():
        return text_buf, text_committed, interp.deep_copy()

    def restore_state(state):
        nonlocal text_buf, text_committed, interp
        text_buf, text_committed, interp = state

    def stringify(v):
        return json.dumps(v, ensure_ascii=False, indent=None, separators=(",", ":"))

    def flush_tokens():
        nonlocal text_buf, text_committed
        if text_buf == "":
            return
        tokens = PhiTokenizer.ll_tokenizer().tokenize_str(text_buf)
        for idx, token in enumerate(tokens):
            n = interp.validate_tokens_raw([token])
            if n != 1:
                if dbg:
                    print(
                        "FAIL", text_committed, ll_tok.dbg_tokens(tokens[0 : idx + 1])
                    )
                raise TokensInvalid()
            interp.commit_token(token)
        text_committed += text_buf
        if dbg:
            print("OK", text_committed)
        text_buf = ""

    def append_text(text: str):
        nonlocal text_buf
        text_buf += text

    def validate_text(text: str):
        flush_tokens()
        tokens = PhiTokenizer.ll_tokenizer().tokenize_str(text)
        if interp.validate_tokens_raw(tokens) != len(tokens):
            if dbg:
                print("FAIL_V", text_committed, text)
            return False
        if dbg:
            print("OK_V", text_committed, text)
        return True

    def check(obj):
        if isinstance(obj, dict):
            if magic_key in obj:
                append_text(obj[magic_key])
            else:
                flush_tokens()
                committed_len = len(text_committed)
                append_text("{")
                key_order = []
                keys_left = list(obj.keys())
                if not keys_left:
                    append_text("}")
                    flush_tokens()
                while keys_left:
                    is_ok = False
                    possible_keys = [
                        k for k in keys_left if validate_text(stringify(k) + ":")
                    ]
                    for key in possible_keys:
                        prev_state = save_state()
                        try:
                            append_text(stringify(key) + ":")
                            check(obj[key])
                            # make sure we can still advance to all future keys
                            for key2 in possible_keys:
                                if key2 != key:
                                    if not validate_text("," + stringify(key2) + ":"):
                                        raise TokensInvalid()
                            if len(keys_left) == 1:
                                append_text("}")
                            else:
                                append_text(",")
                            flush_tokens()

                            # all good, we can continue
                            keys_left.remove(key)
                            key_order.append((key, obj[key]))
                            is_ok = True
                            break
                        except TokensInvalid:
                            restore_state(prev_state)
                            continue
                    if not is_ok:
                        raise ReorderError(
                            "no key can be advanced\n"
                            + stringify(obj)
                            + "\n\ntext_committed: "
                            + text_committed
                            + "\n\nkeys_left: "
                            + str(keys_left)
                            + "\npossible_keys: "
                            + str(possible_keys)
                            + "\nkey_order: "
                            + str(key_order)
                        )
                obj.clear()
                for k, v in key_order:
                    obj[k] = v
                if text_buf != "":
                    raise ValueError("text_buf not empty", text_buf)
                obj_str = text_committed[committed_len:]
                obj[magic_key] = obj_str
        elif isinstance(obj, list):
            append_text("[")
            for idx, v in enumerate(obj):
                if idx > 0:
                    append_text(",")
                check(v)
            append_text("]")
        else:
            append_text(stringify(obj))

    interp.start_without_prompt()

    root_obj = copy.deepcopy(test["data"])
    try:
        check(root_obj)
        flush_tokens()
        if not interp.is_accepting():
            raise ValueError("reordering failed", stringify(root_obj))
        remove_magic_key(root_obj)
        test["data"] = root_obj
        # print("reordering succeeded", stringify(root_obj))
    except ReorderError as e:
        if not test["valid"]:
            # kind of expected, save what we can
            remove_magic_key(root_obj)
            test["data"] = root_obj
            stats.num_half_wrong += 1
            return
        raise e
    except TokensInvalid:
        if not test["valid"]:
            # kind of expected, save what we can
            remove_magic_key(root_obj)
            test["data"] = root_obj
            stats.num_half_wrong += 1
            return
        raise ValueError("reordering failed", stringify(root_obj))


def remove_constraints(obj):
    if isinstance(obj, dict):
        if "properties" in obj:
            for k, v in obj["properties"].items():
                remove_constraints(v)
            return
        for k in [
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "pattern",
            "minLength",
            "maxLength",
            "format",
        ]:
            if k in obj:
                del obj[k]
        for k, v in obj.items():
            remove_constraints(v)
    elif isinstance(obj, list):
        for v in obj:
            remove_constraints(v)


def mk_interp(schema):
    return llguidance.LLInterpreter(
        PhiTokenizer.ll_tokenizer(),
        json.dumps(
            {
                "grammars": [
                    {
                        "json_schema": schema,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        log_level=0,
        enable_ff_tokens=False,
        enable_backtrack=False,
    )


def process_file(file_name):
    file_base = file_name.split("/")[-1]

    stats.files += 1

    with open(file_name) as f:
        pos_data = json.loads(f.read())

    schema = pos_data["schema"]
    tests = pos_data["tests"]

    schema_simplified = copy.deepcopy(schema)
    remove_constraints(schema_simplified)

    try:
        interp0 = mk_interp(schema)
    except Exception as e:
        # print("interpreter creation error", file_name, str(e))
        stats.grammar_error += 1
        return

    interp1 = mk_interp(schema_simplified)

    num_reordered = 0

    for idx, test in enumerate(tests):
        test_str = json.dumps(test["data"], ensure_ascii=False, indent=None)
        ll_tok = PhiTokenizer.ll_tokenizer()
        tokens = ll_tok.tokenize_str(test_str)
        # print(ll_tok.dbg_tokens(tokens))

        interp = interp0.deep_copy()

        interp.start_without_prompt()

        try:
            tests_valid = True
            for tidx, token in enumerate(tokens):
                n = interp.validate_tokens_raw([token])
                if n == 0:
                    tests_valid = False
                    if test["valid"]:
                        # print(
                        #     "test fails",
                        #     file_name,
                        #     idx,
                        #     ll_tok.dbg_tokens(tokens[0 : tidx + 1]),
                        # )
                        reorder_json(test, interp0)
                        stats.valid_error += 1
                        num_reordered += 1
                    else:
                        # first try with a simplified schema, that
                        # doesn't have string/number constraints
                        reorder_json(test, interp1)
                        reorder_json(test, interp0)
                        num_reordered += 1

                    break
                interp.commit_token(token)

            if tests_valid and not interp.is_accepting():
                tests_valid = False
                if test["valid"]:
                    print("test fails at the end", file_name, idx)
                    stats.valid_error += 1

            if tests_valid:
                if test["valid"]:
                    stats.valid_ok += 1
                else:
                    print("invalid test shouldn't pass", file_name, idx)
                    stats.invalid_error += 1
            else:
                if not test["valid"]:
                    stats.invalid_ok += 1
        except ReorderError as e:
            print("reorder error", file_name, idx, str(e))
            stats.num_reorder_errors += 1
        except Exception as e:
            stats.num_exn += 1
            # raise e
            if "consider making your grammar left-recursive" not in str(e):
                print("validation error", file_name, idx, repr(e))

    if num_reordered > 0:
        with open(file_name, "w") as f:
            f.write(json.dumps(pos_data, ensure_ascii=False, indent=2))


files = []
for arg in sys.argv[1:]:
    if arg.endswith(".json"):
        files.append(arg)
    else:
        files.extend(glob.glob(arg + "/*.json"))
print(len(files))
files.sort()

for idx, f in enumerate(files):
    if idx > 0 and idx % 500 == 0:
        print(idx, stats.__dict__, file=sys.stderr)
    process_file(f)

print(stats.__dict__)
