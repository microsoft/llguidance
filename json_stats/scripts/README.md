
Typical script usage:

- `json_to_unique_tests.py $JSB_DATA/json/*` - creates/updates `$JSB_DATA/unique_tests/*.json` 
- `gen_tests_with_llm.py $JSB_DATA/unique_tests` - creates `$JSB_DATA/work/responses/*.json`
- `add_tests.py $JSB_DATA/work/responses` - updates `$JSB_DATA/unique_tests/*.json` with generated tests
- `reorder.py $JSB_DATA/unique_tests` - reorders keys in JSON to llguidance

Other scripts:

- `unique_tests_to_json.py $JSB_DATA/unique_tests` - updates `$JSB_DATA/json/*.json` based on tests
- `add_reasons.py $JSB_DATA/unique_tests` - update python validation errors
