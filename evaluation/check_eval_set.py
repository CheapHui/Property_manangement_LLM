from eval_io import load_eval_set, LoadOptions
from validate import validate_eval_set, format_errors

items = load_eval_set(
    "evaluation/eval_set_v1.jsonl",
    LoadOptions(
        normalize_ids=False,     # 你想統一 id 時先開 True
        require_unique_id=True,
    ),
)

errs = validate_eval_set(items)
if errs:
    print("❌ Validation failed")
    print(format_errors(errs))
    raise SystemExit(1)

print(f"✅ OK. Loaded {len(items)} items.")

