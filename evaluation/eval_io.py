# eval/io.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class LoadOptions:
    strict_json: bool = True          # invalid json -> raise
    skip_blank: bool = True
    require_unique_id: bool = True
    normalize_ids: bool = False       # optional: normalize to PMHK-EVAL-<task>-<nnn>
    id_prefix: str = "PMHK-EVAL"
    keep_original_id: bool = True     # if normalize, store old id in _old_id
    filter_tasks: Optional[Set[str]] = None
    filter_task_groups: Optional[Set[str]] = None


def read_jsonl(path: str, *, strict: bool = True, skip_blank: bool = True) -> List[JsonDict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")

    items: List[JsonDict] = []
    for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if skip_blank and not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            if strict:
                raise ValueError(f"Invalid JSON at line {ln}: {e}") from e
            else:
                continue
        if not isinstance(obj, dict):
            raise ValueError(f"Line {ln} is not a JSON object")
        items.append(obj)
    return items


def write_jsonl(path: str, items: Iterable[JsonDict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _apply_filters(items: List[JsonDict], opt: LoadOptions) -> List[JsonDict]:
    if opt.filter_tasks:
        items = [it for it in items if it.get("task") in opt.filter_tasks]
    if opt.filter_task_groups:
        items = [it for it in items if it.get("task_group") in opt.filter_task_groups]
    return items


def _check_unique_ids(items: List[JsonDict]) -> None:
    seen = set()
    for it in items:
        _id = it.get("id")
        if not isinstance(_id, str) or not _id:
            raise ValueError("Missing/invalid 'id' (must be non-empty string)")
        if _id in seen:
            raise ValueError(f"Duplicate id: {_id}")
        seen.add(_id)


def normalize_ids_task_based(
    items: List[JsonDict],
    *,
    prefix: str = "PMHK-EVAL",
    keep_original_id: bool = True,
) -> List[JsonDict]:
    """
    Normalize IDs to: PMHK-EVAL-<task>-<nnn>, per task sequence in current file order.
    Also sync eval_variants[].variant_id to <id>-V1 / <id>-V2 / ...
    """
    from collections import defaultdict

    by_task: Dict[str, List[JsonDict]] = defaultdict(list)
    for it in items:
        task = it.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("Missing/invalid 'task' for id normalization")
        by_task[task].append(it)

    # assign new ids
    used: Set[str] = set()
    for task, lst in by_task.items():
        for i, it in enumerate(lst, start=1):
            new_id = f"{prefix}-{task}-{i:03d}"
            if new_id in used:
                raise ValueError(f"Duplicate after normalize: {new_id}")
            used.add(new_id)

            if keep_original_id and it.get("id") != new_id:
                it["_old_id"] = it.get("id")

            it["id"] = new_id

            # sync variants
            vs = it.get("eval_variants", [])
            if isinstance(vs, list):
                for j, v in enumerate(vs, start=1):
                    if isinstance(v, dict):
                        v["variant_id"] = f"{new_id}-V{j}"

    return items


def load_eval_set(path: str, opt: Optional[LoadOptions] = None) -> List[JsonDict]:
    opt = opt or LoadOptions()

    items = read_jsonl(path, strict=opt.strict_json, skip_blank=opt.skip_blank)
    items = _apply_filters(items, opt)

    if opt.normalize_ids:
        items = normalize_ids_task_based(items, prefix=opt.id_prefix, keep_original_id=opt.keep_original_id)

    if opt.require_unique_id:
        _check_unique_ids(items)

    return items