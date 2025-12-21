import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple


def extract_chunk_index(doc_id: str) -> int | None:
    """
    Try extract chunk index from IDs like:
      icac_bm_1_chunk12 -> 12
    """
    if not doc_id:
        return None
    m = re.search(r"(?:chunk|ck|seg)[\-_ ]*(\d+)$", doc_id, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"_chunk(\d+)", doc_id, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def normalize_spacing(s: str) -> str:
    if not isinstance(s, str):
        return s
    # normalize spaces around "第" ... "節"
    s = re.sub(r"第\s*(\d+(?:\.\d+)*)\s*節", r"第 \1 節", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # keep newlines (useful for steps), but remove weird trailing spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_doc_type(source: str) -> str:
    # simple heuristic; you can expand later
    s = (source or "").lower()
    if "icac" in s or "guideline" in s or "document" in s:
        return "guideline"
    return "unknown"


def infer_publisher(source: str) -> str:
    s = (source or "").lower()
    if "icac" in s:
        return "ICAC"
    if "lands tribunal" in s:
        return "Lands Tribunal"
    if "bmo" in s or "cap.344" in s:
        return "HKSAR"
    return "unknown"


def normalize_record(raw: Dict[str, Any], file_name: str) -> Tuple[Dict[str, Any], list[str]]:
    warnings = []

    # handle typos / legacy keys
    doc_id = raw.get("ID") or raw.get("id")
    source = raw.get("SOURCE") or raw.get("source")
    chapter = raw.get("CHAPTER") or raw.get("chapter")
    section = raw.get("SECTION") or raw.get("section")
    title = raw.get("TITLE") or raw.get("title")
    content = raw.get("CONTENT") or raw.get("text") or raw.get("content") or ""
    category = raw.get("CATEGORY") or raw.get("CATERGORY") or raw.get("category")  # typo fix
    section_type = raw.get("SECTION_TYPE") or raw.get("SCTION_TYPE") or raw.get("section_type")  # typo fix

    if not doc_id:
        warnings.append("missing id")
    if not content:
        warnings.append("missing content/text")

    # normalize strings
    chapter_n = normalize_spacing(chapter) if isinstance(chapter, str) else chapter
    section_n = normalize_spacing(section) if isinstance(section, str) else section
    title_n = title.strip() if isinstance(title, str) else title
    text_n = clean_text(content)

    doc_type = raw.get("doc_type") or infer_doc_type(source)
    publisher = raw.get("publisher") or infer_publisher(source)
    lang = raw.get("lang") or "zh-HK"

    chunk_index = raw.get("chunk_index")
    if chunk_index is None and isinstance(doc_id, str):
        chunk_index = extract_chunk_index(doc_id)

    normalized = {
        "id": doc_id,
        "source": source,
        "doc_type": doc_type,
        "publisher": publisher,
        "category": category,
        "section_type": section_type,
        "chapter": chapter_n,
        "section": section_n,
        "title": title_n,
        "text": text_n,
        "lang": lang,
        "chunk_index": chunk_index,
        "file_name": file_name,
    }

    return normalized, warnings


def process_file(in_path: Path, out_path: Path) -> Dict[str, Any]:
    total = 0
    ok = 0
    bad_json = 0
    warn_count = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            normalized, warnings = normalize_record(raw, in_path.name)
            if warnings:
                warn_count += 1

            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            ok += 1

    return {
        "file": in_path.name,
        "total_lines": total,
        "written": ok,
        "bad_json": bad_json,
        "warn_lines": warn_count,
        "out": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="folder containing *.jsonl")
    ap.add_argument("--out_dir", required=True, help="output folder for normalized *.jsonl")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        raise SystemExit(f"in_dir not found: {in_dir}")

    files = sorted(in_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files in: {in_dir}")

    reports = []
    for fp in files:
        out_fp = out_dir / fp.name
        rep = process_file(fp, out_fp)
        reports.append(rep)

    # simple summary
    total_in = sum(r["total_lines"] for r in reports)
    total_out = sum(r["written"] for r in reports)
    total_bad = sum(r["bad_json"] for r in reports)
    total_warn = sum(r["warn_lines"] for r in reports)

    print("=== Normalize JSONL Summary ===")
    print(f"files: {len(reports)}")
    print(f"input lines: {total_in}")
    print(f"output lines: {total_out}")
    print(f"bad json lines: {total_bad}")
    print(f"warn lines: {total_warn}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
