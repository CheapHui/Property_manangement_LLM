import json
import re
from pathlib import Path
from typing import Iterable, List, Dict, Any

from langchain_core.documents import Document


def _autofix_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    """
    MVP runtime 修補：唔改你原檔，只係 load 入嚟時令 filter/trace 更好用。
    """
    md = dict(md)

    # normalize doc_type/publisher
    if md.get("doc_type") in (None, "", "unknown"):
        if md.get("category") == "court_case":
            md["doc_type"] = "case"
        elif md.get("publisher") == "ICAC":
            md["doc_type"] = "guideline"

    if md.get("publisher") in (None, "", "unknown"):
        src = (md.get("source") or "").lower()
        if "hkca" in src or "court of appeal" in src:
            md["publisher"] = "HKCA"

    # ensure chunk_index is int if possible
    ci = md.get("chunk_index")
    if isinstance(ci, str) and ci.isdigit():
        md["chunk_index"] = int(ci)

    return md

def _infer_case_no(file_name: str, doc_id: str | None) -> str | None:
    s = f"{file_name} {doc_id or ''}"
    # e.g. "CACV 30299.jsonl" -> "CACV 302/1999" (approx) OR at least "CACV 30299"
    m = re.search(r"\b(CACV|CACC|CACI|HCMP|HCA|DCCJ)\s*([0-9]{1,5})(?:\/([0-9]{2,4}))?\b", s, re.IGNORECASE)
    if not m:
        return None
    prefix = m.group(1).upper()
    num = m.group(2)
    year = m.group(3)
    if year:
        return f"{prefix} {num}/{year}"
    return f"{prefix} {num}"

def _infer_pinpoint(md: Dict[str, Any]) -> str | None:
    # Prefer explicit fields if you add later
    for k in ("section_no", "para_no", "chapter_no"):
        v = md.get(k)
        if v:
            if k == "section_no":
                return f"s.{v}"
            if k == "para_no":
                return f"para.{v}"
            if k == "chapter_no":
                return f"ch.{v}"
    # fallback to 'section' / 'chapter' strings if present
    sec = md.get("section")
    if isinstance(sec, str) and sec.strip():
        return sec.strip()[:60]
    ch = md.get("chapter")
    if isinstance(ch, str) and ch.strip():
        return ch.strip()[:60]
    return None

def _autofix_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    md = dict(md)

    # normalize doc_type
    dt = md.get("doc_type")
    cat = md.get("category")
    if dt in (None, "", "unknown"):
        if cat == "court_case":
            md["doc_type"] = "case"
        elif cat in ("ordinance", "bmo", "law"):
            md["doc_type"] = "ordinance"
        elif (md.get("publisher") == "ICAC") or (isinstance(md.get("source"), str) and "icac" in md["source"].lower()):
            md["doc_type"] = "guideline"
        else:
            md["doc_type"] = "unknown"

    # normalize publisher
    if md.get("publisher") in (None, "", "unknown"):
        src = (md.get("source") or "").lower()
        if "hkca" in src or "court of appeal" in src:
            md["publisher"] = "HKCA"
        elif "icac" in src:
            md["publisher"] = "ICAC"
        else:
            md["publisher"] = md.get("publisher") or "unknown"

    # chunk_index int
    ci = md.get("chunk_index")
    if isinstance(ci, str) and ci.isdigit():
        md["chunk_index"] = int(ci)

    # ---- NEW: citation-friendly fields (runtime only) ----
    if not md.get("display_title"):
        # prefer title, else section, else chapter, else id
        md["display_title"] = md.get("title") or md.get("section") or md.get("chapter") or md.get("id") or "?"

    if md["doc_type"] == "case" and not md.get("case_no"):
        md["case_no"] = _infer_case_no(md.get("file_name", ""), md.get("id"))

    if not md.get("pinpoint"):
        md["pinpoint"] = _infer_pinpoint(md)

    return md


def load_cleaned_jsonl_folder(folder: str) -> List[Document]:
    """
    Load all *.jsonl under folder. Each line is a record with:
      - text/content/body: str
      - other keys: metadata
    """
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    docs: List[Document] = []
    for fp in sorted(p.glob("*.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)

                # Accept multiple possible text fields (some source files are uppercase)
                text = (
                    r.get("text")
                    or r.get("TEXT")
                    or r.get("content")
                    or r.get("CONTENT")
                    or r.get("body")
                )
                text = (text or "").strip()
                if not text:
                    continue

                # Preserve metadata (remove whichever text key was used)
                # Normalize keys to lowercase for consistent access
                md = {k.lower(): v for k, v in r.items() if k not in {"text", "TEXT", "content", "CONTENT", "body"}}

                # Fix common typos in keys (e.g., CATERGORY -> category, SCTION_TYPE -> section_type)
                if "catergory" in md:
                    md["category"] = md.pop("catergory")
                if "sction_type" in md:
                    md["section_type"] = md.pop("sction_type")

                md["__line_no"] = line_no
                md["__path"] = str(fp)

                md = _autofix_metadata(md)

                docs.append(Document(page_content=text, metadata=md))
    return docs
