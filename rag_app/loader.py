import json
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


def load_cleaned_jsonl_folder(folder: str) -> List[Document]:
    """
    Load all *.jsonl under folder. Each line is a record with:
      - text: str
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
                text = (r.get("text") or "").strip()
                if not text:
                    continue

                md = {k: v for k, v in r.items() if k != "text"}
                md["__line_no"] = line_no
                md["__path"] = str(fp)

                md = _autofix_metadata(md)

                docs.append(Document(page_content=text, metadata=md))
    return docs
