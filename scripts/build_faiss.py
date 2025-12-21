import argparse
from pathlib import Path

from rag_app.loader import load_cleaned_jsonl_folder

# Embeddings：優先用 langchain_huggingface，如果無就用 langchain_community 版本
def make_embeddings(model_name: str):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_dir", required=True, help="folder containing cleaned *.jsonl")
    ap.add_argument("--out_dir", required=True, help="folder to save FAISS index")
    ap.add_argument("--emb_model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    args = ap.parse_args()

    docs = load_cleaned_jsonl_folder(args.jsonl_dir)
    if not docs:
        raise SystemExit("No documents loaded. Check your jsonl_dir and records' text.")

    embeddings = make_embeddings(args.emb_model)

    from langchain_community.vectorstores import FAISS
    vs = FAISS.from_documents(docs, embeddings)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out))

    print("=== FAISS Build Done ===")
    print(f"docs: {len(docs)}")
    print(f"index saved to: {out}")


if __name__ == "__main__":
    main()
