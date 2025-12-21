import argparse

from rag_app.rag_chain import make_rag_chain

def make_embeddings(model_name: str):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)

def make_llm():
    """
    MVP：你可以先用任何 LangChain 支援嘅 chat model。
    例子：
      - ChatOpenAI（如果你有 API）
      - ChatOllama（本機 Ollama）
      - 你自家 vLLM / OpenAI-compatible endpoint
    我呢度先用一個最常見嘅 pattern：如果你有 OpenAI-compatible，就用 ChatOpenAI。
    """
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except Exception as e:
        raise RuntimeError(
            "你未裝/未設定 LLM client。建議：\n"
            "1) 用 OpenAI-compatible：pip install langchain-openai\n"
            "2) 或用 Ollama：pip install langchain-ollama\n"
            "然後改 make_llm()。\n"
        ) from e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--emb_model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    embeddings = make_embeddings(args.emb_model)

    from langchain_community.vectorstores import FAISS
    vs = FAISS.load_local(args.index_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": args.k})

    llm = make_llm()
    chain = make_rag_chain(retriever, llm)

    print("=== Property-LLM-HK MVP (FAISS) ===")
    print("輸入問題，打 exit 離開。\n")

    while True:
        q = input("你：").strip()
        if not q or q.lower() in ("exit", "quit"):
            break
        ans = chain.invoke({"question": q})
        print("\n助理：")
        print(ans)
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
