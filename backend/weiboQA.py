from buildFAISSIndex import get_embedding_model, build_faiss_index

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import os
from typing import List, Tuple

from datetime import datetime

from time_question_helper import looks_like_recent_question, parse_created_at, get_most_recent_docs, dedupe_docs
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FAISS_INDEX_PATH = BASE_DIR / "weibo_faiss_index"
# ---------- Sort Document by timestamp ----------
def sort_docs_by_time_desc(docs: List[Document]) -> List[Document]:
    def parse_time(doc: Document):
        m = doc.metadata or {}
        t = m.get("created_at")
        if t is None:
            return datetime.min

        try:
            return datetime.fromisoformat(str(t))
        except Exception:
            return datetime.min # minimal time if parsing fails

    return sorted(docs, key=parse_time, reverse=True)


# ---------- Setup LLM and embeddings ----------

# os.environ['OPENAI_API_KEY'] = openaikey
# llm = OpenAI(temperature=0.4)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

llm = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=600
)

embeddings = get_embedding_model(provider="openai") # "hf" or "openai"

# ---------- Load FAISS vector store ----------
def load_faiss_vectorstore(index_path: str = str(FAISS_INDEX_PATH)) -> FAISS:
    print(f"Loading FAISS vector store from: {index_path}")
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Loaded FAISS vector store with", vectorstore.index.ntotal, "vectors.")
    # sanity check: inspect one stored doc
    any_id = list(vectorstore.docstore._dict.keys())[0]
    doc0 = vectorstore.docstore.search(any_id)
    print("DEBUG loaded doc created_at:", doc0.metadata.get("created_at"))
    print("DEBUG loaded doc create_time:", doc0.metadata.get("create_time")) 

    return vectorstore

# ---------- Format retrieved docs into context text ----------
def format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata or {}

        created_at = m.get("created_at", "Unknown time")
        likes = m.get("like_num", "N/A")
        comments = m.get("comment_num", "N/A")
        reposts = m.get("repost_num", "N/A")

        parts.append(
            f"[Post {i} | time={created_at} | likes={likes} | comments={comments} | reposts={reposts}]\n"
            f"{doc.page_content}\n"
        )

    return "\n\n".join(parts)

# ---------- Answer a question ----------
def answer_question(question: str, vectorstore: FAISS, k: int = 5) -> Tuple[str, List[Document]]:
    # retrieve more than k, trim later
    RETRIEVAL_FLOOR_FOR_RECENT = 15
    FINAL_CONTEXT_CAP = k
    semantic_k = max(k, RETRIEVAL_FLOOR_FOR_RECENT) if looks_like_recent_question(question) else k
    retriever = vectorstore.as_retriever(search_kwargs={"k": semantic_k})
    docs = retriever.invoke(question)  # list[Document]
    print("semantic docs:", len(docs))

    # If user asks "recent/latest", add newest posts to context
    if looks_like_recent_question(question):
        recent_docs = get_most_recent_docs(vectorstore, n=8)
        for d in recent_docs:
            m = d.metadata or {}
            print(m.get("created_at"), m.get("post_id"), m.get("weibo_id"))
        docs = dedupe_docs(docs + recent_docs)
        print("after dedupe:", len(docs))

    # sort by time desc
    docs = sorted(docs, key=parse_created_at, reverse=True)
    print("top10 times:", [ (d.metadata or {}).get("created_at") for d in docs[:10] ])

    # keep context short
    docs = docs[:FINAL_CONTEXT_CAP]
    print("final docs:", len(docs))

    if not docs:
        return "我没有找到和这个问题相关的微博内容，所以暂时无法回答。(I couldn't find any relevant posts.)"

    docs = sort_docs_by_time_desc(docs) # sort by time descending
    context = format_context(docs)

    prompt = f"""
You are a bilingual assistant (Chinese and English) answering questions about a Chinese actor's Weibo posts.

You are given some Weibo posts (each has Chinese and English text).
Use ONLY this content to answer the user's question.
If the information is not in these posts, say you don't know instead of guessing.

/* User question (may be Chinese or English): */
{question}

/* Relevant Weibo posts: */
{context}

/* Instructions:
- If the question is in Chinese, answer in Chinese.
- If the question is in English, answer in English.
- Base your answer ONLY on the posts above.
- If there is not enough information, say that honestly.
- If the question contains words related to time like "latest", "recent", "最近", "最新", or "new drama",
  pay special attention to posts with the most recent timestamps.
*/

Answer:
""".strip()

    answer = llm.invoke(prompt).content
    return answer, docs

if __name__ == "__main__":
    print("Weibo QA assistant ready. Ask a question (Chinese or English).")
    print("Example: 罗云熙最近在微博上有提到他的工作计划吗？")
    print("Example: What did he say about his latest drama?")
    print("-" * 60)

    vectorstore = load_faiss_vectorstore()
    try:
        while True:
            q = input("\nYour question (or 'exit'): ").strip()
            if q.lower() in {"exit", "quit"}:
                print("Bye!")
                break

            ans, docs = answer_question(q, vectorstore, k=5)
            print("\n--- Answer ---")
            print(ans)
    except KeyboardInterrupt:
        print("\nBye!")