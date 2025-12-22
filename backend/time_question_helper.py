from datetime import datetime
import re
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---------- Heuristic to detect "recent" questions ----------
RECENT_PATTERNS = [
    r"\brecent\b", r"\blatest\b", r"\bnewest\b",
    r"最近", r"最新", r"近况", r"近期"
]

def looks_like_recent_question(q: str) -> bool:
    s = (q or "").lower()
    return any(re.search(p, s) for p in RECENT_PATTERNS)

def parse_created_at(doc: Document) -> datetime:
    t = (doc.metadata or {}).get("created_at")
    if not t:
        return datetime.min
    try:
        return datetime.fromisoformat(str(t))
    except Exception:
        return datetime.min

# def get_most_recent_docs(vectorstore: FAISS, n: int = 8) -> List[Document]:
#     all_docs = list(vectorstore.docstore._dict.values())
#     all_docs.sort(key=parse_created_at, reverse=True)
#     return all_docs[:n]

def get_most_recent_docs(vectorstore: FAISS, n: int = 8) -> List[Document]:
    all_docs = list(vectorstore.docstore._dict.values())
    all_docs.sort(key=parse_created_at, reverse=True)

    seen = set()
    unique_recent = []

    for d in all_docs:
        m = d.metadata or {}
        pid = m.get("post_id") or m.get("weibo_id") or d.page_content  # fallback

        if pid in seen:
            continue
        seen.add(pid)
        unique_recent.append(d)

        if len(unique_recent) >= n:
            break

    return unique_recent

def dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        m = d.metadata or {}
        key = m.get("post_id") or (m.get("created_at"), d.page_content[:50])
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out
