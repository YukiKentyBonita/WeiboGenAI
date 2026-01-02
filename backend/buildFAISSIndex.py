import pandas as pd
import numpy as np
import faiss
import os
import re
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

# ---------- Normalize Weibo create_time format ----------
def normalize_weibo_create_time(raw: str, reference: datetime | None = None, default_year: int | None = None) -> str | None:
    """
    Converts time strings to 'YYYY-MM-DD HH:MM:SS' or returns None.
    """
    if raw is None:
        return None

    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None

    s = s.replace("\xa0", " ").strip()
    s = re.sub(r"\s*来自.*$", "", s).strip()

    #  default_year if provided
    if reference is None:
        if default_year is not None:
            reference = datetime(default_year, 1, 1)  
        else:
            reference = datetime.now()               

    # ISO-like: YYYY-MM-DD HH:MM(:SS)
    try:
        dt = datetime.fromisoformat(s)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    # Chinese: MM月DD日 HH:MM
    m = re.match(r"^\s*(\d{1,2})月(\d{1,2})日\s+(\d{1,2}):(\d{2})\s*$", s)
    if m:
        month, day, hh, mm = map(int, m.groups())

        # if default_year exists
        if default_year is not None:
            year = default_year
        else:
            year = reference.year
            if month > reference.month + 1:
                year -= 1

        dt = datetime(year, month, day, hh, mm, 0)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    return None


# ---------- Setup API key and create embeddings ----------
def get_embedding_model(provider: str = "hf"):
    """
    provider = "hf"      -> HuggingFace (free, local)
    provider = "openai"  -> OpenAI (paid, API)
    """
    if provider == "hf":
        print("Using HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)")
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN is not set. "
                "Run: export HUGGINGFACEHUB_API_TOKEN=your_token_here"
            )
        return HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    elif provider == "openai":
        print("Using OpenAI embeddings (text-embedding-3-small)")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

# ---------- Load processed posts ----------
def load_processed_posts(csv_path: str = "../data/processed/posts_processed.csv") -> pd.DataFrame:
    print(f"Loading processed posts from: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, engine='python')
    print(f"Loaded {len(df)} posts...")
    print("COLUMNS:", list(df.columns))
    print("HEAD ROW 0:", df.iloc[0].to_dict())
    return df

# ---------- Convert rows into Documents ----------
def build_documents(df: pd.DataFrame) -> list[Document]:
    documents: list[Document] = []
    for _, row in df.iterrows():

        # Combine Chinese and English content
        content_zn = str(row.get('content', ''))
        content_en = str(row.get('content_en', ''))
        content = f"Chinese: {content_zn}\nEnglish: {content_en}"

        raw_time = row.get("create_time") or row.get("created_at")  # supports either column name
        created_at = normalize_weibo_create_time(raw_time, default_year=2025)

        if len(documents) < 100:
            print("DEBUG raw_time:", repr(raw_time))
            print("DEBUG created_at:", repr(created_at))

        # Build metadata and document
        metadata = {
            'post_id': row.get('weibo_id') or None,
            'created_at': created_at,
            'raw_zn': content_zn,
            'raw_en': content_en,

            "like_num": row.get("like_num"),
            "comment_num": row.get("comment_num"),
            "repost_num": row.get("repost_num"),

            "has_image": row.get("raw_img") is not None,
            "has_video": row.get("video_link") is not None
        }
        documents.append(Document(page_content=content, metadata=metadata))
    print(f"Converted {len(documents)} rows into Documents.")
    return documents

# ---------- Text splitter  ----------
def SimpleTextSplitter(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50,) -> list[Document]:
    split_docs: list[Document] = []
    for doc in documents:
        text = doc.page_content or ""
        text = str(text)
        if not text.strip():
            continue
        start = 0
        text_length = len(text)
        step = max(chunk_size - chunk_overlap, 1)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]
            split_docs.append(Document(page_content=chunk_text, metadata=doc.metadata))
            start += step
    return split_docs

# ---------- Build FAISS index ----------
def build_faiss_index(csv_path: str, index_dir: str = "weibo_faiss_index"):
    df = load_processed_posts(csv_path)
    documents = build_documents(df)
    embeddings = get_embedding_model(provider="openai")

    print("Splitting documents into chunks...")
    split_docs = SimpleTextSplitter(documents, chunk_size=500, chunk_overlap=50)
    print(f"Split into {len(split_docs)} chunks.")

    print("Building FAISS index...")
    
    texts = [d.page_content for d in split_docs]
    metadatas = [d.metadata for d in split_docs]

    # Get embeddings
    raw_vectors = embeddings.embed_documents(texts)

    # Force 2D float32 matrix
    vectors = np.array(raw_vectors, dtype=np.float32)

    # Validate
    if vectors.ndim != 2:
        raise ValueError(f"Embeddings not 2D. Got shape={vectors.shape}, dtype={vectors.dtype}")

    if vectors.shape[0] == 0:
        raise ValueError("No embeddings returned.")

    if not vectors.flags["C_CONTIGUOUS"]:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)

    print("vectors dtype:", vectors.dtype, "shape:", vectors.shape, "contiguous:", vectors.flags["C_CONTIGUOUS"])

    index.add(vectors)

    # Build docstore and id mapping required by LangChain FAISS wrapper
    ids = [str(i) for i in range(len(texts))]

    docs_dict = {
        ids[i]: Document(page_content=texts[i], metadata=metadatas[i])
        for i in range(len(texts))
    }

    storevector = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(docs_dict),
        index_to_docstore_id={i: ids[i] for i in range(len(ids))},
    )

    print(f"Saving FAISS index to: {index_dir}...")
    storevector.save_local(index_dir)

    print("FAISS index built and saved successfully! ✅")

if __name__ == "__main__":
    build_faiss_index("../data/processed/posts_processed.csv", "weibo_faiss_index")