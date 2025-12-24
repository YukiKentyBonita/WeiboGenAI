# Weibo GenAI Q&A Application

A bilingual (Chinese–English) GenAI question-answering system built on Weibo posts, combining web scraping, semantic search (FAISS), and LLM-based answering with time-aware retrieval.

---

## Overview

This project is an end-to-end GenAI application that allows users to ask natural-language questions about a specific Weibo account’s posts and receive concise, context-aware answers.
The system scrapes Weibo content, processes and stores post text with metadata, indexes it using vector embeddings, retrieves relevant posts, and constructs a grounded answer using a large language model. 

---

## Current Features

- Scrape public Weibo posts using authenticated cookies  
- Bilingual (Chinese–English) question support  
- Vector-based semantic search using FAISS  
- Time-aware retrieval for “recent / latest” questions  
- Context assembly with deduplication and size limits  
- LLM-based answer generation with debug logging  

---

## Example Questions

```text
What has he been posting about recently?
Did he mention traveling this year?
最近他有没有提到工作相关的内容？
他最近的动态是什么？
```

---

## High-Level Architecture

```text
User Question
    ↓
Query Expansion
    ↓
Vector Retrieval (FAISS)
    ↓
Recency Boost + Deduplication + Context Curation
    ↓
LLM Answer Generation
    ↓
Final Answer
```

---

## Project Structure

```text
.
├── backend/              # Ingestion, processing, embedding, indexing, Q&A orchestration (question → answer)
    └──  weibo_faiss_index/    # Persisted FAISS vector index
├── data/                 # Raw and processed data (CSV)
    └── raw
    └── processed
├── docs/
│   └── DESIGN.md         # Full system documentation (one big self-doc)
├── datahandling          # Data retrieval and preprocessing
├── README.md
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Weibo cookies for authenticated scraping
- OpenAI API key (for embeddings and LLM)

### Installation

```bash
- git clone <repo-url>
- cd <repo-name>
- python -m venv venv
- source venv/bin/activate
- pip3 install -r requirements.txt
```

### Configuration
- export OPENAI_API_KEY="YOUR_KEY_HERE"
- additional config (cookies path, model name, etc.) if needed
- change cookies path in datahandling/PostsDoloader.py to specify which user's posts to download

### Typical Workflow
1) Scrape / ingest Weibo posts: python3 PostsDownloader.py
2) Clean and process text data: python3 DataPreprocessing.py
3) Build embeddings and FAISS index: python3 buildFAISSIndex.py
4) Ask questions via the Q&A module: python3 -m streamlit run backend/weibo_streamlit_app.py

---

## Design Goals

- Ground answers strictly in retrieved Weibo content
- Support time-sensitive questions reliably
- Keep token usage and costs predictable
- Maintain clear separation between ingestion, retrieval, and generation

---

## Documentation

- README.md – High-level overview and entry point
- docs/DESIGN.md – Complete system documentation (architecture + file/function explanations)
- Docstrings – Function-level documentation (used for generated docs)


