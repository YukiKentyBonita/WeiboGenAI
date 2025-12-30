# Weibo GenAI Q&A System — Design & Implementation

This document provides a detailed, end-to-end explanation of the system architecture,
data pipeline, retrieval logic, and code structure for the Weibo GenAI Q&A application.

It is intended for:
- personal maintenance and future extension
- technical interviews and system design discussions

---

## 1. System Overview

### 1.1 Problem Statement

The Weibo GenAI Q&A System enables users to ask natural-language questions (in English or Chinese) about the public posts of a specific Weibo account and receive concise, context-aware answers.
The system collects Weibo posts using authenticated cookies, transforms the raw content into searchable vector embeddings, and applies a retrieval-augmented generation (RAG) approach to generate LLM-generated answers in the original Weibo data. Additional handling is implemented for time-related questions (e.g., “recent”, “latest”) to ensure chronologically relevant information is considered.

### 1.2 High-Level Flow

```text
User enters a question (English or Chinese)
    ↓
(Optional) Query expansion and normalization
    ↓
Semantic retrieval from vector store (FAISS)
    ↓
Recency-aware adjustment for time-related questions
    ↓
Context construction from retrieved posts
    ↓
LLM invocation with structured prompt and context
    ↓
LLM generates a grounded natural-language answer
```

---

## 2. Data Ingestion

### 2.1 Weibo Data Source
The system ingests data from a specific Weibo account by scraping **publicly available content**, including posts, basic profile information, and social relationship metadata (followers/following).
Authenticated cookies are used to access and retrieve public content in a manner consistent with how a logged-in user would view the platform. These cookies are obtained manually via a web browser’s developer tools and are used solely to authenticate requests during data collection.

**Assumptions and constraints:**
- Only public Weibo content is collected
- Cookie-based authentication must remain valid during scraping
- Changes to Weibo’s frontend or access policies may require scraper updates

### 2.2 Raw Data Format

Scraped data is stored as CSV files before any processing or transformation. Each file captures a different aspect of the Weibo account’s data.

**Raw data files:**

- `follows.csv`  
  - `uid1`: source user ID  
  - `uid2`: target user ID  
  - `crawl_time`: timestamp of data collection  
  - `relationship`: follower/following relationship type  

- `posts.csv`  
  - `uid`: user ID  
  - `weibo_id`: unique post identifier  
  - `origin_link`: original Weibo post URL  
  - `content`: post text content  
  - `like_num`: number of likes  
  - `repost_num`: number of reposts  
  - `comment_num`: number of comments  
  - `create_time`: post creation timestamp  
  - `crawl_time`: timestamp of data collection  
  - `device`: posting device (if available)  
  - `img`, `raw_img`: image URLs  
  - `video_link`: video URL (if present)  
  - `location`: location metadata (if available)  

- `userprofile.csv`  
  - `userid`: user ID  
  - `nickname`: display name  
  - `gender`: reported gender  
  - `birthday`: reported birthday  
  - `province`, `city`: location metadata  
  - `introduction`: profile description  
  - `vip_level`: VIP status  
  - `labels`: user-defined labels  
  - `authentication`: verification information  

All raw data files are stored in the following directory:

```text
./data/raw/
```

---

## 3. Data Processing & Cleaning

### 3.1 Data Cleaning
- Raw Weibo data is cleaned to remove fields that are not relevant to semantic retrieval or question answering (crawl_time, device, etc.)
- This step reduces noise in the dataset and ensures that embeddings are generated only from semantically meaningful text content.

### 3.2 Language Handling

The system supports both Chinese and English queries.

- Original Weibo posts are in Chinese
- User questions may be in either Chinese or English

Amazon Translate is used to translate post content in advance. Translated text is stored alongside the original content and used during embedding and retrieval.

---

## 4. Embedding & Indexing

### 4.1 Embedding Model

The system supports two embedding models:

- **HuggingFace `all-MiniLM-L6-v2`**
  - Free and runs locally
  - Suitable for development and small-scale experiments

- **OpenAI `text-embedding-3-small`**
  - Paid API-based model
  - Provides improved semantic accuracy for retrieval

The choice between models depends on factors such as project scale, cost constraints, and desired retrieval quality.

### 4.2 FAISS Index

The system uses a FAISS vector index for efficient similarity search.

- `faiss.IndexFlatL2` is used for exact nearest-neighbor search
- This index type is simple, reliable, and appropriate for the current dataset size

At the current stage, dynamic index updates (incremental insertion or deletion) are not supported. The index is rebuilt when new data is ingested. 

---

## 5. Question Understanding & Retrieval

### 5.1 Query Expansion / Normalization
- User input questions are expanded into search-based queries
- The expanded query is used internally to guide semantic retrieval. The original user question is preserved and passed to the LLM during answer generation. This allows the system to improve retrieval accuracy without altering the user’s original intent.

### 5.2 Semantic Retrieval

Semantic retrieval is performed using vector similarity search over a FAISS index built from embedded Weibo posts.
- The retrieval parameter `k` controls the number of semantically similar posts returned
- A relatively small `k` is currently used during development and testing
- Further experimentation is planned to determine an optimal `k` value based on retrieval quality and context window constraints

### 5.3 Time-Aware Retrieval Logic

The system includes special handling for time-related queries, such as those asking about recent or latest activity.
Time-related intent is detected by checking for keywords including:
`recent`, `latest`, `newest`, `最近`, `最新`, `近况`, `近期`

When detected:
- Semantic retrieval is performed as usual
- Additional posts are selected based on recency (newest first)
- Recent posts are injected into the retrieved document set

This logic is necessary because semantic similarity search alone does not account for temporal relevance and may fail to detect the most recent posts.

---

## 6. Context Construction

### 6.1 Deduplication

Before constructing the final context passed to the LLM, duplicate documents are removed.
- Duplicates may arise due to overlap between semantic retrieval results and recency-based retrieval
- Documents are deduplicated using a stable unique identifier, `post_id`
- Each Weibo post is included at most once in the final context

### 6.2 Context Window Management
The system enforces limits on the number of documents included in the final context.
- Recent documents: up to 8 posts
- Semantic documents: user-controlled range (`min=1`, `max=10`)
After deduplication, all selected posts are sorted in descending order by creation time before being formatted into a single context block. This ordering prioritizes recent information while still considering semantic relevance.

---

## 7. Answer Generation

### 7.1 Prompt Structure

The final prompt sent to the LLM consists of:
- The user’s original question
- The internally generated search-focused query
- Set of relevant Weibo posts retrieved from the vector store
- Additional system-level instructions to guide answer style and grounding

This structure ensures that the LLM generates answers based on retrieved Weibo content rather than relying on its internal knowledge.

### 7.2 Output Format

- The system returns only the final natural-language answer to the user
- For development and debugging purposes, relevant retrieved posts and intermediate metadata can be optionally avaliable to the developer

This separation keeps the user experience simple while allowing inspection and evaluation during development.

---

## 8. Code Structure & Key Files

### 8.1 Backend Overview
The `backend/` directory contains the core system logic, including:
- FAISS index creation and loading
- Semantic and time-aware retrieval logic
- Embedding model and LLM configuration
- The Streamlit-based web application for user interaction

### 8.2 Data Handling Scripts
The `datahandling/` directory contains scripts responsible for:
- Downloading raw Weibo data
- Cleaning and preprocessing post content
- Preparing data for embedding and indexing

These scripts form the offline data preparation pipeline for the system.

### 8.3 Entry Points
The application can be accessed in two ways:
- Running scripts directly from the command line for development and testing
- Accessing the Streamlit web application via the following URL: `https://weibogenai-uxshzkf34fe5axdxw3sttt.streamlit.app/`

---

## 9. Known Limitations

- Dynamic post updates are not currently supported; new posts require re-running the ingestion pipeline and rebuilding the FAISS index
- All posts are currently translated into English in advance, which may introduce noise for Chinese-language queries

A hybrid retrieval strategy using both original Chinese posts and translated English posts is planned to address this limitation.

---

## 10. Future Improvements

- Improve prompt design by providing richer contextual information to the LLM
- Replace local CSV storage with a database to support incremental updates
- Implement hybrid FAISS indices for multilingual retrieval

---
