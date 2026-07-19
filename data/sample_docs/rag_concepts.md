# RAG Technical Reference: Core Concepts

**AI Engineering Reference | Version 1.4**

---

## Chunking Strategies

Before documents can be embedded and stored in a vector database, they must be split into chunks. Chunk size is one of the most impactful parameters in a RAG system.

**Fixed-size chunking** splits text at a fixed character or token count regardless of content boundaries. It is simple to implement but often creates chunks that cut sentences mid-way, losing semantic coherence.

**Recursive character splitting** uses a hierarchy of separators (paragraph breaks `\n\n`, line breaks `\n`, sentence endings `. `, word spaces ` `) and tries each in order. It splits on the most natural boundary that keeps chunks under the target size. This produces semantically coherent chunks while staying within token limits. It is the recommended default for unstructured text.

**Semantic chunking** uses the embedding model itself to identify topic boundaries: embed every sentence, compute similarity to adjacent sentences, and split where similarity drops sharply. This produces the most coherent chunks but is expensive (requires one embedding call per sentence).

**Typical parameters:**
- `chunk_size`: 256–1024 tokens. Smaller = more precise retrieval; larger = more context per chunk.
- `chunk_overlap`: 10–20% of chunk_size. Overlap ensures that content near chunk boundaries is fully represented in at least one chunk.

---

## Embedding Models Comparison

An embedding model transforms text into a dense vector. The quality of retrieval depends directly on the embedding model's ability to capture semantic meaning.

| Model | Dimensions | Speed | Cost | Best For |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fast | Free (local) | General purpose, English |
| `all-mpnet-base-v2` | 768 | Medium | Free (local) | Higher accuracy than MiniLM |
| `text-embedding-3-small` | 1536 | Fast | ~$0.00002/1K tokens | OpenAI-hosted, multilingual |
| `text-embedding-3-large` | 3072 | Medium | ~$0.00013/1K tokens | Highest OpenAI accuracy |

For most enterprise knowledge retrieval tasks in English, `all-MiniLM-L6-v2` provides excellent quality at zero cost. Use OpenAI's embeddings when multilingual support or maximum accuracy is required.

**Critical rule:** Documents and queries must always be embedded with the **same model**. Mixing models makes similarity scores meaningless.

---

## Similarity Search: Cosine, Dot Product, and L2

Vector databases find the most similar vectors using a distance (or similarity) metric.

**Cosine similarity** measures the angle between two vectors, ignoring magnitude. Two vectors pointing in the same direction have cosine similarity = 1.0; perpendicular vectors = 0.0; opposite vectors = -1.0. Cosine is the standard for text similarity because it is scale-invariant: a 1000-word document and a 10-word document can be equally similar to a query.

**Dot product** (inner product) measures both angle and magnitude. If vectors are L2-normalized (magnitude = 1), dot product equals cosine similarity. Many systems pre-normalize embeddings so these metrics are equivalent.

**L2 (Euclidean) distance** measures straight-line distance between vector endpoints. Lower = more similar. L2 is sensitive to magnitude, which makes it less suitable for text unless embeddings are normalized.

**Recommendation:** Use cosine similarity (or dot product with normalized embeddings) for RAG. ChromaDB's HNSW index defaults to cosine when you specify `{"hnsw:space": "cosine"}`.

---

## Metadata Filtering

Metadata filtering adds a pre-filter or post-filter step to similarity search, restricting results to documents that match specific criteria.

**Example use cases:**
- Retrieve only documents tagged with `department:legal`
- Retrieve only documents ingested after a certain date
- Retrieve only documents from a specific source file

**How ChromaDB implements it:** The `where` parameter accepts a MongoDB-style filter dict:
```python
{"tags": {"$in": ["legal", "compliance"]}}
{"source": "policy_manual.pdf"}
```

Filtering happens before similarity ranking in ChromaDB's HNSW index, so it doesn't increase search latency significantly.

---

## Re-ranking Approaches

After initial retrieval, re-ranking improves precision by applying a more expensive but accurate scoring model to the top-K candidates.

**Cross-encoder re-ranking:** A cross-encoder model takes (query, chunk) pairs and scores them jointly — much more accurate than embedding-based scoring but O(k) inference calls. Use models like `cross-encoder/ms-marco-MiniLM-L-6-v2` from Hugging Face.

**Reciprocal Rank Fusion (RRF):** Combine rankings from multiple retrieval strategies (e.g., dense embedding search + BM25 keyword search). Fuse the rank lists using RRF formula. Effective and computationally cheap.

**LLM-as-reranker:** Send the query and top-K chunks to an LLM and ask it to rank by relevance. Expensive but highest quality. Only practical with fast, cheap models.

For most enterprise RAG deployments, cross-encoder re-ranking with the top 20 candidates provides the best accuracy/cost trade-off.
