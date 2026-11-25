
## Experiment: Character vs Language-Aware Chunking

Repository: ChargeFlow

Chunk overlap: 100

| Method | Chunk Size | Total Chunks | Average Length | Maximum Length |
|---|---:|---:|---:|---:|
| Character | 500 | 1146 | 402.96 | 500 |
| Language | 500 | 1220 | 369.65 | 500 |
| Character | 1000 | 525 | 843.19 | 998 |
| Language | 1000 | 598 | 731.64 | 1000 |
| Character | 1500 | 345 | 1271.22 | 1498 |
| Language | 1500 | 403 | 1075.65 | 1499 |

### Observations

1. Smaller chunk sizes generated more chunks.
2. Language-aware splitting generated more chunks than character-based splitting in this experiment.
3. Language-aware splitting produced shorter average chunks.
4. Chunk statistics alone cannot determine retrieval quality.

### Conclusion

I will use language-aware splitting with a chunk size of
1000 and overlap of 100 for the initial implementation.
I will evaluate retrieval quality later.




## Phase 3: Embeddings and Vector Database

### Implementation
- Used HuggingFace all-MiniLM-L6-v2 for embeddings.
- Generated 384-dimensional embeddings.
- Stored 598 code chunks in PostgreSQL using pgvector.
- Configured cosine distance for vector comparison.
- Implemented persistent database storage.
- Added unique IDs for chunks.
- Added database rebuild functionality.

### Experiments
- Tested similarity search with different queries.
- Examined cosine distance scores.
- Verified database persistence.
- Tested language-based metadata filtering.
- Tested Maximum Marginal Relevance (MMR).

### Observations
- Lower cosine distance indicates greater similarity.
- Metadata filtering restricts retrieval to matching languages.
- MMR balances relevance and diversity.
- Existing databases can be loaded without regenerating embeddings.

### Current Database
- Repository: ChargeFlow
- Files loaded: 68
- Chunks stored: 598
- Embedding dimensions: 384
