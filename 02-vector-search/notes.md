# 02 - Vector Search Notes

As already seen in the [first module](../01-intro/notes.md), search engines are based on exactly matching the (key)words contained in a query to documents. This obviously has its limitations as concepts may be expressed in many different ways.

## What Is Vector Search?

Vector search is broader and introduces more capabilities. It can match texts that may not use the same words, but share a common contex bacause it works at a **semantic-level** (meaning of words). Besides **text**, it and can be used to perform multimodal searches with **images**, **videos** or **audio**.

Similarity is mesured numerically between vectors representing the data. These are known as **embeddings** and they are inferred by an **embedding model** (deep neural network).

## Data Preparation

* Make sure the data is **cleaned and chunked** (divided into pieces) so that it can be digested by embedding models:  `answer`.
* Determine **vectorized fields** and **metadata fields** for filtering: `course`, `section`.

>Depending on the data, relying on just overlaping (key)words to match questions to answers can lead to inaccurate results if the context is meaningful.

## Embedding Model

The choice of the best embedding model is made according to certain **factors**:
* The domain of the data.
* A trade-off between precision and resource usage (storage and memory).
* The computational cost of inference.

>Benchmarking different alternatives is recommended.



### Fast Embed

Fast Embed is an embedding solutions optimized to use with Qdrant. It has low-latency and CPU-friendly, which means it is good to perform local inference. It supports the most common types of embeddings, including dense, sparse and multivector.

Listing supported models.
```python
from fastembed import TextEmbedding

TextEmbedding.list_supported_models()
```

The chosen model is [jina-embeddings-v2-small-en](https://huggingface.co/jinaai/jina-embeddings-v2-small-en), which is a dense (trained using deep neural networks) embedding suitable for unimodal English text. It outputs small vectors of `512 dimensions` and uses `cosine similarity` as a distance metric (the closer the values the closer the cosine's value is to 1).


## Quadrant

Quadrant is a dedicated and scalable vector search solution. We will use it for semantic similarity search as part of RAG, although it has more use cases.

* Check [Quadrant articles](https://qdrant.tech/articles/) for a deep-dive.
* Check `semantic_search.ipynb` for the full vector search sample code.

### Basic Configuration

Running it with Docker.

```shell
docker run -p 6333:6333 -p 6334:6334 \
   -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
   qdrant/qdrant
```

- **Port 6333** – REST API
- **Port 6334** – gRPC API

> It comes with a web ui to help explore data visually: http://localhost:6333/dashboard.

Installing `quadrant` client and `fastembed` data vectorization library:

```shell
pip install -q "qdrant-client[fastembed]>=1.14.2"
```
Setting up client.
```python
from qdrant_client import QdrantClient, models

client = QdrantClient("http://localhost:6333")
```

### Vector Indexing

Creating a collection (vector index) for a specific problem.
```python
collection_name = 'zoomcamp-rag'
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=512,
        distance=models.Distance.COSINE
    )
)
```
* We need to specify the **same configuration that the embedding uses**.

A point represents a single document or chunk. It consists of an **ID** (integer or UUID), the **embedding** for the data point and **payload** to add metadata for interpretability/filtering.

```python
model_handle = 'jina-embeddings-v2-small-en'
point = models.PointStruct(
   id=0,
   vector=models.Document(text=doc['text'], model=model_handle),
   payload={
         "text": doc['text'],
         "section": doc['section'],
         "course": course['course']
   }
)
```


Points can be added to a collection once constructed. Fastembed will  automatically download and use the declared embedding model.

```python
client.upsert(
    collection_name=collection_name,
    points=[point]
)
```

>Besides this basic upsert (one API call per point), check other ways to efficiently [upload points](https://qdrant.tech/documentation/concepts/points/#upload-points).

### Visualizing Vectors

The vector space (2D projection) and closeness of data points (colored by course) can be visualized using the UI with this query:
```json
{
  "limit": 948,
  "color_by": {
    "payload": "course"
  }
}
```

### Applied Vector Search
Vector search coverts query text into an embedding and compares it
with stored points using the distance metric defined.
```python
results = client.query_points(
   collection_name=collection_name,
   query=models.Document(
      text=query,
      model=model_handle 
   ),
   limit=5,
   with_payload=True
)
```

#### Filtering

Filtering requires creating an index for payload data.

```python
client.create_payload_index(
    collection_name=collection_name,
    field_name="course",
    field_schema="keyword"
)
```

An example search filter, requiring semantic
results to match the course provided.
```python
...
 query_filter=models.Filter(
   must=[
         models.FieldCondition(
            key='course',
            match=models.MatchValue(
               value='mlops-zoomcamp'
            )
         )
   ]
)
...
```

> More about it on [Qdrant Filtering](https://qdrant.tech/articles/vector-search-filtering/).

## Hybrid Search

Hybrid search combines different retrieval methods to build more reliable systems. Statistical keyword-based techniques are often used alongside semantic search:

* **TF-IDF**: The Term Frequency (TF) increases with a term’s occurrences in a document, though not linearly. The Inverse Document Frequency (IDF) lowers the weight of terms common across all documents, highlighting distinctive ones.

* **BM25**: An enhanced version of TF-IDF that adds document length normalization, preventing longer documents from having an unfair advantage.

These methods typically use sparse vectors, where most dimensions are zero. Non-zero entries indicate the presence of specific terms. Sparse representations can adapt to new, unseen words by adding new dimensions dynamically.

Hybrid search can be implemented in two common ways:

* Using a dense retriever (semantic) followed by a sparse reranker (statistical).

* Applying Reciprocal Rank Fusion (RRF) to merge rankings from multiple retrieval methods.

Check [hybrid search notebook](./hybrid_search.ipynb) for a **simple implementation** in Quadrant.

## Usage in RAG

The `rag.ipynb` notebook implements the same [modular RAG pipeline](../01-intro/notes.md#overview) switching the text/keyword search function with vector search.

The document's question and answer fields are joined and embeded because both are considered semantically useful for the RAG system. 
```python
text = doc['question'] + ' ' + doc['text']
vector = models.Document(text=text, model=model_handle)
```

## Cosine Similarity

Check [vector search homework](./homework.ipynb) for a better understanding of how cosine similarity works.

## Building A search Engine

Check the [vector search engine notebook](./vector_search_engine.ipynb) to review how to build a basic one from scratch, including how to apply the `hit-rate` metric to measure its performance.


