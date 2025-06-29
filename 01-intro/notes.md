# 01 - Introduction Notes

We'll treat LLMs as blackbox models that receive prompts as input and generate answers as output. Intuitively, they do so by predicting the most logical sequence of words that follow.

## RAG (Retrieval Augmented Generation)

RAG is a framework that enables LLMs to answer questions using external knowledge that is not part of their training data. It consists of two main steps:

- Retrieving relevant information in the form of documents from the knowledge base.
  
- Providing the LLM with an augmented prompt that includes the original question and the retrieved context.
  
![RAG](assets/RAG.png)

> The technologies used for the retrieval system and the LLM model can vary across different implementations.

[Source](https://python.langchain.com/docs/concepts/rag/)

## Configuring environment

Create Python environment.
```bash
python3 -m venv .venv
```

Activate environment and install dependencies.
```bash
pip install tqdm notebook==7.1.2 openai elasticsearch==8.13.0 pandas scikit-learn ipywidgets
```

## Retrieval

Run `parse-faq.ipynb` notebook to download FAQ documents from GDrive, parse and store them in `documents.json`.

**Search engines** can be used to perform document searches based on text queries.

### Minsearch
This is a simple in-memory search engine is based on the TF-IDF and cosine similarity.
[Source](https://github.com/alexeygrigorev/minsearch)

Indices can be created on several **text fields**. **Keyword fields** allow filtering the documents, in this case by course.
```python
index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)
index.fit(documents)
```
When performing the search, **boosting** determines the importance that each text field has to find the most relevant documents.
```python
boost = {'question': 3.0, 'section': 0.5}
results = index.search(
    query=q,
    filter_dict={'course': 'data-engineering-zoomcamp'},
    boost_dict=boost,
    num_results=5
)
```

### Elasticsearch
This is a widely-used, production search engine (memory persistent).

Running it with Docker.
```shell
docker run -it \
    --rm \
    --name elasticsearch \
    -m 4GB \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.17.6
```

>Check that it is up and running with `curl http://localhost:9200`.


Index settings for creation:

```json
{
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}
```
Query to retrieve documents:
```json
{
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": "<QUERY_HERE>",
                    "fields": ["question^3", "text", "section"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "data-engineering-zoomcamp"
                }
            }
        }
    }
}
```

Check more about [multimatch queries](./elastic-search.md).

## Generation

Groq has been chosen as a cloud LLM service since it has a free tier. Check [OpenAI alternatives](../awesome-llms.md).

Configuring OpenAI client to work with [Groq](https://console.groq.com/docs/openai).
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)
```
>Make sure that the `GROQ_API_KEY` env variable is available.

The typical prompt template has the following structure:
```python
template = """
You're a course teaching assistant. 
Answer the QUESTION based on the CONTEXT from the FAQ database. 
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
"""
```

Once the prompt is formatted with the question and context documents, sending it to one of Groq's models is achieved with the following code:
```python
response = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages=[
        {
            "role": "user", 
            "content": prompt
        }
    ]
)
answer = response.choices[0].message.content
```

### Estimating costs

Costs are estimated based on the number of input/output tokens. Tokens are integer values that represent a unit of text and are understood by LLMs. Each model uses a different encoding; for example, OpenAI's models use the `tiktoken` library.

```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o")
tokens = encoding.encode(prompt)
num_tokens = len(tokens)
```

## Overview

In terms of code, if we add all the pieces together a modular RAG pipeline looks like this:
```python
def rag(query: str) -> str:
    results = search(query)
    prompt = build_prompt(query, results)
    answer = llm(prompt)
    return answer
```
> `results` is the list of documents. Full code on [rag-intro notebook](./rag-intro.ipynb).

