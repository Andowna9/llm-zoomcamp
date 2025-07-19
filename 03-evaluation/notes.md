# 03 - Evaluation Notes

**Evaluating the retrieval process** in a RAG pipeline is important because it helps to determine the search method that achieves better results, whether it is keyword or vector based.

A ground truth dataset is required to link queries to their known relevant documents. This correspondence is what will allow us to calculate [search/ranking evaluation metrics](./search_evaluation/evaluation-metrics.md).

## Ground Truth Data 
There are different ways of getting ground truth data:

- **Human annotators** -> very expensive
- **Automatically** based on user queries and evaluation results -> more complex
- **LLMs** -> simplest and cheapest, consider using **parallel processing** (threads) when making many calls

### ID Assignment

Each document must have a unique way of being identified. Two recommended options:

- **Generate the ID based on content** using a hashing function. Check for **ID collisions** and use a **combination of fields** that guarantees uniqueness.
- **Keep the ID** from the original source when possible.

### Dataset Generation

Check [ground-truth-data.ipynb](./search_evaluation/ground-truth-data.ipynb) as an example of leveraging LLMs for this task. The following prompt is used to generate 5 questions per document:

```python
prompt_template = """
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record. 

The record:

section: {section}
question: {question}
answer: {text}

Provide the output in parsable JSON without using code blocks:

["question1", "question2", ..., "question5"]
""".strip()
```

> This is a simplified example in which each question has only one relevant document.


## Search Evaluation

> For a full example of **evaluating text search with elasticsearch and minsearch**, check the [text evaluation notebook](./search_evaluation/evaluate-text.ipynb).
>
> For a full example of **evaluating vector search with minsearch and qdrant** check the [homework notebook](./homework.ipynb).

Before applying any metrics, search results are needed for every ground truth data record. For instance, as a list of boolean values where `True` means the relevant document is among the top results for that particular query.

```python
relevance_total = [
    [True, False, False, False, False], # Query 1
    [False, False, False, False, False], # Query 2
    [False, False, False, False, False], # Query 3
    [False, False, False, False, False], # Query 4
    [False, False, False, False, False] # Query 5
]
```

### Hit Rate

This metric evaluates the proportion of query results that include the relevant document out of the total searches performed.

```python
def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)
```

### MRR (Mean Reciprocal Rank)

This metric is similar, but also takes into account the rank of the relevant document. As the position at which the document is found increases the score is reduced by a fraction of its rank: `1/1`, `1/2`, `1/3`, ...

```python
def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)
```

> In both metrics, a higher value represents better performance.

## RAG Evaluation

Search is just one part of the RAG system. If we want to **evaluate how good the prompts and the LLM answers are** there are also other techinques available.

### Offline Evaluation

> Check the [offline evaluation notebook](./rag_evaluation/offline-rag-evaluation.ipynb) for a full example.

This kind of evaluation is done before deployment, ideal for **testing performance on different LLMs**. 

It consists of comparing **LLM generated answers** to **original answers** for ground truth data queries.

## Cosine Similarity

One way of evaluating answers similarity is using the cosine similarity metric.

```python
v_llm = model.encode(answer_llm)
v_orig = model.encode(answer_orig)

v_llm.dot(v_orig)
```

> As in vector search, the texts are vectorized using an embedding model first.

### LLM As Judge

Another way of evaluation is using an LLM as the one that compares and explains
why answers may be similar or not. An example prompt:

```python
prompt_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer compared to the original answer provided.
Based on the relevance and similarity of the generated answer to the original answer, you will classify
it as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Original Answer: {answer_orig}
Generated Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the original
answer and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()
```

> The prompt must be very specific about the result format to avoid parsing errors.


### Online Evaluation
This kind of evaluation is performed on an active system.

- A/B tests, experimental features
- User feedback: Are the answers good?

> Monitoring is closely related and consists of observing the overall health of the system.