# Agents Workshop

An agent is an AI system that can make autonmous decisions on what actions to take from observing
the environment. In case of LLMs, the environment is usually a chat and the idea is for the agent to
**decide which available tools are best for completing a specific task**, also cheking **context kept** from previous
interactions.

Capabilities of an agent when compared to basic RAG:

- Making multiple search queries.
- Combining information from different sources.
- Chaining actions together.
- Deciding when to stop searching.
- Using its own knowdledge if needed.

> Check out [the notebook](notebook.ipynb) for the full code.

## Agentic RAG

The agentic prompt tells the LLM that it can either answer directly or look up the context. Different responses are represented with output JSON templates.


```python
prompt_template = """
You're a course teaching assistant.

You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.
At the beginning the context is EMPTY.

<QUESTION>
{question}
</QUESTION>

<CONTEXT> 
{context}
</CONTEXT>

If CONTEXT is EMPTY, you can use our FAQ database.
In this case, use the following output template:

{{
"action": "SEARCH",
"reasoning": "<add your reasoning here>"
}}

If you can answer the QUESTION using CONTEXT, use this template:

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "CONTEXT"
}}

If the context doesn't contain the answer, use your own knowledge to answer the question

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "OWN_KNOWLEDGE"
}}

Make sure that each output you return is only a well formatted JSON.
""".strip()
```

Then, agentic RAG can be implmented like this:

```python
def agentic_rag(question):
    context = "EMPTY"
    prompt = prompt_template.format(question=question, context=context)
    answer_json = llm(prompt)
    answer = json.loads(answer_json)

    if answer['action'] == 'SEARCH':
        print('need to perform search...')
        search_results = search(question)
        context = build_context(search_results)
        
        prompt = prompt_template.format(question=question, context=context)
        answer_json = llm(prompt)
        answer = json.loads(answer_json)

    return answer
```

### Search action

Example search action when accesing FAQ is needed by LLM.
```json
{
    "action": "SEARCH",
    "reasoning": "The context is empty, 
        so I need to search the FAQ database."
 }
```

### Answer action

Example answer to a general question.
```json
{
    "action": "ANSWER",
    "answer": "To set up Docker, 
        you'll need to download and install 
        the Docker Desktop application 
        from the official Docker website.",
    "source": "OWN_KNOWLEDGE"
}
```

Example answer to a course question based on the search context from the FAQ.

```json
{
    "action": "ANSWER",
    "answer": "To get a course certificate, 
        you need to finish the course 
        with a 'live' cohort.",
    "source": "CONTEXT"
}
```

## Agentic Search

Now to perform multiple searches, a more complex prompt is needed. This one gives the LLM the possibility
to iteratively perform search and add context until the maximum iterations are reached or the answer action
is taken.

```python
prompt_template = """
You're a course teaching assistant.

You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.

The CONTEXT is build with the documents from our FAQ database.
SEARCH_QUERIES contains the queries that were used to retrieve the documents
from FAQ to and add them to the context.
PREVIOUS_ACTIONS contains the actions you already performed.

At the beginning the CONTEXT is empty.

You can perform the following actions:

- Search in the FAQ database to get more data for the CONTEXT
- Answer the question using the CONTEXT
- Answer the question using your own knowledge

For the SEARCH action, build search requests based on the CONTEXT and the QUESTION.
Carefully analyze the CONTEXT and generate the requests to deeply explore the topic. 

Don't use search queries used at the previous iterations.

Don't repeat previously performed actions.

Don't perform more than {max_iterations} iterations for a given student question.
The current iteration number: {iteration_number}. If we exceed the allowed number 
of iterations, give the best possible answer with the provided information.

Output templates:

If you want to perform search, use this template:

{{
"action": "SEARCH",
"reasoning": "<add your reasoning here>",
"keywords": ["search query 1", "search query 2", ...]
}}

If you can answer the QUESTION using CONTEXT, use this template:

{{
"action": "ANSWER_CONTEXT",
"answer": "<your answer>",
"source": "CONTEXT"
}}

If the context doesn't contain the answer, use your own knowledge to answer the question

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "OWN_KNOWLEDGE"
}}

Make sure that each output you return per iteration is only a well formatted JSON.

<QUESTION>
{question}
</QUESTION>

<SEARCH_QUERIES>
{search_queries}
</SEARCH_QUERIES>

<CONTEXT> 
{context}
</CONTEXT>

<PREVIOUS_ACTIONS>
{previous_actions}
</PREVIOUS_ACTIONS>
""".strip()
```

It can implemented using a `while` loop.
```python
search_queries = []
search_results = []
previous_actions = []

iteration = 0

while True:
    print(f'ITERATION #{iteration}...')

    context = build_context(search_results)
    prompt = prompt_template.format(
        question=question,
        context=context,
        search_queries="\n".join(search_queries),
        previous_actions='\n'.join([json.dumps(a) for a in previous_actions]),
        max_iterations=3,
        iteration_number=iteration
    )

    print(prompt)

    answer_json = llm(prompt)
    answer = json.loads(answer_json)
    print(json.dumps(answer, indent=2))

    previous_actions.append(answer)

    action = answer['action']
    if action != 'SEARCH':
        break

    keywords = answer['keywords']
    search_queries = list(set(search_queries) | set(keywords))
    
    for k in keywords:
        res = search(k)
        search_results.extend(res)

    search_results = dedup(search_results)
    
    iteration = iteration + 1
    if iteration >= 4:
        break

    print()
```

>The code manages the state of search results, queries and previous values so that the prompt holds a history to help the LLM. `dedup` is a function that removes duplicate results.


## Function Calling

Maintaining custom prompts that define agent actions and response logic is quite cumbersome. That is why *function calling* frameworks exist.

- Check https://console.groq.com/docs/tool-us

Create the tool function, which must return text:

```python
def search(query):
    boost = {'question': 3.0, 'section': 0.5}
    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5,
        output_ids=True
    )
    return json.dumps(results)
```

Define the tool's metadata, including its parameters.
```python
search_tool = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the FAQ database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to look up in the course FAQ."
                }
            },
            "required": ["query"],
        }
    }
}
```

Make a call to the model passing the available tools.

```python
response = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages=chat_messages,
    tools=[search_tool],
    tool_choice="auto"
)
```

Finally, process the response to see if the tool was
executed or a final response was given.

- Add the response message to the chat.
- Iterate over the tools used, if any, making calls and adding the results to the chat as context for further conversations.



> Be explicit about telling the LLm when to call each function in the system prompt.


