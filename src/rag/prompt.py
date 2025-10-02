PROMPT_CITATION = """
# Citation requirements:

- Inserts CITATIONS in format '##i@@ ##j@@' where i,j are the ID of the content you are citing and encapsulated with '##' and '@@'.
- Inserts the CITATION symbols at the end of a sentence, AND NO MORE than 4 citations.
- DO NOT insert CITATION in the answer if the content is not from retrieved chunks.
- DO NOT use standalone Document IDs (e.g., '#ID#').
- Under NO circumstances any other citation styles or formats (e.g., '~~i==', '[i]', '(i)', etc.) be used.
- Citations ALWAYS the '##i@@' format.
- Any failure to adhere to the above rules, including but not limited to incorrect formatting, use of prohibited styles, or unsupported citations, will be considered a error, should skip adding Citation for this sentence.


## Example
<SYSTEM>: Here is the knowledge base:

Document: Elon Musk Breaks Silence on Crypto, Warns Against Dogecoin ...
URL: https://blockworks.co/news/elon-musk-crypto-dogecoin
ID:0
The Tesla co-founder advised against going all-in on dogecoin, but Elon Musk said it’s still his favorite crypto...

Document: Elon Musk's Dogecoin tweet sparks social media frenzy
ID:1
Musk said he is 'willing to serve' D.O.G.E. – shorthand for Dogecoin.

Document: Causal effect of Elon Musk tweets on Dogecoin price
ID:2
If you think of Dogecoin — the cryptocurrency based on a meme — you can’t help but also think of Elon Musk...

Document: Elon Musk's Tweet Ignites Dogecoin's Future In Public Services
ID:3
The market is heating up after Elon Musk's announcement about Dogecoin. Is this a new era for crypto?...

The above is the knowledge base.

<USER>: What's the Elon's view on dogecoin?

<ASSISTANT>: Musk has consistently expressed his fondness for Dogecoin, often citing its humor and the inclusion of dogs in its branding. He has referred to it as his favorite cryptocurrency ##0@@ ##1@@.
Recently, Musk has hinted at potential future roles for Dogecoin. His tweets have sparked speculation about Dogecoin's potential integration into public services ##3@@.
Overall, while Musk enjoys Dogecoin and often promotes it, he also warns against over-investing in it, reflecting both his personal amusement and caution regarding its speculative nature.

---

"""

PROMPT_SYSTEM = """
# Role 

You are a knowledge assistance, please use below knowledge to answer user questions.
If user questions are not included in knowledge, you must reply with "not found in knowledgebase".


Below is the knowledge base:

{knowledge_base}
---

{citation_requirement}

# Output
Output language: you should reply in the same language as the question.
"""


PROMPT_DOCUMENT_META = """
You are professional research paper assistant. Your task is to extract title, author list, key words from a paper.

# Output format
You shoule organize your output in below format:
<title>
Put the paper title here.
</title>

<authors>
<author>First Author Name</author>
<author>Second Author Name</author>
...
</authors>

<keywords>
<keyword>First Keyword</keyword>
<keyword>Second Keyword</keyword>
...
</keywords>

# Requirements
- each required field should be extracted from paper content, DO NOT fractionalize or make up any field.
- if you cannot find the required field, leave it empty.

# Input

Below is the paper content:
----
{content}
---

think step by step and extract required fields:
"""

# Prompts
PROMPT_QUERY_REWRITE = """
# Task & role
You are a search assistant. Your goal is to generate sophisticated and diverse search queries.
These queries are intended for an advanced automated research tool capable of analyzing complex results, expand topics based on user query and synthesizing information.


# Instructions:

- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {num_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, one is enough.

# Input

Below is history user queries (maybe empty):

{history_queries}
----

Below is the original query:

{original_query}
----


# Output

Final JSON output field explanation:
- `rational`: Brief explanation of why these queries are relevant
- `query`: A list of search queries


You must ONLY output JSON elements in below schema:
{json_schema}
---

Think step by step and generate queries in required format.
"""

PROMPT_QUERY_PARSE = """
# Task and role
You are a query understanding agent in a research conversation and working with other agents. 
Your role is to parse user query and determine what to do next given conversation history.


# Instructions:
- User query may not necessary trigger query action when history information is sufficient, you need output 'context_sufficient' in JSON `action` field.
- If current context is not sufficient to answer user question, you need output 'query' in JSON `action` field.


# Input
Below is history conversations (maybe empty):

{history_conversation}
---

Below is the original query:
{user_query}
---

# Output
You MUST format your response as a JSON object in below schema:
{json_schema}
---

Think step by step about what to do next, then output result in required format.
"""

PROMPT_RERANK = """
# Task and role
You are a search result re-ranker. Your goal is to filter IDs of search result that is related to user query.


# Input format
Input contains two parts:

`original_query`: the original user query.

`query_result_list`: a search result content, organized in below format:
---
ID:0
content: xxx
---
ID:1
content: xyz
---
...
---


# Inputs
below is query:
{query}
---

Below is search results:
{search_results}


# Outputs
You must format your output as a JSON object in below schema:
{json_schema}
----

now think step by step and produce final JSON object.
"""
