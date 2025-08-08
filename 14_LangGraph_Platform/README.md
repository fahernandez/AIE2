<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 14: Build & Serve Agentic Graphs with LangGraph</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| – | – | – | – | You are here! | – | – |

# Build 🏗️

Run the repository and complete the following:

- 🤝 Breakout Room Part #1 — Building and serving your LangGraph Agent Graph
  - Task 1: Getting Dependencies & Environment
    - Configure `.env` (OpenAI, Tavily, optional LangSmith)
  - Task 2: Serve the Graph Locally
    - `uv run langgraph dev` (API on http://localhost:2024)
  - Task 3: Call the API
    - `uv run test_served_graph.py` (sync SDK example)
  - Task 4: Explore assistants (from `langgraph.json`)
    - `agent` → `simple_agent` (tool-using agent)
    - `agent_helpful` → `agent_with_helpfulness` (separate helpfulness node)

- 🤝 Breakout Room Part #2 — Using LangGraph Studio to visualize the graph
  - Task 1: Open Studio while the server is running
    - https://smith.langchain.com/studio?baseUrl=http://localhost:2024
  - Task 2: Visualize & Stream
    - Start a run and observe node-by-node updates
  - Task 3: Compare Flows
    - Contrast `agent` vs `agent_helpful` (tool calls vs helpfulness decision)

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

- Create and deploy a locally hosted MCP server with FastMCP.
- Extend your tools in `tools.py` to allow your LangGraph to consume the MCP Server.
</details>

# Ship 🚢

- Running local server (`langgraph dev`)
- Short demo showing both assistants responding

# Share 🚀
- Walk through your graph in Studio
- Share 3 lessons learned and 3 lessons not learned


#### ❓ Question:

What is the purpose of the `chunk_overlap` parameter when using 
`RecursiveCharacterTextSplitter` to prepare documents for RAG, and what trade-offs 
arise as you increase or decrease its value?

### Answer 
The parameter controls how many characters are shared between consecutive text chunks when splitting documents for RAG. Its main purpose is to ensure that important information near the boundaries of one chunk is not lost and remains accessible in the next chunk.

Increasing th `chunk_overlap` reduces the risk of missing relevant context at chunk boundaries, improving the likelihood that a retriever or LLM accesses all necessary information, but it also increases redundancy, leading to more duplicated content across chunks, higher storage and retrieval costs, and potentially more overlapping or repetitive results. Whereas decreasing `chunk_overlap` reduces duplication, making storage and retrieval more efficient and minimizing repeated content in responses, but it raises the risk that information split at chunk boundaries is lost or inaccessible to the model, which can harm retrieval quality and answer completeness.

#### ❓ Question:
Your retriever is configured with `search_kwargs={"k": 5}`. How would adjusting `k` 
likely affect RAGAS metrics such as Context Precision and Context Recall in practice, 
and why?

### Answer 
Increasing the `k` means your retriever will return more documents per query. In practice, raising `k` tends to increase Context Recall—because there’s a higher chance that at least one of the retrieved chunks contains the ground truth answer or relevant context. However, as you increase `k`, Context Precision often decreases, since more retrieved documents are likely to be irrelevant or only loosely related to the query. Conversely, lowering `k` usually improves Context Precision (a higher proportion of retrieved chunks are relevant), but may reduce Context Recall (greater risk of missing the needed information). The optimal `k` balances these trade-offs based on your use case and the redundancy or density of relevant information in your corpus.

#### ❓ Question:

Compare the `agent` and `agent_helpful` assistants defined in `langgraph.json`. Where does the helpfulness evaluator fit in the graph, and under what condition should execution route back to the agent vs. terminate?

### Answer
The helpfulness evaluator acts as a decision node in the `agent_helpful` graph. After the agent generates a response, the evaluator checks whether the answer is sufficiently helpful with respect to the original user query. If the response is deemed helpful, the graph terminates. If not, execution loops back to the agent node for another attempt, allowing the agent to refine its answer. This mechanism ensures that only responses meeting a certain standard of helpfulness are returned to the user.