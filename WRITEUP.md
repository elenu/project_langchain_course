# Project Writeup

## 1. Architecture & Routing Decisions

- **Overall design:** The system is a workflow-based assistant built with LangGraph and LangChain-style components. The orchestrator is created in [src/agent.py](src/agent.py) via `create_workflow()` which compiles a StateGraph representing the agent pipeline.
- **Entry point & routing:** The graph entry node is `classify_intent`. That node classifies the user's intent into one of `qa`, `summarization`, `calculation` (or `unknown`) and returns a short routing key in `next_step`. The graph uses conditional edges to route to `qa_agent`, `summarization_agent`, or `calculation_agent` and then flows to `update_memory` before `END`.
- **Agent nodes:** Each agent node constructs a chat prompt using the templates in [src/prompts.py](src/prompts.py). Agents call `invoke_react_agent()` which binds tools and calls the LLM agent with a response schema. See `qa_agent`, `summarization_agent`, and `calculation_agent` in [src/agent.py](src/agent.py).
- **Tools & capabilities:** Tools live in [src/tools.py](src/tools.py). Available tools include: calculator, document search, document reader, and document statistics. Tools are created in `get_all_tools()` and passed into the compiled workflow so agents can use them during LLM tool-enabled calls.

## 2. State & Memory

- **State container:** The state shape is defined by `AgentState` in [src/agent.py](src/agent.py). Key fields: `messages`, `user_input`, `intent`, `next_step`, `conversation_summary`, `active_documents`, `current_response`, `tools_used`, `session_id`, `user_id`, and `actions_taken`.
- **Session persistence:** Session metadata and conversation history are stored as `SessionState` objects in [src/schemas.py](src/schemas.py) and persisted as JSON files under the `sessions/` folder by `DocumentAssistant` in [src/assistant.py](src/assistant.py). The assistant exposes `start_session()` and `process_message()` to manage sessions and invoke the workflow.
- **In-graph memory updates:** After an agent node completes, `update_memory` summarizes the recent conversation using the `MEMORY_SUMMARY_PROMPT` (defined in [src/prompts.py](src/prompts.py)) and a structured schema `UpdateMemoryResponse` (in [src/schemas.py](src/schemas.py)). The summary and discovered `document_ids` are written back to the workflow state and persisted to the session file.
- **Tool usage logging:** Each tool uses the `ToolLogger` (in [src/tools.py](src/tools.py)) to record calls to the `logs/` directory. This helps auditing and reproducing agent decisions.

## 3. Structured Output

- **Schema enforcement:** Structured outputs are implemented with Pydantic models in [src/schemas.py](src/schemas.py): `AnswerResponse`, `SummarizationResponse`, `CalculationResponse`, `UpdateMemoryResponse`, and `UserIntent`.
- **LLM integration:** The code uses the LLM's structured output features: `llm.with_structured_output(SomeSchema)` and `create_agent(..., response_format=SomeSchema)` to request model outputs that conform to the schema. See `classify_intent()` and `invoke_react_agent()` in [src/agent.py](src/agent.py).

## 4. Example Conversations

Below are concise examples showing expected inputs and high-level outputs. These demonstrate routing, tool use, and structured returns.

- Example 1 — Q&A (document lookup)
  - User: "Which invoice mentions Acme Corporation and how much was charged?"
  - Flow: `classify_intent` → `qa_agent` (invokes document_search and/or document_reader) → `update_memory`
  - Expected structured response (high-level):
    - `intent`: `{ "intent_type": "qa", "confidence": 0.9, "reasoning": "..." }`
    - `response.current_response` conforms to `AnswerResponse` with `question`, `answer`, `sources` (list of document IDs), and `confidence`.
    - `tools_used` may include `document_search` and `document_reader`.

- Example 2 — Summarization
  - User: "Summarize the Service Agreement for Healthcare Partners." 
  - Flow: `classify_intent` → `summarization_agent` (retrieves contract, composes summary) → `update_memory`
  - Expected structured response: `SummarizationResponse` with `summary`, `key_points`, `document_ids`, and `original_length`.

- Example 3 — Calculation
  - User: "What's the total outstanding across invoices over $50,000?" 
  - Flow: `classify_intent` → `calculation_agent` (uses document search to find invoices then `calculator` tool) → `update_memory`
  - Expected structured response: `CalculationResponse` with `expression`, numeric `result`, and `explanation` describing steps.

- Example 4 — Unknown / fallback
  - User: "I have a question." (insufficient detail)
  - Flow: `classify_intent` may return `unknown` and the router falls back to `qa` by default. The assistant should then ask a clarifying question.

## Quick developer notes

- To trace execution start in `DocumentAssistant.start_session()` and call `process_message()` to run a message through the workflow. See [src/assistant.py](src/assistant.py).
- Tool implementations and logging live in [src/tools.py](src/tools.py); update `ToolLogger` or add tools via `get_all_tools()`.
- To change routing logic or add nodes, modify `create_workflow()` in [src/agent.py](src/agent.py) and add corresponding node functions and schema types.
