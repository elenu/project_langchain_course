# DocDacity — Document Assistant

DocDacity is a small multi-agent document assistant built with LangChain and LangGraph. It can classify user intent (Q&A, summarization, calculation), retrieve documents, call tools (e.g., calculator), and maintain conversation state across turns.

## Quick Start

1. Clone the repository and change into the project folder:

```bash
cd project_langchain_course
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY="sk-..."
```

5. Run the assistant:

```bash
.venv/bin/python main.py
```

## Usage

- When prompted, enter a `user ID` (press Enter for `demo_user`).
- At `Enter Message:` type natural language queries like:
	- "Summarize the contract documents"
	- "What's the total in invoice INV-001?"
	- "Calculate the sum of invoice totals"

Special commands while the assistant is running:
- `/help` — show available commands
- `/docs` — list available documents
- `/quit` — exit the assistant cleanly
- Ctrl+C — also exits (KeyboardInterrupt is handled)

Conversation sessions are saved to the `sessions/` directory so you can resume later.

## Project Structure

```
Project_Lesson_15/
├── main.py                # CLI entrypoint and interactive loop
├── requirements.txt       # Python dependencies
├── README.md.             # Descriptive file
└── src/                   # Source code
		├── assistant.py       # `DocumentAssistant` wrapper + process_message
		├── agent.py           # LangGraph workflow and agent nodes 
		├── prompts.py         # Prompt templates (system and chat prompts)
		├── retrieval.py       # Simulated retriever / documents
		├── schemas.py         # Pydantic models used for structured output
		├── tools.py           # LangChain tool implementations (calculator, etc)
		└── __init__.py
```

## Configuration & Common Issues

- API key in `.env` file and contains `OPENAI_API_KEY`.

- Insufficient/shared budget: this project sometimes points to a shared/test OpenAI endpoint that can run out of quota. This is what happened and I submitted the key from another for the correction of the project. If you see errors like "Insufficient budget" or "Exceeded budget", provide your own OpenAI key.

## Built With

* [LangChain](https://github.com/langchain-ai/langchain) — orchestration and prompts
* [LangGraph](https://github.com/langgraph-ai/langgraph) — graph-style agent workflows
* [OpenAI API] — LLM backend (via `langchain-openai` integration)

## License

MIT license.
