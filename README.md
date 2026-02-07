<p align="center">
  <img src="llm-use.png" alt="llm-use_Logo" width="500"/>

[![License](https://img.shields.io/github/license/llm-use/llm-use)](https://github.com/llm-use/llm-use/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/llm-use/llm-use)](https://github.com/llm-use/llm-use/stargazers)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

</p>

Universal LLM orchestrator for running a “planner + workers + synthesis” flow across multiple providers (Anthropic, OpenAI, Ollama, llama.cpp). It chooses between single‑shot or parallel execution, aggregates costs, and stores session logs locally.

## Highlights
- Provider‑agnostic: mix cloud and local models.
- Cost tracking per run with a breakdown.
- Session history saved to `~/.llm-use/sessions`.
- Works fully offline with Ollama.
- Optional real web scraping + caching.
- Optional MCP server (via PolyMCP).
- TUI chat mode with live logs.

## Requirements
- Python 3.10+
- Optional provider SDKs: `anthropic`, `openai`
- `requests` (for Ollama HTTP calls)
- Ollama installed and running for local models
- Optional: `beautifulsoup4` for scraping
- Optional: `polymcp` + `uvicorn` for MCP server

## Installation
```bash
pip install requests

# Optional: cloud providers
pip install anthropic openai

# Optional: Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Optional: scraping
pip install beautifulsoup4

# Optional: MCP server
pip install polymcp uvicorn

# Optional: Playwright (dynamic scraping)
pip install playwright
playwright install

# Install as a package (editable)
pip install -e .
```

## Quick Start (Local Only)
```bash
ollama pull llama3.1:70b
ollama pull llama3.1:8b

python3 cli.py exec \
  --orchestrator ollama:llama3.1:70b \
  --worker ollama:llama3.1:8b \
  --task "Research AI from 5 sources"
```

## Quick Start (Hybrid)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
ollama pull llama3.1:8b

python3 cli.py exec \
  --orchestrator anthropic:claude-3-7-sonnet-20250219 \
  --worker ollama:llama3.1:8b \
  --task "Compare 10 products"
```

## TUI Chat
```bash
python3 cli.py chat \
  --orchestrator anthropic:claude-3-7-sonnet-20250219 \
  --worker ollama:llama3.1:8b
```

## MCP Server (PolyMCP)
```bash
python3 cli.py mcp \
  --orchestrator anthropic:claude-3-7-sonnet-20250219 \
  --worker ollama:llama3.1:8b \
  --host 127.0.0.1 \
  --port 8000
```

## Install Extras (Helper)
```bash
python3 cli.py install --all
```

## Usage
### Basic
```bash
python3 cli.py exec \
  --orchestrator <provider>:<model> \
  --worker <provider>:<model> \
  --task "your task"
```

### Router (Cheap Model to Skip Orchestration)
```bash
python3 cli.py exec \
  --router ollama:llama3.1:8b \
  --orchestrator anthropic:claude-3-7-sonnet-20250219 \
  --worker openai:gpt-4o-mini \
  --task "Explain TCP in 5 bullets"
```

### Router via llama.cpp Local Path
```bash
python3 cli.py exec \
  --router-path /path/to/your/router/model \
  --llama-cpp-url http://localhost:8080 \
  --orchestrator anthropic:claude-3-7-sonnet-20250219 \
  --worker openai:gpt-4o-mini \
  --task "Explain TCP in 5 bullets"
```

If the router model fails or is unavailable, it falls back to a heuristic router.

### Heuristic Router Rules (No Hardcoded Keywords)
By default the heuristic uses only length + URL signals. You can add your own patterns in `router_rules.json` (or set `LLM_USE_ROUTER_RULES` to a custom path).

### Learned Router (Lightweight ML)
The router also learns from past tasks by storing (task, mode) pairs and using cosine similarity on token vectors. This is local, cheap, and improves routing over time. Clear the cache to reset (`~/.llm-use/cache.sqlite`).

### Parallel Worker Control
```bash
python3 cli.py exec \
  --orchestrator anthropic:claude-3-7-sonnet-20250219 \
  --worker anthropic:claude-3-5-haiku-20241022 \
  --max-workers 8 \
  --task "Summarize 20 documents"
```

### Disable Cache
```bash
python3 cli.py exec \
  --orchestrator openai:gpt-4o \
  --worker openai:gpt-4o-mini \
  --no-cache \
  --task "Draft a brief memo"
```

### Real Scraping (Workers)
```bash
python3 cli.py exec \
  --orchestrator openai:gpt-4o \
  --worker openai:gpt-4o-mini \
  --enable-scrape \
  --task "Find 3 sources about X and summarize them"
```

### Dynamic Scraping (Playwright)
```bash
python3 cli.py exec \
  --orchestrator openai:gpt-4o \
  --worker openai:gpt-4o-mini \
  --enable-scrape \
  --scrape-backend playwright \
  --task "Find 3 sources about X and summarize them"
```

### Stats
```bash
python3 cli.py stats
```

### Router Reset (Clear Learned Memory)
```bash
python3 cli.py router-reset
```

### Router Export / Import
```bash
python3 cli.py router-export --out router_examples.json
python3 cli.py router-import --in router_examples.json
```

The export includes `created` timestamp and optional `confidence` if available.

## Python Package
```bash
pip install -e .
llm-use exec --orchestrator ollama:llama3.1:70b --worker ollama:llama3.1:8b --task "Hello"
```

## Concrete Examples (Agent Support)
These examples show how to use the orchestrator as the “brain” that delegates work to cheaper or local workers.

### Multi‑source research with final synthesis
```bash
python3 cli.py exec \
  --orchestrator anthropic:claude-3-7-sonnet-20250219 \
  --worker openai:gpt-4o-mini \
  --task "Collect 8 reliable sources on X and produce a pros/cons summary"
```

### Concurrent document analysis (agent brief)
```bash
python3 cli.py exec \
  --orchestrator openai:gpt-4o \
  --worker openai:gpt-4o-mini \
  --max-workers 6 \
  --task "Analyze 6 documents and return an executive brief with risks and opportunities"
```

### Privacy‑first local pipeline (offline agent)
```bash
python3 cli.py exec \
  --orchestrator ollama:qwen2.5:72b \
  --worker ollama:mistral:7b \
  --task "Extract requirements from internal notes and produce a checklist"
```

### Brainstorm + validation (creative + critic)
```bash
python3 cli.py exec \
  --orchestrator anthropic:claude-3-7-sonnet-20250219 \
  --worker ollama:llama3.1:8b \
  --task "Generate 20 ideas, then pick the top 5 with brief rationale"
```

## Best Practices for Agents
- Define the expected output format in the task (bullets, table, JSON).
- Avoid vague tasks: ask for decomposition and synthesis with clear criteria.
- Use cheaper workers for data gathering and a stronger orchestrator for synthesis.
- Set `--max-workers` based on rate limits and the number of subtasks.
- For sensitive data, prefer Ollama or isolated environments.

## File/CSV Examples (Prompt‑In‑File)
If your agent works on structured inputs, it helps to include the content directly in the prompt.

### Summarize a local file
```bash
python3 cli.py exec \
  --orchestrator openai:gpt-4o \
  --worker openai:gpt-4o-mini \
  --task "Summarize in 5 bullets the content of this file:\n\n$(cat notes.txt)"
```

### CSV analysis (schema + insights)
```bash
python3 cli.py exec \
  --orchestrator anthropic:claude-3-7-sonnet-20250219 \
  --worker anthropic:claude-3-5-haiku-20241022 \
  --task "Analyze the CSV below, describe the schema and 3 insights:\n\n$(cat data.csv)"
```

### JSON output for agent pipelines
```bash
python3 cli.py exec \
  --orchestrator ollama:llama3.1:70b \
  --worker ollama:llama3.1:8b \
  --task "Extract requirements in JSON with keys: title, priority, rationale:\n\n$(cat requirements.md)"
```

## Providers and Models
The following model names are recognized out of the box. You can also pass custom models with `provider:model`.

### Anthropic
- `claude-3-5-haiku-20241022`
- `claude-3-7-sonnet-20250219`
- `claude-4-opus-20250514`

### OpenAI
- `gpt-4o-mini`
- `gpt-4o`
- `o1`

### Ollama
- `llama3.1:70b`
- `llama3.1:8b`
- `qwen2.5:72b`
- `mistral:7b`

### llama.cpp (OpenAI-compatible server)
Use `llama_cpp:<model>` with a llama.cpp server that exposes `/v1/chat/completions`.

## Python API
```python
from llm_use import Orchestrator, ModelConfig

orch = Orchestrator(
    orchestrator=ModelConfig(name="llama3.1:70b", provider="ollama"),
    worker=ModelConfig(name="llama3.1:8b", provider="ollama")
)

result = orch.execute("Your task")
print(f"Cost: ${result['cost']:.6f}")
print(result["output"])
```

## Cost Notes
Costs are estimated using provider list prices per million tokens and token counts returned by the SDKs. For Ollama, cost is zero by default. Token usage for Ollama is estimated from word counts.

## Troubleshooting
### Ollama not found
```bash
ollama serve
ollama list
```

### Missing API keys
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

## Testing
```bash
pip install pytest
pytest
```

## License
MIT
