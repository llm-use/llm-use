# Quick Examples

## Setup

### Ollama (Local - FREE)
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:70b
ollama pull llama3.1:8b
```

### Cloud APIs
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

---

## Test 1: All Ollama (FREE)

```bash
python3 cli.py exec \
  --orchestrator ollama:llama3.1:70b \
  --worker ollama:llama3.1:8b \
  --task "Explain quantum computing"
```

**Expected:**
- Mode: single
- Cost: $0.00
- Time: ~3s

---

## Test 1b: Router (Cheap Model)

```bash
python3 cli.py exec \
  --router ollama:llama3.1:8b \
  --orchestrator anthropic:claude-3-7-sonnet-20250219 \
  --worker openai:gpt-4o-mini \
  --task "Explain TCP in 5 bullets"
```

---

## Test 1c: Router via llama.cpp Path

```bash
python3 cli.py exec \
  --router-path /path/to/your/router/model \
  --llama-cpp-url http://localhost:8080 \
  --orchestrator anthropic:claude-3-7-sonnet-20250219 \
  --worker openai:gpt-4o-mini \
  --task "Explain TCP in 5 bullets"
```

---

## Test 2: Parallel Task (Ollama)

```bash
python3 cli.py exec \
  --orchestrator ollama:llama3.1:70b \
  --worker ollama:llama3.1:8b \
  --task "Research AI from TechCrunch, Verge, Wired, ArsTechnica, Engadget"
```

**Expected:**
- Mode: parallel
- Workers: 5
- Cost: $0.00
- Time: ~18s

---

## Test 3: Hybrid (Cloud + Local)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

python3 cli.py exec \
  --orchestrator claude-3-7-sonnet-20250219 \
  --worker ollama:llama3.1:8b \
  --task "Compare Salesforce, HubSpot, Zoho"
```

**Expected:**
- Mode: parallel
- Workers: 3
- Cost: ~$0.006
- Time: ~10s

---

## Test 4: All Cloud (Anthropic)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

python3 cli.py exec \
  --orchestrator claude-3-7-sonnet-20250219 \
  --worker claude-3-5-haiku-20241022 \
  --task "Analyze sentiment for 5 reviews"
```

**Expected:**
- Mode: parallel
- Workers: 5
- Cost: ~$0.007
- Time: ~8s

---

## Test 5: Real Scraping (Workers)

```bash
python3 cli.py exec \
  --orchestrator openai:gpt-4o \
  --worker openai:gpt-4o-mini \
  --enable-scrape \
  --task "Find 3 sources about X and summarize them"
```

---

## Test 6: Playwright Scraping (Dynamic)

```bash
python3 cli.py exec \
  --orchestrator openai:gpt-4o \
  --worker openai:gpt-4o-mini \
  --enable-scrape \
  --scrape-backend playwright \
  --task "Find 3 sources about X and summarize them"
```

---

## TUI Chat

```bash
python3 cli.py chat \
  --orchestrator anthropic:claude-3-7-sonnet-20250219 \
  --worker ollama:llama3.1:8b
```

---

## MCP Server (PolyMCP)

```bash
python3 cli.py mcp \
  --orchestrator anthropic:claude-3-7-sonnet-20250219 \
  --worker ollama:llama3.1:8b \
  --host 127.0.0.1 \
  --port 8000
```

---

## Install Extras

```bash
python3 cli.py install --all
```

---

## Python API

```python
from llm_use import Orchestrator, ModelConfig

# All local
orch = Orchestrator(
    orchestrator=ModelConfig(name="llama3.1:70b", provider="ollama"),
    worker=ModelConfig(name="llama3.1:8b", provider="ollama")
)

result = orch.execute("What is AI?")
print(result['output'])

# Hybrid
orch = Orchestrator(
    orchestrator=ModelConfig(name="claude-3-7-sonnet-20250219", provider="anthropic"),
    worker=ModelConfig(name="llama3.1:8b", provider="ollama"),
    anthropic_key="sk-ant-..."
)

result = orch.execute("Compare 5 products")
print(f"Cost: ${result['cost']:.6f}")
```

---

## Stats

```bash
python3 cli.py stats
```

Shows:
- Total sessions
- Parallel vs single
- Total cost
- Recent sessions

---

## Router Reset

```bash
python3 cli.py router-reset
```

---

## Router Export / Import

```bash
python3 cli.py router-export --out router_examples.json
python3 cli.py router-import --in router_examples.json
```

---

## Cost Examples

**5-source research:**
- All Opus: $0.075
- Sonnet + Haiku: $0.007
- Sonnet + Ollama: $0.006
- Ollama + Ollama: $0.00

---

**Start testing!** 🚀
