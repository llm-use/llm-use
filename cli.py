#!/usr/bin/env python3
"""
llm-use v2.0 - Universal Orchestrator
Supports: Anthropic, OpenAI, Ollama + custom models

Author: Vincenzo
"""

import os
import sys
import json
import time
import hashlib
import logging
import sqlite3
import threading
import subprocess
import requests
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import traceback

# Timeouts (seconds)
WORKER_GLOBAL_TIMEOUT = 120
WORKER_CALL_TIMEOUT = 30
SCRAPE_TIMEOUT = 15

# Logging
log_dir = Path.home() / ".llm-use"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "orchestrator.log")]
)
logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

try:
    from polymcp import expose_tools_http
    HAS_POLYMCP = True
except ImportError:
    HAS_POLYMCP = False

# Default models
DEFAULT_MODELS = {
    "claude-3-5-haiku-20241022": {"provider": "anthropic", "cost_in": 0.25, "cost_out": 1.25},
    "claude-3-7-sonnet-20250219": {"provider": "anthropic", "cost_in": 3.0, "cost_out": 15.0},
    "claude-4-opus-20250514": {"provider": "anthropic", "cost_in": 15.0, "cost_out": 75.0},
    "gpt-4o-mini": {"provider": "openai", "cost_in": 0.15, "cost_out": 0.60},
    "gpt-4o": {"provider": "openai", "cost_in": 2.50, "cost_out": 10.0},
    "o1": {"provider": "openai", "cost_in": 15.0, "cost_out": 60.0},
    "llama3.1:70b": {"provider": "ollama", "cost_in": 0.0, "cost_out": 0.0},
    "llama3.1:8b": {"provider": "ollama", "cost_in": 0.0, "cost_out": 0.0},
    "qwen2.5:72b": {"provider": "ollama", "cost_in": 0.0, "cost_out": 0.0},
    "mistral:7b": {"provider": "ollama", "cost_in": 0.0, "cost_out": 0.0}
}

URL_RE = re.compile(r"https?://[^\s\)\]\}\>\"\']+")
DEFAULT_ROUTER_RULES_PATH = str(Path.home() / ".llm-use" / "router_rules.json")

ORCHESTRATOR_PROMPT = """Analyze this task and decide how to execute it.

If parallelizable, return:
{{
  "mode": "parallel",
  "subtasks": [{{"id": 1, "task": "..."}}, {{"id": 2, "task": "..."}}],
  "combine_strategy": "..."
}}

If single, return:
{{
  "mode": "single",
  "response": "complete answer"
}}

Max {max_workers} subtasks. Task: {task}

JSON:"""

ROUTER_PROMPT = """Classify the task complexity and decide whether to use a full orchestrator or a direct single call.

Return JSON:
{
  "route": "simple" | "full",
  "reason": "short reason",
  "confidence": 0.0-1.0
}

Simple = short, low-risk, no research, no multi-step decomposition.
Full = needs decomposition, multi-source, or complex reasoning.

Task: {task}

JSON:"""

SYNTHESIS_PROMPT = """Combine worker results.

Task: {task}
Strategy: {strategy}
Results:
{results}

Final answer:"""

def parse_orchestrator_json(text: str) -> Dict:
    resp = text.strip()
    if resp.startswith("```"):
        parts = resp.split("```")
        if len(parts) > 1:
            resp = parts[1]
            if resp.startswith("json"):
                resp = resp[4:]
            resp = resp.strip()
    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        pass
    # Extract first JSON object by balancing braces.
    start = resp.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", resp, 0)
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(resp)):
        ch = resp[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
        else:
            if ch == "\"":
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = resp[start:i+1]
                    return json.loads(candidate)
    raise json.JSONDecodeError("Unterminated JSON object", resp, start)

class ExecMode(Enum):
    SINGLE = "single"
    PARALLEL = "parallel"

@dataclass
class ModelConfig:
    name: str
    provider: str
    cost_in: float = 0.0
    cost_out: float = 0.0
    max_tokens: int = 4096
    timeout: int = 60

@dataclass
class Call:
    id: str
    model: str
    provider: str
    prompt: str
    response: str
    tokens_in: int
    tokens_out: int
    cost: float
    duration: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None

@dataclass
class Session:
    id: str
    task: str
    mode: ExecMode
    orchestrator_call: Call
    worker_calls: List[Call] = field(default_factory=list)
    synthesis_call: Optional[Call] = None
    output: str = ""
    total_cost: float = 0.0
    total_duration: float = 0.0
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    completed: Optional[str] = None
    error: Optional[str] = None

class OllamaProvider:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        logger.info(f"✅ Ollama: {self.base_url}")
    
    def call(self, model: str, prompt: str, max_tokens: int, temperature: float, timeout: int) -> Tuple[str, int, int]:
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens}}
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = data.get("response", "")
        tokens_in = int(len(prompt.split()) * 1.3)
        tokens_out = int(len(text.split()) * 1.3)
        return text, tokens_in, tokens_out

class LlamaCppProvider:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        logger.info(f"✅ llama.cpp: {self.base_url}")

    def call(self, model: str, prompt: str, max_tokens: int, temperature: float, timeout: int) -> Tuple[str, int, int]:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens") or len(prompt.split()) * 1.3)
        tokens_out = int(usage.get("completion_tokens") or len(text.split()) * 1.3)
        return text, tokens_in, tokens_out

class AnthropicProvider:
    def __init__(self, api_key: str):
        if not HAS_ANTHROPIC:
            raise ValueError("pip install anthropic")
        self.client = Anthropic(api_key=api_key)
        logger.info("✅ Anthropic ready")
    
    def call(self, model: str, prompt: str, max_tokens: int, temperature: float, timeout: int) -> Tuple[str, int, int]:
        r = self.client.messages.create(model=model, max_tokens=max_tokens, temperature=temperature, messages=[{"role": "user", "content": prompt}], timeout=timeout)
        return r.content[0].text, r.usage.input_tokens, r.usage.output_tokens

class OpenAIProvider:
    def __init__(self, api_key: str):
        if not HAS_OPENAI:
            raise ValueError("pip install openai")
        self.client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI ready")
    
    def call(self, model: str, prompt: str, max_tokens: int, temperature: float, timeout: int) -> Tuple[str, int, int]:
        r = self.client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature, timeout=timeout)
        return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens

class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                provider TEXT,
                model TEXT,
                prompt_hash TEXT,
                response TEXT,
                tokens_in INTEGER,
                tokens_out INTEGER,
                created REAL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scrape_cache (
                key TEXT PRIMARY KEY,
                url TEXT,
                content TEXT,
                created REAL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS router_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT,
                mode TEXT,
                created REAL
            )
            """
        )
        try:
            self._conn.execute("ALTER TABLE router_examples ADD COLUMN confidence REAL")
        except Exception:
            pass
        self._conn.commit()

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get_llm(self, key: str) -> Optional[Tuple[str, int, int]]:
        with self._lock:
            row = self._conn.execute("SELECT response, tokens_in, tokens_out FROM llm_cache WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        return row[0], int(row[1]), int(row[2])

    def set_llm(self, key: str, provider: str, model: str, prompt: str, response: str, tokens_in: int, tokens_out: int):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, provider, model, prompt_hash, response, tokens_in, tokens_out, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (key, provider, model, self._hash(prompt), response, tokens_in, tokens_out, time.time())
            )
            self._conn.commit()

    def get_scrape(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT content FROM scrape_cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_scrape(self, key: str, url: str, content: str):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO scrape_cache (key, url, content, created) VALUES (?, ?, ?, ?)",
                (key, url, content, time.time())
            )
            self._conn.commit()

    def add_router_example(self, task: str, mode: str, confidence: Optional[float] = None, max_rows: int = 500):
        with self._lock:
            self._conn.execute(
                "INSERT INTO router_examples (task, mode, created, confidence) VALUES (?, ?, ?, ?)",
                (task, mode, time.time(), confidence)
            )
            self._conn.execute(
                "DELETE FROM router_examples WHERE id NOT IN (SELECT id FROM router_examples ORDER BY id DESC LIMIT ?)",
                (max_rows,)
            )
            self._conn.commit()

    def get_router_examples(self, limit: int = 200) -> List[Tuple[str, str, float, Optional[float]]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT task, mode, created, confidence FROM router_examples ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [(r[0], r[1], float(r[2]), (float(r[3]) if r[3] is not None else None)) for r in rows]

class API:
    def __init__(self, anthropic_key: Optional[str], openai_key: Optional[str], ollama_url: str, llama_cpp_url: str, cache: Optional[Cache] = None, enable_cache: bool = True):
        self.providers = {}
        self.cache = cache
        self.enable_cache = enable_cache
        if anthropic_key:
            try:
                self.providers["anthropic"] = AnthropicProvider(anthropic_key)
            except: pass
        if openai_key:
            try:
                self.providers["openai"] = OpenAIProvider(openai_key)
            except: pass
        try:
            self.providers["ollama"] = OllamaProvider(ollama_url)
        except: pass
        try:
            self.providers["llama_cpp"] = LlamaCppProvider(llama_cpp_url)
        except: pass
        if not self.providers:
            raise ValueError("No providers! Install: pip install anthropic openai")
        logger.info(f"Providers: {list(self.providers.keys())}")
    
    def call(self, config: ModelConfig, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, retries: int = 3) -> Call:
        call_id = hashlib.md5(f"{config.name}{time.time()}".encode()).hexdigest()[:8]
        cache_key = None
        if self.cache and self.enable_cache:
            cache_key = hashlib.md5(f"{config.provider}:{config.name}:{max_tokens}:{temperature}:{prompt}".encode()).hexdigest()
            cached = self.cache.get_llm(cache_key)
            if cached:
                text, tin, tout = cached
                return Call(call_id, config.name, config.provider, prompt, text, tin, tout, 0.0, 0.0)
        for attempt in range(retries):
            try:
                t0 = time.time()
                provider = self.providers.get(config.provider)
                if not provider:
                    raise ValueError(f"Provider {config.provider} not available")
                text, tin, tout = provider.call(config.name, prompt, max_tokens, temperature, config.timeout)
                duration = time.time() - t0
                cost = (tin / 1_000_000) * config.cost_in + (tout / 1_000_000) * config.cost_out
                if self.cache and self.enable_cache:
                    self.cache.set_llm(cache_key, config.provider, config.name, prompt, text, tin, tout)
                return Call(call_id, config.name, config.provider, prompt, text, tin, tout, cost, duration)
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{retries}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return Call(call_id, config.name, config.provider, prompt, "", 0, 0, 0.0, 0.0, error=str(e))

class SessionManager:
    def __init__(self):
        self.path = Path.home() / ".llm-use" / "sessions"
        self.path.mkdir(parents=True, exist_ok=True)
        self.current: Optional[Session] = None
    
    def create(self, task: str, mode: ExecMode, orch_call: Call) -> Session:
        sid = hashlib.md5(f"{task}{time.time()}".encode()).hexdigest()[:10]
        self.current = Session(sid, task, mode, orch_call, total_cost=orch_call.cost, total_duration=orch_call.duration)
        logger.info(f"📝 Session {sid}: {mode.value}")
        return self.current
    
    def add_worker(self, call: Call):
        if self.current:
            self.current.worker_calls.append(call)
            self.current.total_cost += call.cost
            self.current.total_duration += call.duration
    
    def add_synthesis(self, call: Call):
        if self.current:
            self.current.synthesis_call = call
            self.current.total_cost += call.cost
            self.current.total_duration += call.duration
    
    def complete(self, output: str, error: Optional[str] = None):
        if not self.current:
            return
        self.current.output = output
        self.current.completed = datetime.now().isoformat()
        self.current.error = error
        try:
            file = self.path / f"{self.current.id}.json"
            data = asdict(self.current)
            data["mode"] = self.current.mode.value
            with open(file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Save failed: {e}")
        logger.info(f"✅ Session {self.current.id}: ${self.current.total_cost:.6f}")
    
    def load_recent(self, n: int = 10) -> List[Session]:
        sessions = []
        for f in sorted(self.path.glob("*.json"), reverse=True)[:n]:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    data["mode"] = ExecMode(data["mode"])
                    data["orchestrator_call"] = Call(**data["orchestrator_call"])
                    if data.get("worker_calls"):
                        data["worker_calls"] = [Call(**c) for c in data["worker_calls"]]
                    if data.get("synthesis_call"):
                        data["synthesis_call"] = Call(**data["synthesis_call"])
                    sessions.append(Session(**data))
            except: pass
        return sessions

class Orchestrator:
    def __init__(self, orchestrator: ModelConfig, worker: ModelConfig, max_workers: int = 10, anthropic_key: Optional[str] = None, openai_key: Optional[str] = None, ollama_url: str = "http://localhost:11434", llama_cpp_url: str = "http://localhost:8080", enable_cache: bool = True, enable_scrape: bool = False, max_scrape_urls: int = 3, scrape_backend: str = "requests", event_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None, router: Optional[ModelConfig] = None):
        cache = Cache(Path.home() / ".llm-use" / "cache.sqlite")
        self.api = API(anthropic_key, openai_key, ollama_url, llama_cpp_url, cache=cache, enable_cache=enable_cache)
        self.sessions = SessionManager()
        self.orch_config = orchestrator
        self.worker_config = worker
        self.router_config = router
        self.max_workers = max_workers
        self.enable_scrape = enable_scrape
        self.max_scrape_urls = max_scrape_urls
        self.scrape_backend = scrape_backend
        self.event_cb = event_cb
        logger.info(f"🎭 Orch: {orchestrator.provider}:{orchestrator.name}")
        logger.info(f"🤖 Worker: {worker.provider}:{worker.name}")
        if router:
            logger.info(f"🧭 Router: {router.provider}:{router.name}")

    def _emit(self, name: str, payload: Dict[str, Any]):
        if self.event_cb:
            try:
                self.event_cb(name, payload)
            except Exception:
                pass

    def _record_router_example(self, task: str, mode: str, confidence: Optional[float] = None):
        cache = self.api.cache
        if not cache:
            return
        try:
            cache.add_router_example(task, mode, confidence=confidence)
        except Exception:
            pass
    
    def execute(self, task: str) -> Dict[str, Any]:
        logger.info(f"📥 Task: {task[:100]}...")
        try:
            logger.info("🎯 Analyzing...")
            self._emit("analyze_start", {"task": task})
            if self.router_config:
                routed = self._route(task)
                if routed["route"] == "simple":
            return self._execute_simple(task, routed["_call"], confidence=routed.get("confidence"))
            decision = self._get_decision(task)
            if decision["mode"] == "single":
                return self._execute_single(task, decision)
            else:
                return self._execute_parallel(task, decision)
        except Exception as e:
            logger.error(f"Failed: {e}\n{traceback.format_exc()}")
            raise
    
    def _get_decision(self, task: str) -> Dict:
        prompt = ORCHESTRATOR_PROMPT.format(task=task, max_workers=self.max_workers)
        call = self.api.call(self.orch_config, prompt, max_tokens=2048, temperature=0.3)
        if call.error:
            raise Exception(f"Decision failed: {call.error}")
        try:
            decision = parse_orchestrator_json(call.response)
            decision["_call"] = call
            logger.info(f"📊 Mode: {decision['mode']}")
            return decision
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {call.response}")
            raise Exception(f"Invalid JSON: {e}")

    def _route(self, task: str) -> Dict[str, Any]:
        prompt = ROUTER_PROMPT.format(task=task)
        call = self.api.call(self.router_config, prompt, max_tokens=256, temperature=0.0)
        if call.error:
            logger.warning(f"Router failed, falling back to heuristic: {call.error}")
            return self._route_heuristic(task)
        try:
            decision = parse_orchestrator_json(call.response)
            decision["_call"] = call
            route = decision.get("route", "full")
            logger.info(f"🧭 Route: {route}")
            return decision
        except json.JSONDecodeError as e:
            logger.error(f"Invalid router JSON: {call.response}")
            return self._route_heuristic(task)

    def _load_router_rules(self) -> Dict[str, Any]:
        path = os.getenv("LLM_USE_ROUTER_RULES", DEFAULT_ROUTER_RULES_PATH)
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _match_patterns(self, patterns: List[str], text: str) -> int:
        hits = 0
        for p in patterns:
            try:
                if re.search(p, text, flags=re.IGNORECASE):
                    hits += 1
            except re.error:
                continue
        return hits

    def _route_heuristic(self, task: str) -> Dict[str, Any]:
        t = task.strip()
        word_count = len(t.split())
        has_url = bool(URL_RE.search(t))
        rules = self._load_router_rules()
        full_patterns = rules.get("full_patterns", [])
        simple_patterns = rules.get("simple_patterns", [])
        full_hits = self._match_patterns(full_patterns, t)
        simple_hits = self._match_patterns(simple_patterns, t)

        learned = self._route_learned(task)
        if learned:
            return learned

        route = "simple"
        reason = "short_simple_task"
        if has_url or word_count > 140 or full_hits > 0:
            route = "full"
            reason = "complex_or_research"
        elif word_count > 60 and simple_hits == 0:
            route = "full"
            reason = "long_uncertain"

        return {
            "route": route,
            "reason": reason,
            "confidence": 0.65 if route == "simple" else 0.55,
            "_call": Call("heuristic", "router", "heuristic", task, route, 0, 0, 0.0, 0.0)
        }

    def _route_learned(self, task: str) -> Optional[Dict[str, Any]]:
        cache = self.api.cache
        if not cache:
            return None
        examples = cache.get_router_examples(limit=200)
        if not examples:
            return None
        vec = self._tf_vector(task)
        if not vec:
            return None
        best = (0.0, None)
        for ex_task, ex_mode, _created, _conf in examples:
            ex_vec = self._tf_vector(ex_task)
            if not ex_vec:
                continue
            sim = self._cosine_sim(vec, ex_vec)
            if sim > best[0]:
                best = (sim, ex_mode)
        if best[0] >= 0.35 and best[1] in ("single", "parallel"):
            route = "simple" if best[1] == "single" else "full"
            return {
                "route": route,
                "reason": "learned_similarity",
                "confidence": min(0.85, 0.5 + best[0]),
                "_call": Call("learned", "router", "learned", task, route, 0, 0, 0.0, 0.0)
            }
        return None

    def _tf_vector(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r"[a-zA-Z0-9]{2,}", text.lower())
        if not tokens:
            return {}
        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total = float(sum(counts.values()))
        return {k: v / total for k, v in counts.items()}

    def _cosine_sim(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = 0.0
        for k, v in a.items():
            if k in b:
                dot += v * b[k]
        na = sum(v * v for v in a.values()) ** 0.5
        nb = sum(v * v for v in b.values()) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def _execute_simple(self, task: str, router_call: Call, confidence: Optional[float] = None) -> Dict[str, Any]:
        logger.info("⚡ ROUTED SIMPLE")
        self._emit("single_start", {"routed": True})
        session = self.sessions.create(task, ExecMode.SINGLE, router_call)
        call = self.api.call(self.worker_config, task, 2048, 0.5)
        self.sessions.add_worker(call)
        output = call.response if not call.error else f"Error: {call.error}"
        self.sessions.complete(output, error=call.error)
        self._record_router_example(task, "single", confidence=confidence)
        self._emit("single_done", {"cost": session.total_cost, "duration": session.total_duration})
        return {"output": output, "mode": "single", "model": f"{self.worker_config.provider}:{self.worker_config.name}", "cost": session.total_cost, "duration": session.total_duration, "session_id": session.id, "routed": True}
    
    def _execute_single(self, task: str, decision: Dict) -> Dict:
        logger.info("💬 SINGLE mode")
        self._emit("single_start", {})
        orch_call = decision["_call"]
        output = decision["response"]
        session = self.sessions.create(task, ExecMode.SINGLE, orch_call)
        if getattr(self, "enable_scrape", False):
            grounded = self._ground_single(task, output)
            if grounded:
                self.sessions.add_synthesis(grounded["_call"])
                output = grounded["output"]
        self.sessions.complete(output)
        self._record_router_example(task, "single")
        self._emit("single_done", {"cost": session.total_cost, "duration": session.total_duration})
        return {"output": output, "mode": "single", "model": f"{self.orch_config.provider}:{self.orch_config.name}", "cost": session.total_cost, "duration": session.total_duration, "session_id": session.id}
    
    def _execute_parallel(self, task: str, decision: Dict) -> Dict:
        subtasks = decision["subtasks"]
        combine_strategy = decision["combine_strategy"]
        orch_call = decision["_call"]
        logger.info(f"🎭 PARALLEL ({len(subtasks)} workers)")
        self._emit("parallel_start", {"workers": len(subtasks)})
        session = self.sessions.create(task, ExecMode.PARALLEL, orch_call)
        try:
            logger.info("🚀 Spawning...")
            worker_results = self._spawn_workers(subtasks)
            succeeded = len([r for r in worker_results if not r.error])
            logger.info(f"✅ Done: {succeeded}/{len(worker_results)}")
            self._emit("workers_done", {"ok": succeeded, "total": len(worker_results)})
            if succeeded == 0:
                raise Exception("All workers failed")
            logger.info("🔧 Synthesizing...")
            final = self._synthesize(task, combine_strategy, worker_results)
            self.sessions.complete(final)
            self._record_router_example(task, "parallel")
            self._emit("parallel_done", {"cost": session.total_cost, "duration": session.total_duration})
            return {"output": final, "mode": "parallel", "orchestrator_model": f"{self.orch_config.provider}:{self.orch_config.name}", "worker_model": f"{self.worker_config.provider}:{self.worker_config.name}", "workers_spawned": len(subtasks), "workers_succeeded": succeeded, "cost": session.total_cost, "breakdown": {"orchestrator": orch_call.cost, "workers": sum(c.cost for c in session.worker_calls), "synthesis": session.synthesis_call.cost if session.synthesis_call else 0}, "duration": session.total_duration, "session_id": session.id}
        except Exception as e:
            logger.error(f"Parallel failed: {e}")
            self.sessions.complete("", error=str(e))
            self._emit("parallel_error", {"error": str(e)})
            raise
    
    def _spawn_workers(self, subtasks: List[Dict]) -> List[Call]:
        results = []
        with ThreadPoolExecutor(max_workers=min(len(subtasks), self.max_workers)) as executor:
            future_to_subtask = {executor.submit(self._run_worker, st): st for st in subtasks}
            futures = list(future_to_subtask.keys())
            try:
                for future in as_completed(future_to_subtask, timeout=WORKER_GLOBAL_TIMEOUT):
                    st = future_to_subtask[future]
                    try:
                        self._emit("worker_wait", {"id": st["id"]})
                        call = future.result(timeout=WORKER_CALL_TIMEOUT)
                        self.sessions.add_worker(call)
                        results.append(call)
                        logger.info(f"{'✓' if not call.error else '✗'} Worker {st['id']}")
                        self._emit("worker_done", {"id": st["id"], "ok": call.error is None})
                    except TimeoutError:
                        logger.error(f"✗ Worker {st['id']} timeout")
                        results.append(Call(f"err{st['id']}", self.worker_config.name, self.worker_config.provider, st["task"], "", 0, 0, 0.0, 0.0, error="timeout"))
                        self._emit("worker_done", {"id": st["id"], "ok": False})
                    except Exception as e:
                        logger.error(f"✗ Worker {st['id']}: {e}")
                        results.append(Call(f"err{st['id']}", self.worker_config.name, self.worker_config.provider, st["task"], "", 0, 0, 0.0, 0.0, error=str(e)))
                        self._emit("worker_done", {"id": st["id"], "ok": False})
            except TimeoutError:
                logger.error("✗ Global worker timeout")
                for future in futures:
                    if not future.done():
                        future.cancel()
                        st = future_to_subtask[future]
                        results.append(Call(f"err{st['id']}", self.worker_config.name, self.worker_config.provider, st["task"], "", 0, 0, 0.0, 0.0, error="timeout"))
        return results

    def _run_worker(self, subtask: Dict) -> Call:
        task = subtask["task"]
        enable_scrape = getattr(self, "enable_scrape", False)
        max_scrape_urls = getattr(self, "max_scrape_urls", 3)
        if enable_scrape:
            task = task + "\n\nIf you need sources, list up to {n} URLs prefixed with 'URL:'.".format(n=max_scrape_urls)
        call = self.api.call(self.worker_config, task, 1024, 0.5)
        if not enable_scrape or call.error:
            return call
        urls = self._extract_urls(call.response)
        if not urls:
            return call
        scraped = self._scrape_urls(urls)
        if not scraped:
            return call
        followup_prompt = """You previously replied with possible sources.
Use the scraped content below to provide a final, grounded answer.

Original task:
{task}

Scraped content:
{content}

Final answer:""".format(task=subtask["task"], content=scraped)
        followup = self.api.call(self.worker_config, followup_prompt, 1024, 0.4)
        if not followup.error and followup.response:
            return followup
        return call

    def _ground_single(self, task: str, output: str) -> Optional[Dict[str, Any]]:
        urls = self._extract_urls(output)
        if not urls:
            return None
        scraped = self._scrape_urls(urls)
        if not scraped:
            return None
        prompt = """You provided an answer that referenced sources.\nUse the scraped content below to verify and improve the answer.\n\nOriginal task:\n{task}\n\nScraped content:\n{content}\n\nImproved answer:""".format(task=task, content=scraped)
        call = self.api.call(self.orch_config, prompt, max_tokens=2048, temperature=0.4)
        if call.error or not call.response:
            return None
        return {"output": call.response, "_call": call}

    def _extract_urls(self, text: str) -> List[str]:
        urls = []
        for line in text.splitlines():
            if line.strip().lower().startswith("url:"):
                url = line.split(":", 1)[1].strip()
                if url.startswith("http"):
                    urls.append(url)
        if not urls:
            urls = URL_RE.findall(text)
        seen = set()
        out = []
        max_scrape_urls = getattr(self, "max_scrape_urls", 3)
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
            if len(out) >= max_scrape_urls:
                break
        return out

    def _scrape_urls(self, urls: List[str]) -> str:
        if self.scrape_backend == "playwright":
            if not HAS_PLAYWRIGHT:
                return ""
            return self._scrape_urls_playwright(urls)
        if not HAS_BS4:
            return ""
        parts = []
        cache = self.api.cache
        for url in urls:
            key = hashlib.md5(url.encode()).hexdigest()
            cached = cache.get_scrape(key) if cache else None
            if cached:
                parts.append(f"SOURCE: {url}\n{cached}")
                continue
            try:
                r = requests.get(url, timeout=SCRAPE_TIMEOUT, headers={"User-Agent": "llm-use/2.0"})
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
                    tag.decompose()
                text = " ".join(soup.get_text(separator=" ").split())
                text = text[:4000]
                if cache:
                    cache.set_scrape(key, url, text)
                parts.append(f"SOURCE: {url}\n{text}")
            except Exception as e:
                logger.warning(f"Scrape failed {url}: {e}")
        return "\n\n".join(parts)

    def _scrape_urls_playwright(self, urls: List[str]) -> str:
        parts = []
        cache = self.api.cache
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            for url in urls:
                key = hashlib.md5(url.encode()).hexdigest()
                cached = cache.get_scrape(key) if cache else None
                if cached:
                    parts.append(f"SOURCE: {url}\n{cached}")
                    continue
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=SCRAPE_TIMEOUT * 1000)
                    html = page.content()
                    if not HAS_BS4:
                        text = " ".join(html.split())
                    else:
                        soup = BeautifulSoup(html, "html.parser")
                        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
                            tag.decompose()
                        text = " ".join(soup.get_text(separator=" ").split())
                    text = text[:4000]
                    if cache:
                        cache.set_scrape(key, url, text)
                    parts.append(f"SOURCE: {url}\n{text}")
                except Exception as e:
                    logger.warning(f"Playwright scrape failed {url}: {e}")
            browser.close()
        return "\n\n".join(parts)
    
    def _synthesize(self, task: str, strategy: str, worker_results: List[Call]) -> str:
        results_text = "\n\n".join([f"Worker {i+1}:\n{c.response}" for i, c in enumerate(worker_results) if not c.error and c.response])
        if not results_text:
            raise Exception("No successful results")
        prompt = SYNTHESIS_PROMPT.format(task=task, strategy=strategy, results=results_text)
        call = self.api.call(self.orch_config, prompt, max_tokens=2048, temperature=0.5)
        self.sessions.add_synthesis(call)
        if call.error:
            raise Exception(f"Synthesis failed: {call.error}")
        return call.response
    
    def stats(self):
        sessions = self.sessions.load_recent(100)
        if not sessions:
            print("\nNo sessions\n")
            return
        total_cost = sum(s.total_cost for s in sessions)
        parallel = [s for s in sessions if s.mode == ExecMode.PARALLEL]
        print(f"\n{'='*60}\nSTATISTICS\n{'='*60}")
        print(f"Total: {len(sessions)}")
        print(f"Parallel: {len(parallel)} ({len(parallel)/len(sessions)*100:.0f}%)")
        print(f"Cost: ${total_cost:.4f}")
        print(f"Avg: ${total_cost/len(sessions):.4f}")
        print(f"{'='*60}\n")
        print("Recent:")
        for s in sessions[:5]:
            icon = "🎭" if s.mode == ExecMode.PARALLEL else "💬"
            workers = f"({len(s.worker_calls)} workers)" if s.mode == ExecMode.PARALLEL else ""
            print(f"{icon} {s.id} | {s.task[:40]:<40} | ${s.total_cost:.4f} {workers}")
        print()

def print_stats():
    sessions = SessionManager().load_recent(100)
    if not sessions:
        print("\nNo sessions\n")
        return
    total_cost = sum(s.total_cost for s in sessions)
    parallel = [s for s in sessions if s.mode == ExecMode.PARALLEL]
    print(f"\n{'='*60}\nSTATISTICS\n{'='*60}")
    print(f"Total: {len(sessions)}")
    print(f"Parallel: {len(parallel)} ({len(parallel)/len(sessions)*100:.0f}%)")
    print(f"Cost: ${total_cost:.4f}")
    print(f"Avg: ${total_cost/len(sessions):.4f}")
    print(f"{'='*60}\n")
    print("Recent:")
    for s in sessions[:5]:
        icon = "🎭" if s.mode == ExecMode.PARALLEL else "💬"
        workers = f"({len(s.worker_calls)} workers)" if s.mode == ExecMode.PARALLEL else ""
        print(f"{icon} {s.id} | {s.task[:40]:<40} | ${s.total_cost:.4f} {workers}")
    print()

def stats_snapshot() -> Dict[str, Any]:
    sessions = SessionManager().load_recent(100)
    if not sessions:
        return {"total": 0, "parallel": 0, "cost": 0.0, "avg_cost": 0.0, "recent": []}
    total_cost = sum(s.total_cost for s in sessions)
    parallel = [s for s in sessions if s.mode == ExecMode.PARALLEL]
    recent = []
    for s in sessions[:5]:
        recent.append({
            "id": s.id,
            "task": s.task,
            "mode": s.mode.value,
            "cost": s.total_cost,
            "workers": len(s.worker_calls)
        })
    return {
        "total": len(sessions),
        "parallel": len(parallel),
        "cost": total_cost,
        "avg_cost": total_cost / len(sessions),
        "recent": recent
    }

def simple_scrape(url: str, cache: Optional[Cache] = None, backend: str = "requests") -> str:
    if backend == "playwright":
        if not HAS_PLAYWRIGHT:
            return ""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=SCRAPE_TIMEOUT * 1000)
                html = page.content()
                if not HAS_BS4:
                    text = " ".join(html.split())
                else:
                    soup = BeautifulSoup(html, "html.parser")
                    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
                        tag.decompose()
                    text = " ".join(soup.get_text(separator=" ").split())
                text = text[:4000]
                key = hashlib.md5(url.encode()).hexdigest()
                if cache:
                    cache.set_scrape(key, url, text)
                return text
            except Exception as e:
                logger.warning(f"Playwright scrape failed {url}: {e}")
                return ""
            finally:
                browser.close()
    if not HAS_BS4:
        return ""
    key = hashlib.md5(url.encode()).hexdigest()
    if cache:
        cached = cache.get_scrape(key)
        if cached:
            return cached
    try:
        r = requests.get(url, timeout=SCRAPE_TIMEOUT, headers={"User-Agent": "llm-use/2.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        text = text[:4000]
        if cache:
            cache.set_scrape(key, url, text)
        return text
    except Exception as e:
        logger.warning(f"Scrape failed {url}: {e}")
        return ""

def build_chat_prompt(history: List[Tuple[str, str]], max_turns: int = 6) -> str:
    recent = history[-(max_turns * 2):]
    lines = []
    for role, text in recent:
        lines.append(f"{role.upper()}: {text}")
    lines.append("ASSISTANT:")
    return "You are in a chat with a user. Respond naturally and helpfully.\n\nConversation:\n" + "\n".join(lines)

def run_chat_tui(orch: Orchestrator):
    import curses
    import textwrap
    import queue
    from collections import deque

    log_lines = deque(maxlen=200)
    log_lock = threading.Lock()

    removed_handlers = []
    for h in list(logger.handlers):
        if type(h) is logging.StreamHandler:
            logger.removeHandler(h)
            removed_handlers.append(h)

    class UILogHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)
            with log_lock:
                log_lines.append(msg)

    handler = UILogHandler()
    handler.setFormatter(logging.Formatter("%H:%M:%S [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    history: List[Tuple[str, str]] = []
    input_buf = ""
    status = "Ready"
    busy = False
    q: "queue.Queue[Tuple[str, str]]" = queue.Queue()
    state = {
        "mode": "",
        "workers_total": 0,
        "workers_done": 0,
        "workers_ok": 0,
        "cost": 0.0,
        "duration": 0.0
    }

    def on_event(name: str, payload: Dict[str, Any]):
        if name == "parallel_start":
            state["mode"] = "parallel"
            state["workers_total"] = int(payload.get("workers", 0))
            state["workers_done"] = 0
            state["workers_ok"] = 0
        elif name == "worker_done":
            state["workers_done"] += 1
            if payload.get("ok"):
                state["workers_ok"] += 1
        elif name == "parallel_done":
            state["cost"] = float(payload.get("cost", 0.0))
            state["duration"] = float(payload.get("duration", 0.0))
        elif name == "single_start":
            state["mode"] = "single"
            state["workers_total"] = 0
            state["workers_done"] = 0
            state["workers_ok"] = 0
        elif name == "single_done":
            state["cost"] = float(payload.get("cost", 0.0))
            state["duration"] = float(payload.get("duration", 0.0))

    orch.event_cb = on_event

    def worker(prompt: str):
        try:
            result = orch.execute(prompt)
            q.put(("ok", result["output"]))
        except Exception as e:
            q.put(("err", str(e)))

    def draw(stdscr):
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        log_h = min(6, max(3, h // 5))
        input_h = 3
        chat_h = max(4, h - log_h - input_h - 2)

        title = "llm-use chat  |  /quit to exit"
        stdscr.addnstr(0, 0, title.ljust(w), w - 1)

        y = 1
        chat_lines = []
        status_w = 0
        left_w = w
        if w >= 80:
            status_w = min(30, w // 3)
            left_w = w - status_w - 1
        for role, text in history:
            prefix = "You: " if role == "user" else "AI: "
            wrapped = textwrap.wrap(text, width=max(10, left_w - len(prefix) - 2))
            if not wrapped:
                wrapped = [""]
            chat_lines.append(prefix + wrapped[0])
            for line in wrapped[1:]:
                chat_lines.append(" " * len(prefix) + line)
        visible_chat = chat_lines[-chat_h:]
        for line in visible_chat:
            if y >= 1 + chat_h:
                break
            stdscr.addnstr(y, 0, line.ljust(left_w), left_w - 1)
            y += 1

        if status_w:
            sx = left_w + 1
            sy = 1
            stdscr.vline(1, left_w, "|", chat_h)
            stdscr.addnstr(sy, sx, "Status".ljust(status_w), status_w - 1)
            sy += 1
            stdscr.addnstr(sy, sx, f"Mode: {state['mode']}".ljust(status_w), status_w - 1)
            sy += 1
            if state["workers_total"] > 0:
                done = state["workers_done"]
                total = state["workers_total"]
                pct = int((done / total) * 100) if total else 0
                bar_w = max(10, status_w - 6)
                filled = int((done / total) * bar_w) if total else 0
                bar = "[" + "#" * filled + "-" * (bar_w - filled) + "]"
                stdscr.addnstr(sy, sx, f"Workers {done}/{total}".ljust(status_w), status_w - 1)
                sy += 1
                stdscr.addnstr(sy, sx, bar[:status_w-1].ljust(status_w), status_w - 1)
                sy += 1
                stdscr.addnstr(sy, sx, f"Success: {state['workers_ok']}".ljust(status_w), status_w - 1)
                sy += 1
                stdscr.addnstr(sy, sx, f"{pct}%".ljust(status_w), status_w - 1)
                sy += 1
            stdscr.addnstr(sy, sx, f"Cost: ${state['cost']:.4f}".ljust(status_w), status_w - 1)
            sy += 1
            stdscr.addnstr(sy, sx, f"Time: {state['duration']:.1f}s".ljust(status_w), status_w - 1)

        sep_y = 1 + chat_h
        stdscr.hline(sep_y, 0, "-", w - 1)

        log_y = sep_y + 1
        stdscr.addnstr(log_y, 0, f"Logs: {status}".ljust(w), w - 1)
        log_y += 1
        with log_lock:
            logs = list(log_lines)[- (log_h - 2):]
        for line in logs:
            if log_y >= sep_y + log_h:
                break
            stdscr.addnstr(log_y, 0, line.ljust(w), w - 1)
            log_y += 1

        input_y = sep_y + log_h + 1
        stdscr.hline(input_y - 1, 0, "-", w - 1)
        prompt = "> " + input_buf
        stdscr.addnstr(input_y, 0, prompt.ljust(w), w - 1)
        stdscr.move(input_y, min(len(prompt), w - 2))

        stdscr.refresh()

    def curses_main(stdscr):
        nonlocal input_buf, status, busy
        curses.curs_set(1)
        stdscr.nodelay(True)
        stdscr.keypad(True)

        while True:
            try:
                kind, payload = q.get_nowait()
                if kind == "ok":
                    history.append(("assistant", payload))
                    status = "Ready"
                else:
                    history.append(("assistant", f"Error: {payload}"))
                    status = "Ready"
                busy = False
            except queue.Empty:
                pass

            draw(stdscr)

            try:
                ch = stdscr.getch()
            except curses.error:
                ch = -1

            if ch == -1:
                time.sleep(0.02)
                continue

            if ch in (curses.KEY_EXIT, 3):
                break

            if busy:
                continue

            if ch in (10, 13):
                msg = input_buf.strip()
                input_buf = ""
                if not msg:
                    continue
                if msg.lower() in ("/quit", "/exit"):
                    break
                history.append(("user", msg))
                prompt = build_chat_prompt(history)
                status = "Working..."
                busy = True
                t = threading.Thread(target=worker, args=(prompt,), daemon=True)
                t.start()
                continue

            if ch in (curses.KEY_BACKSPACE, 127, 8):
                input_buf = input_buf[:-1]
                continue

            if 0 <= ch < 256:
                input_buf += chr(ch)

    try:
        curses.wrapper(curses_main)
    finally:
        for h in removed_handlers:
            logger.addHandler(h)

def run_mcp_server(orch: Orchestrator, host: str, port: int):
    if not HAS_POLYMCP:
        raise ValueError("pip install polymcp uvicorn")
    try:
        import uvicorn
    except ImportError:
        raise ValueError("pip install uvicorn")

    cache = orch.api.cache

    def exec_task(task: str) -> Dict[str, Any]:
        return orch.execute(task)

    def stats() -> Dict[str, Any]:
        return stats_snapshot()

    def scrape_url(url: str) -> Dict[str, str]:
        content = simple_scrape(url, cache=cache, backend=orch.scrape_backend)
        return {"url": url, "content": content}

    app = expose_tools_http([exec_task, stats, scrape_url])
    uvicorn.run(app, host=host, port=port)

def run_install(extras: List[str], dry_run: bool = False):
    pkgs = []
    if "scrape" in extras:
        pkgs.append("beautifulsoup4")
    if "mcp" in extras:
        pkgs.extend(["polymcp", "uvicorn"])
    if "playwright" in extras:
        pkgs.append("playwright")
    if not pkgs:
        print("Nothing to install. Use --scrape, --mcp, --playwright, or --all.")
        return
    cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    if dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=False)
    if "playwright" in extras:
        print("Run: playwright install")

def router_reset():
    cache = Cache(Path.home() / ".llm-use" / "cache.sqlite")
    try:
        cache._conn.execute("DELETE FROM router_examples")
        cache._conn.commit()
        print("Router memory cleared.")
    except Exception as e:
        print(f"Router reset failed: {e}")

def router_export(path: str):
    cache = Cache(Path.home() / ".llm-use" / "cache.sqlite")
    try:
        rows = cache.get_router_examples(limit=10000)
        data = [{"task": t, "mode": m, "created": c, "confidence": conf} for t, m, c, conf in rows]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(data)} examples to {path}")
    except Exception as e:
        print(f"Router export failed: {e}")

def router_import(path: str):
    cache = Cache(Path.home() / ".llm-use" / "cache.sqlite")
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Invalid format")
        for item in data:
            task = item.get("task")
            mode = item.get("mode")
            conf = item.get("confidence")
            if task and mode in ("single", "parallel"):
                cache.add_router_example(task, mode, confidence=conf)
        print(f"Imported {len(data)} examples from {path}")
    except Exception as e:
        print(f"Router import failed: {e}")

def parse_model(model_str: str) -> ModelConfig:
    if ":" in model_str and not model_str.endswith(":"):
        parts = model_str.split(":", 1)
        provider, model = parts[0], parts[1]
    else:
        model = model_str
        if model in DEFAULT_MODELS:
            provider = DEFAULT_MODELS[model]["provider"]
        else:
            raise ValueError(f"Unknown model: {model}. Use provider:model")
    if model in DEFAULT_MODELS:
        cfg = DEFAULT_MODELS[model]
        return ModelConfig(name=model, provider=provider, cost_in=cfg.get("cost_in", 0.0), cost_out=cfg.get("cost_out", 0.0))
    else:
        return ModelConfig(name=model, provider=provider)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="llm-use v2.0")
    subparsers = parser.add_subparsers(dest="command")

    def add_common_args(p):
        p.add_argument("--orchestrator", default="claude-3-7-sonnet-20250219")
        p.add_argument("--worker", default="claude-3-5-haiku-20241022")
        p.add_argument("--router", default=None)
        p.add_argument("--router-path", default=None, help="Use a local llama.cpp model path as router (implies llama_cpp provider)")
        p.add_argument("--max-workers", type=int, default=10)
        p.add_argument("--ollama-url", default="http://localhost:11434")
        p.add_argument("--llama-cpp-url", default="http://localhost:8080")
        p.add_argument("--enable-scrape", action="store_true")
        p.add_argument("--max-scrape-urls", type=int, default=3)
        p.add_argument("--scrape-backend", choices=["requests", "playwright"], default="requests")
        p.add_argument("--no-cache", action="store_true")

    exec_parser = subparsers.add_parser("exec")
    exec_parser.add_argument("--task", required=True)
    add_common_args(exec_parser)

    chat_parser = subparsers.add_parser("chat")
    add_common_args(chat_parser)

    mcp_parser = subparsers.add_parser("mcp")
    add_common_args(mcp_parser)
    mcp_parser.add_argument("--host", default="127.0.0.1")
    mcp_parser.add_argument("--port", type=int, default=8000)

    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--all", action="store_true")
    install_parser.add_argument("--scrape", action="store_true")
    install_parser.add_argument("--mcp", action="store_true")
    install_parser.add_argument("--playwright", action="store_true")
    install_parser.add_argument("--dry-run", action="store_true")

    stats_parser = subparsers.add_parser("stats")
    router_reset_parser = subparsers.add_parser("router-reset")
    router_export_parser = subparsers.add_parser("router-export")
    router_export_parser.add_argument("--out", required=True)
    router_import_parser = subparsers.add_parser("router-import")
    router_import_parser.add_argument("--in", dest="inp", required=True)
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if args.command == "exec":
        try:
            orch_config = parse_model(args.orchestrator)
            worker_config = parse_model(args.worker)
            if args.router_path:
                router_config = ModelConfig(name=os.path.expanduser(args.router_path), provider="llama_cpp")
            else:
                router_config = parse_model(args.router) if args.router else None
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        orch = Orchestrator(
            orch_config,
            worker_config,
            args.max_workers,
            anthropic_key,
            openai_key,
            args.ollama_url,
            args.llama_cpp_url,
            enable_cache=not args.no_cache,
            enable_scrape=args.enable_scrape,
            max_scrape_urls=args.max_scrape_urls,
            scrape_backend=args.scrape_backend,
            router=router_config
        )
        result = orch.execute(args.task)
        print(f"\n{'='*60}\nOUTPUT\n{'='*60}")
        print(result["output"])
        print(f"\n{'='*60}")
        print(f"Mode: {result['mode']}")
        if result["mode"] == "parallel":
            print(f"Workers: {result['workers_spawned']} spawned, {result['workers_succeeded']} ok")
            print(f"\nBreakdown:")
            print(f"  Orch: ${result['breakdown']['orchestrator']:.6f}")
            print(f"  Workers: ${result['breakdown']['workers']:.6f}")
            print(f"  Synth: ${result['breakdown']['synthesis']:.6f}")
        print(f"\nCost: ${result['cost']:.6f}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Session: {result['session_id']}\n")
    elif args.command == "chat":
        try:
            orch_config = parse_model(args.orchestrator)
            worker_config = parse_model(args.worker)
            if args.router_path:
                router_config = ModelConfig(name=os.path.expanduser(args.router_path), provider="llama_cpp")
            else:
                router_config = parse_model(args.router) if args.router else None
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        orch = Orchestrator(
            orch_config,
            worker_config,
            args.max_workers,
            anthropic_key,
            openai_key,
            args.ollama_url,
            args.llama_cpp_url,
            enable_cache=not args.no_cache,
            enable_scrape=args.enable_scrape,
            max_scrape_urls=args.max_scrape_urls,
            scrape_backend=args.scrape_backend,
            router=router_config
        )
        run_chat_tui(orch)
    elif args.command == "mcp":
        try:
            orch_config = parse_model(args.orchestrator)
            worker_config = parse_model(args.worker)
            if args.router_path:
                router_config = ModelConfig(name=os.path.expanduser(args.router_path), provider="llama_cpp")
            else:
                router_config = parse_model(args.router) if args.router else None
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        orch = Orchestrator(
            orch_config,
            worker_config,
            args.max_workers,
            anthropic_key,
            openai_key,
            args.ollama_url,
            args.llama_cpp_url,
            enable_cache=not args.no_cache,
            enable_scrape=args.enable_scrape,
            max_scrape_urls=args.max_scrape_urls,
            scrape_backend=args.scrape_backend,
            router=router_config
        )
        try:
            run_mcp_server(orch, args.host, args.port)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.command == "install":
        extras = []
        if args.all:
            extras = ["scrape", "mcp", "playwright"]
        else:
            if args.scrape:
                extras.append("scrape")
            if args.mcp:
                extras.append("mcp")
            if args.playwright:
                extras.append("playwright")
        run_install(extras, dry_run=args.dry_run)
    elif args.command == "stats":
        print_stats()
    elif args.command == "router-reset":
        router_reset()
    elif args.command == "router-export":
        router_export(args.out)
    elif args.command == "router-import":
        router_import(args.inp)

if __name__ == "__main__":
    main()
