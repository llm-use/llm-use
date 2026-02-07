import importlib.util
import time
from pathlib import Path
from dataclasses import asdict

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "cli.py"

spec = importlib.util.spec_from_file_location("llm_use", MODULE_PATH)
llm_use = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_use)


def test_parse_orchestrator_json_codefence():
    text = """```json\n{\"mode\": \"single\", \"response\": \"ok\"}\n```"""
    out = llm_use.parse_orchestrator_json(text)
    assert out["mode"] == "single"
    assert out["response"] == "ok"


def test_parse_orchestrator_json_embedded():
    text = "Here you go:\n{\"mode\": \"parallel\", \"subtasks\": [], \"combine_strategy\": \"x\"}\nThanks"
    out = llm_use.parse_orchestrator_json(text)
    assert out["mode"] == "parallel"
    assert out["combine_strategy"] == "x"


class FakeAPI:
    def call(self, *args, **kwargs):
        time.sleep(0.1)
        return llm_use.Call("c1", "m", "p", "", "", 0, 0, 0.0, 0.0)


def test_spawn_workers_global_timeout(monkeypatch):
    monkeypatch.setattr(llm_use, "WORKER_GLOBAL_TIMEOUT", 0.01)
    monkeypatch.setattr(llm_use, "WORKER_CALL_TIMEOUT", 0.01)

    orch = llm_use.Orchestrator.__new__(llm_use.Orchestrator)
    orch.api = FakeAPI()
    orch.worker_config = llm_use.ModelConfig(name="m", provider="p")
    orch.sessions = llm_use.SessionManager()
    orch.max_workers = 2

    orch_call = llm_use.Call("orch", "m", "p", "", "", 0, 0, 0.0, 0.0)
    orch.sessions.create("task", llm_use.ExecMode.PARALLEL, orch_call)

    subtasks = [{"id": 1, "task": "t1"}, {"id": 2, "task": "t2"}]
    results = orch._spawn_workers(subtasks)
    assert len(results) == 2
    assert all(r.error == "timeout" for r in results)


def test_print_stats(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))

    sessions_dir = tmp_path / ".llm-use" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    orch_call = llm_use.Call("orch", "m", "p", "", "", 0, 0, 0.0, 0.0)
    session = llm_use.Session(
        id="abc123",
        task="test",
        mode=llm_use.ExecMode.SINGLE,
        orchestrator_call=orch_call,
        worker_calls=[],
        synthesis_call=None,
        output="ok",
        total_cost=0.0,
        total_duration=0.0,
    )
    data = asdict(session)
    data["mode"] = session.mode.value
    with open(sessions_dir / "abc123.json", "w") as f:
        import json
        json.dump(data, f)

    llm_use.print_stats()
    out = capsys.readouterr().out
    assert "STATISTICS" in out
    assert "Total: 1" in out

def test_import_shim():
    import llm_use as lib
    assert hasattr(lib, "Orchestrator")
