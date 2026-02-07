"""
Import shim for cli.py so it can be used as a library.
"""

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent / "cli.py"
_spec = spec_from_file_location("llm_use_cli", _MODULE_PATH)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore

Orchestrator = _mod.Orchestrator
ModelConfig = _mod.ModelConfig
SessionManager = _mod.SessionManager
Call = _mod.Call
ExecMode = _mod.ExecMode
parse_model = _mod.parse_model
print_stats = _mod.print_stats
stats_snapshot = _mod.stats_snapshot
main = _mod.main

__all__ = [
    "Orchestrator",
    "ModelConfig",
    "SessionManager",
    "Call",
    "ExecMode",
    "parse_model",
    "print_stats",
    "stats_snapshot",
    "main",
]
