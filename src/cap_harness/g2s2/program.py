# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

import ast
import contextlib
from dataclasses import dataclass
import importlib
import io
import math
import time
import traceback
from types import MappingProxyType
from typing import Any

import numpy as np

from cap_harness.g2s2.api import PROGRAM_HELPERS, S2Api
from cap_harness.g2s2.episode import EpisodeEnded
from cap_harness.g2s2.motion import MotionError

def _warm_numpy_methods() -> None:
    a = np.ones((2, 3))
    b = a > 0
    calls = (
        lambda: a.sum(), lambda: a.sum(axis=0), lambda: a.mean(), lambda: a.mean(axis=0), lambda: a.std(), lambda: a.std(axis=0),
        lambda: a.var(), lambda: a.var(axis=0), lambda: a.max(), lambda: a.max(axis=0), lambda: a.min(), lambda: a.min(axis=0),
        lambda: a.prod(), lambda: a.prod(axis=0), lambda: a.clip(0, 1), lambda: b.all(), lambda: b.all(axis=1), lambda: b.any(),
        lambda: b.any(axis=1), lambda: b.sum(), lambda: b.sum(axis=0), lambda: a.cumsum(), lambda: a.cumprod(), lambda: a.round(1),
        lambda: a.argmax(), lambda: a.argmin(), lambda: a.argsort(), lambda: a.nonzero(), lambda: a.dot(a.T), lambda: a.trace(),
        lambda: np.median(a, axis=0), lambda: np.percentile(a, 50, axis=0), lambda: a.astype(np.float32).mean(),
    )
    for call in calls:
        try:
            call()
        except Exception:
            pass


_warm_numpy_methods()

ALLOWED_IMPORTS = frozenset({"math", "numpy"})
ALLOWED_ALIASES = {"numpy": {"np", None}, "math": {None}}


def _restricted_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    if level != 0 or name not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed in Season II programs")
    return importlib.import_module(name)


SAFE_BUILTINS = MappingProxyType(
    {
        "__import__": _restricted_import,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "RuntimeError": RuntimeError,
        "ZeroDivisionError": ZeroDivisionError,
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate,
        "float": float, "int": int, "isinstance": isinstance, "len": len, "list": list, "max": max,
        "min": min, "print": print, "range": range, "reversed": reversed, "round": round, "set": set,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip, "map": map, "filter": filter,
        "None": None, "True": True, "False": False,
    }
)


class _Validator(ast.NodeVisitor):
    _forbidden = (ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.Global, ast.Nonlocal, ast.With, ast.ImportFrom)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, self._forbidden):
            raise ValueError(f"{type(node).__name__} is not allowed in Season II programs")
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in ALLOWED_IMPORTS or alias.asname not in ALLOWED_ALIASES[alias.name]:
                raise ValueError(f"import of {alias.name!r} (as {alias.asname!r}) is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise ValueError("private and dunder attribute access is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            raise ValueError("private and dunder names are not allowed")
        self.generic_visit(node)


def check_program(code: str) -> Any:
    tree = ast.parse(code, mode="exec")
    _Validator().visit(tree)
    return compile(tree, "<s2-program>", "exec")


@dataclass(slots=True)
class ProgramResult:
    ok: bool
    ended_by_simulator: bool
    error: str | None
    traceback: str | None
    stdout: str
    wall_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "ended_by_simulator": self.ended_by_simulator,
            "error": self.error,
            "traceback": self.traceback,
            "stdout": self.stdout[-20000:],
            "wall_s": round(self.wall_s, 3),
        }


def run_program(code: str, api: S2Api, report: dict[str, Any]) -> ProgramResult:
    started = time.time()
    output = io.StringIO()
    try:
        compiled = check_program(code)
    except (SyntaxError, ValueError) as exc:
        return ProgramResult(False, False, f"program rejected: {exc}", None, "", time.time() - started)
    environment: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "api": api,
        "report": report,
        "np": np,
        "math": math,
        **PROGRAM_HELPERS,
    }
    try:
        with contextlib.redirect_stdout(output):
            exec(compiled, environment, environment)
    except EpisodeEnded as exc:
        return ProgramResult(True, True, f"episode ended by the simulator: {exc}", None, output.getvalue(), time.time() - started)
    except MotionError as exc:
        return ProgramResult(False, False, f"MotionError: {exc}", traceback.format_exc(limit=8), output.getvalue(), time.time() - started)
    except Exception as exc:
        return ProgramResult(False, False, f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=12), output.getvalue(), time.time() - started)
    return ProgramResult(True, False, None, None, output.getvalue(), time.time() - started)


__all__ = ["ALLOWED_IMPORTS", "ProgramResult", "check_program", "run_program"]
