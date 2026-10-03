import ast
import builtins
from pathlib import Path

ROUTER = Path(__file__).resolve().parents[1] / "web" / "backend" / "routers" / "challenges.py"


def _undefined_names(source: str) -> set[str]:
    tree = ast.parse(source)
    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return used - defined


def test_challenges_router_has_no_unresolved_names():
    # Regression guard: mask_email moved to the service module and the router
    # kept calling it by name, so member lists, curves and invites returned 500.
    # Checked statically, so this test never has to import the web stack.
    assert _undefined_names(ROUTER.read_text(encoding="utf-8")) == set()
