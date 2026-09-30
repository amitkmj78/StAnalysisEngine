import ast
from pathlib import Path

import pytest

# Extracted via ast from the source file rather than `import
# web.backend.routers.aws_deploy` -- that module imports boto3 at module
# level, which isn't in requirements-test.txt (CI's deliberately minimal
# dependency set, scoped to pure-logic tests) and would fail collection
# in CI even though it isn't needed to check this string constant's
# content. Reads the real, current source either way, so it can't drift
# out of sync with the router the way a hand-copied duplicate could.
_SOURCE_PATH = Path(__file__).resolve().parent.parent / "web" / "backend" / "routers" / "aws_deploy.py"


def _load_schema_sql() -> str:
    tree = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(_SOURCE_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_SCHEMA_SQL" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("_SCHEMA_SQL assignment not found in aws_deploy.py")


def test_schema_sql_contains_unescaped_braces_that_break_format():
    # Regression guard for the actual bug: _SCHEMA_SQL has literal,
    # unescaped '{}' from `default '{}'::jsonb` columns -- .format() reads
    # those as positional placeholders and raises IndexError. If this
    # assertion ever starts failing because the SQL no longer contains a
    # bare '{}', the .format()-vs-.replace() distinction may no longer
    # matter, but until then the router must use .replace(), not .format().
    schema_sql = _load_schema_sql()
    with pytest.raises(IndexError):
        schema_sql.format(app_user_pw="x", app_service_pw="y")


def test_schema_sql_placeholder_substitution_via_replace():
    # Mirrors exactly what web/backend/routers/aws_deploy.py's deploy
    # worker does -- plain .replace(), not .format() (see the test above
    # for why .format() is unsafe on this string).
    schema_sql = _load_schema_sql()
    rendered = schema_sql.replace("{app_user_pw}", "my-user-pw").replace("{app_service_pw}", "my-service-pw")

    assert "{app_user_pw}" not in rendered
    assert "{app_service_pw}" not in rendered
    assert "my-user-pw" in rendered
    assert "my-service-pw" in rendered
    # The unrelated literal braces must survive untouched -- confirms the
    # substitution didn't also mangle real SQL content.
    assert "default '{}'::jsonb" in rendered
