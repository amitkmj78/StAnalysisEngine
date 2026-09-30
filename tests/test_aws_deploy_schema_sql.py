import pytest

from web.backend.routers.aws_deploy import _SCHEMA_SQL


def test_schema_sql_contains_unescaped_braces_that_break_format():
    # Regression guard for the actual bug: _SCHEMA_SQL has literal,
    # unescaped '{}' from `default '{}'::jsonb` columns -- .format() reads
    # those as positional placeholders and raises IndexError. If this
    # assertion ever starts failing because the SQL no longer contains a
    # bare '{}', the .format()-vs-.replace() distinction may no longer
    # matter, but until then the router must use .replace(), not .format().
    with pytest.raises(IndexError):
        _SCHEMA_SQL.format(app_user_pw="x", app_service_pw="y")


def test_schema_sql_placeholder_substitution_via_replace():
    # Mirrors exactly what web/backend/routers/aws_deploy.py's deploy
    # worker does -- plain .replace(), not .format() (see the test above
    # for why .format() is unsafe on this string).
    rendered = _SCHEMA_SQL.replace("{app_user_pw}", "my-user-pw").replace("{app_service_pw}", "my-service-pw")

    assert "{app_user_pw}" not in rendered
    assert "{app_service_pw}" not in rendered
    assert "my-user-pw" in rendered
    assert "my-service-pw" in rendered
    # The unrelated literal braces must survive untouched -- confirms the
    # substitution didn't also mangle real SQL content.
    assert "default '{}'::jsonb" in rendered
