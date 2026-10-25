import pytest

from task_agent.observability.redaction import Redactor


def test_known_secret_values_are_removed() -> None:
    r = Redactor(["my-very-secret-value", "short"])
    assert r("key=my-very-secret-value; short stays") == "key=[REDACTED]; short stays"


@pytest.mark.parametrize(
    "secret",
    [
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "sk-lf-1234abcd-5678",
        "gsk_abcdefghijklmnopqrstuvwxyz",
        "tvly-abcdefghijklmnop1234",
        "Bearer abcdefghijklmnopqrstuvwxyz",
    ],
)
def test_patterns(secret: str) -> None:
    assert secret not in Redactor()(f"token {secret} end")


def test_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_API_KEY", "value-that-is-secret")
    assert "value-that-is-secret" not in Redactor.from_environment()("x value-that-is-secret")
