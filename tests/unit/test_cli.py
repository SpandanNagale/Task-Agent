from typer.testing import CliRunner

from task_agent.cli import app
from tests.conftest import MOCK_MODELS, SUITE

cli = CliRunner()


def test_version() -> None:
    result = cli.invoke(app, ["version"])
    assert result.exit_code == 0


def test_models_lists_roles() -> None:
    result = cli.invoke(app, ["models", "--models", str(MOCK_MODELS)])
    assert result.exit_code == 0
    assert "planner" in result.output


def test_run_json() -> None:
    result = cli.invoke(app, ["run", "What is 6 x 7?", "--json", "--models", str(MOCK_MODELS)])
    assert result.exit_code == 0, result.output
    assert '"status": "succeeded"' in result.output


def test_eval_min_success_gate(tmp_path) -> None:
    args = [
        "eval",
        str(SUITE),
        "--tag",
        "smoke",
        "--models",
        str(MOCK_MODELS),
        "--out",
        str(tmp_path),
    ]
    assert cli.invoke(app, args).exit_code == 0
    # The mock model always answers 42, so a success floor fails the run.
    assert cli.invoke(app, [*args, "--min-success", "0.5"]).exit_code == 1
    assert list(tmp_path.glob("*.json"))


def test_bad_models_path_exits_2(tmp_path) -> None:
    result = cli.invoke(app, ["models", "--models", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 2


def test_tools_lists_registered() -> None:
    result = cli.invoke(app, ["tools", "--models", str(MOCK_MODELS)])
    assert result.exit_code == 0, result.output
    assert "read_url" in result.output


def test_run_direct_runner() -> None:
    args = ["run", "What is 6 x 7?", "--runner", "direct", "--models", str(MOCK_MODELS)]
    result = cli.invoke(app, args)
    assert result.exit_code == 0, result.output
