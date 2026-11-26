from typer.testing import CliRunner

from proplex import score_sequence
from proplex.cli import app


runner = CliRunner()


def test_help_lists_placeholder_options() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--seq" in result.stdout
    assert "--fasta" in result.stdout
    assert "--verbose" in result.stdout


def test_short_help_alias() -> None:
    result = runner.invoke(app, ["-h"])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_placeholder_options_fail_without_claiming_success() -> None:
    result = runner.invoke(app, ["--seq", "ACDE"])

    assert result.exit_code == 1
    assert "not implemented" in result.stderr


def test_sequence_and_fasta_options_are_mutually_exclusive() -> None:
    result = runner.invoke(app, ["--seq", "ACDE", "--fasta", "input.fasta"])

    assert result.exit_code == 2
    assert "--seq and --fasta are mutually exclusive" in result.stderr


def test_public_scoring_function_is_importable_and_explicitly_unimplemented() -> None:
    try:
        score_sequence("ACDE")
    except NotImplementedError as exc:
        assert "not implemented" in str(exc)
    else:
        raise AssertionError("placeholder score_sequence unexpectedly returned")
