import json

import pytest
from typer.testing import CliRunner

from proplex.cli import _write_results, app, render_residue_dots
from proplex.scoring import ProPlexError, ResidueScore, SequenceResult

runner = CliRunner()


def _result(sequence_id: str | None = None) -> SequenceResult:
    return SequenceResult(
        sequence="AC",
        sequence_id=sequence_id,
        model_name="fake",
        length=2,
        mean_log_probability=-1.0,
        pseudo_perplexity=2.718,
        psn=0.368,
        residue_scores=(
            ResidueScore(1, "A", -0.5, 0.607, "A", 0.607),
            ResidueScore(2, "C", -1.5, 0.223, "C", 0.223),
        ),
    )


def test_root_help_and_bare_command_show_query_command() -> None:
    for arguments in (["--help"], ["-h"], []):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0
        assert "query" in result.stdout
        assert "--seq" not in result.stdout


def test_query_help_and_bare_query_show_query_options() -> None:
    for arguments in (["query", "--help"], ["query", "-h"], ["query"]):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0
        assert "--seq" in result.stdout
        assert "--fasta" in result.stdout
        assert "--output" in result.stdout
        assert "--verbose" not in result.stdout


def test_seq_option_writes_complete_json_and_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("proplex.cli.score_sequence", lambda sequence: _result())
    output = tmp_path / "custom-results.json"
    result = runner.invoke(app, ["query", "--seq", "AC", "--output", str(output)])
    assert result.exit_code == 0
    assert "sequence\t2" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["results"][0]["residue_scores"][0]["probability"] == 0.607
    assert payload["results"][0]["residue_scores"][0]["top_candidate"] == "A"
    assert payload["results"][0]["residue_scores"][0]["top_candidate_probability"] == 0.607


def test_fasta_output_defaults_next_to_input(monkeypatch, tmp_path) -> None:
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">p1\nAC\n", encoding="utf-8")
    monkeypatch.setattr("proplex.cli.score_fasta", lambda path: [_result("p1")])
    result = runner.invoke(app, ["query", "--fasta", str(fasta)])
    assert result.exit_code == 0
    assert (tmp_path / "proteins.proplex.json").is_file()


def test_seq_and_fasta_are_mutually_exclusive() -> None:
    result = runner.invoke(app, ["query", "--seq", "AC", "--fasta", "input.fa"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr


def test_unexpected_model_setup_error_is_reported_without_traceback(monkeypatch, tmp_path) -> None:
    def fail_loading(sequence: str):
        raise RuntimeError("simulated model setup failure")

    monkeypatch.setattr("proplex.cli.score_sequence", fail_loading)
    result = runner.invoke(app, ["query", "--seq", "AC", "--output", str(tmp_path / "result.json")])
    assert result.exit_code == 1
    assert "simulated model setup failure" in result.stderr
    assert "Traceback" not in result.output


def test_json_temp_cleanup_failure_preserves_original_write_error(monkeypatch, tmp_path) -> None:
    def fail_serialization(*args, **kwargs):
        raise ValueError("simulated serialization failure")

    def fail_unlink(*args, **kwargs):
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr("proplex.cli.json.dump", fail_serialization)
    monkeypatch.setattr("pathlib.Path.unlink", fail_unlink)
    with pytest.raises(ProPlexError, match="simulated serialization failure"):
        _write_results(tmp_path / "result.json", [_result()])


def test_residue_dot_display_uses_ten_levels_and_percentages() -> None:
    result = SequenceResult(
        sequence="ACDE",
        sequence_id="protein-1",
        model_name="fake",
        length=4,
        mean_log_probability=-1.0,
        pseudo_perplexity=2.718,
        psn=0.368,
        residue_scores=(
            ResidueScore(1, "A", -4.0, 0.1, "H", 0.25),
            ResidueScore(2, "C", -0.6, 0.55, "Q", 0.279),
            ResidueScore(3, "D", -0.01, 0.99, "F", 0.458),
            ResidueScore(4, "E", 0.0, 1.0, "E", 1.0),
        ),
    )
    rendered = render_residue_dots(result).splitlines()
    assert rendered[0] == ">protein-1"
    assert rendered[1] == "Position  AA Probability  Percent    Candidate   Probability"
    assert [line.split()[2] for line in rendered[2:]] == [".", ".....", ".........", ".........."]
    assert [line.split()[-2:] for line in rendered[2:]] == [
        ["H", "25.00%"],
        ["Q", "27.90%"],
        ["F", "45.80%"],
        ["E", "100.00%"],
    ]
