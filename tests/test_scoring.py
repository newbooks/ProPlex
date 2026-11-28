from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest
import torch

from proplex import SequenceResult, score_sequence
from proplex.scoring import (
    DEFAULT_MODEL,
    _ModelBundle,
    _normalize_sequence,
    _prepare_sequence,
    _TokenizerBundle,
    _top_amino_acid_candidates,
    parse_fasta,
    plan_windows,
    score_fasta,
    select_window,
)


class FakeTokenizer:
    bos_token_id = None
    cls_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    mask_token_id = 3


class FakeModel:
    def __call__(self, input_ids, attention_mask):
        # Target residue IDs are all assigned logit 2; all other vocabulary
        # entries remain zero, making full-vocabulary normalization testable.
        logits = torch.zeros((*input_ids.shape, 32), device=input_ids.device)
        logits[..., 4:24] = 2.0
        # A non-residue vocabulary entry outranks every amino-acid candidate.
        logits[..., 31] = 3.0
        return type("Output", (), {"logits": logits})()


def fake_bundle() -> _ModelBundle:
    residues = "ACDEFGHIKLMNPQRSTVWYBXZUO"
    ids = {residue: index + 4 for index, residue in enumerate(residues)}
    return _ModelBundle(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device=torch.device("cpu"),
        accepted_residues=frozenset(ids),
        residue_token_ids=ids,
    )


def fake_tokenizer_bundle(bundle: _ModelBundle | None = None) -> _TokenizerBundle:
    model_bundle = bundle or fake_bundle()
    return _TokenizerBundle(
        tokenizer=model_bundle.tokenizer,
        accepted_residues=model_bundle.accepted_residues,
        residue_token_ids=model_bundle.residue_token_ids,
    )


def test_sequence_normalization_and_alphabet() -> None:
    accepted = fake_bundle().accepted_residues
    assert _prepare_sequence("  acdBXZUO  ", trim=True) == "ACDBXZUO"
    for invalid in ("", "AC D", "AC-", "AC*"):
        with pytest.raises(ValueError):
            _prepare_sequence(invalid, trim=True)
    assert _normalize_sequence("ACD", accepted) == "ACD"


@pytest.mark.parametrize("sequence", ["", "AC D", "AC*"])
def test_invalid_sequence_is_rejected_before_model_loading(monkeypatch, sequence: str) -> None:
    def fail_if_loaded(_model_name: str):
        raise AssertionError("model loading should not occur for invalid input")

    monkeypatch.setattr("proplex.scoring._load_tokenizer", fail_if_loaded)
    monkeypatch.setattr("proplex.scoring._load_model", fail_if_loaded)
    with pytest.raises(ValueError):
        score_sequence(sequence)


@pytest.mark.parametrize("length", [1, 1021, 1022, 1942, 3000])
def test_windows_cover_sequence_and_respect_policy(length: int) -> None:
    windows = plan_windows(length)
    assert windows[0][0] == 1
    assert windows[-1][1] == length
    assert all(end - start + 1 <= 1021 for start, end in windows)
    if length > 1021:
        assert all(end - start + 1 == 1021 for start, end in windows)
        assert all(right[0] - left[1] - 1 <= 921 for left, right in pairwise(windows))
    for position in range(1, length + 1):
        selected = windows[select_window(position, windows)]
        assert selected[0] <= position <= selected[1]


def test_center_selection_tie_breaks_to_smaller_start() -> None:
    windows = ((1, 5), (3, 7))
    assert select_window(4, windows) == 0


def test_score_sequence_produces_per_residue_and_aggregate(monkeypatch) -> None:
    monkeypatch.setattr("proplex.scoring._load_tokenizer", lambda _: fake_tokenizer_bundle())
    monkeypatch.setattr("proplex.scoring._load_model", lambda _name, _tok: fake_bundle())
    result = score_sequence("acd", model_name=DEFAULT_MODEL)

    assert isinstance(result, SequenceResult)
    assert result.sequence == "ACD"
    assert result.sequence_id is None
    assert result.length == 3
    assert [score.position for score in result.residue_scores] == [1, 2, 3]
    assert [score.residue for score in result.residue_scores] == list("ACD")
    fake_logits = torch.tensor([0.0] * 32)
    fake_logits[4:24] = 2.0
    fake_logits[31] = 3.0
    expected_probability = torch.softmax(fake_logits, dim=0)[4].item()
    assert [score.top_candidate for score in result.residue_scores] == ["A"] * 3
    assert all(
        score.top_candidate_probability == pytest.approx(expected_probability)
        for score in result.residue_scores
    )
    assert all(
        score.probability == pytest.approx(expected_probability) for score in result.residue_scores
    )
    assert result.psn == pytest.approx(1 / result.pseudo_perplexity)


def test_top_candidate_ties_resolve_alphabetically_independent_of_token_ids() -> None:
    log_probs = torch.log_softmax(torch.tensor([[0.0] * 10]), dim=-1)
    candidates = _top_amino_acid_candidates(log_probs, [("B", 3), ("A", 7)])
    assert candidates[0][0] == "A"
    assert candidates[0][1] == pytest.approx(torch.exp(log_probs[0, 7]).item())


def test_fasta_parser_multiline_records(tmp_path: Path) -> None:
    fasta = tmp_path / "records.fasta"
    fasta.write_text(">first description\nACD\nEF\n\n>second\nGH\n", encoding="utf-8")
    assert parse_fasta(fasta) == [("first", "ACDEF"), ("second", "GH")]


@pytest.mark.parametrize("contents", ["", "ACD", ">empty\n", ">x\nACD\n>y\n"])
def test_fasta_parser_rejects_missing_records_or_sequences(tmp_path: Path, contents: str) -> None:
    fasta = tmp_path / "bad.fasta"
    fasta.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_fasta(fasta)


def test_fasta_parser_preserves_sequence_whitespace_for_validation(tmp_path: Path) -> None:
    fasta = tmp_path / "spaced.fasta"
    fasta.write_text(">record\n AC D \t\n", encoding="utf-8")
    assert parse_fasta(fasta) == [("record", " AC D \t")]


def test_score_fasta_preserves_identifiers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("proplex.scoring._load_tokenizer", lambda _: fake_tokenizer_bundle())
    monkeypatch.setattr("proplex.scoring._load_model", lambda _name, _tok: fake_bundle())
    fasta = tmp_path / "records.fasta"
    fasta.write_text(">first record\nACD\n>second\nEFG\n", encoding="utf-8")
    results = score_fasta(fasta)
    assert [result.sequence_id for result in results] == ["first", "second"]


def test_fasta_uses_selected_tokenizer_alphabet(monkeypatch, tmp_path: Path) -> None:
    bundle = fake_bundle()
    token_ids = {**bundle.residue_token_ids, "J": 29}
    bundle = replace(
        bundle,
        residue_token_ids=token_ids,
        accepted_residues=frozenset(token_ids),
    )
    tokenizer_bundle = fake_tokenizer_bundle(bundle)
    monkeypatch.setattr("proplex.scoring._load_tokenizer", lambda _: tokenizer_bundle)
    monkeypatch.setattr("proplex.scoring._load_model", lambda _name, _tok: bundle)
    fasta = tmp_path / "extended.fasta"
    fasta.write_text(">record\nAJ\n", encoding="utf-8")
    results = score_fasta(fasta)
    assert results[0].sequence == "AJ"


def test_fasta_edge_whitespace_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("proplex.scoring._load_tokenizer", lambda _: fake_tokenizer_bundle())
    monkeypatch.setattr("proplex.scoring._load_model", lambda _name, _tok: fake_bundle())
    fasta = tmp_path / "spaced.fasta"
    fasta.write_text(">record\n AC \n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid character"):
        score_fasta(fasta)
