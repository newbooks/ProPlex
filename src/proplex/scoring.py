"""Sequence validation, ESM-2 masked scoring, and result types."""

from __future__ import annotations

import logging
import math
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "facebook/esm2_t30_150M_UR50D"
WINDOW_LENGTH = 1021
MINIMUM_OVERLAP = 100
INFERENCE_BATCH_SIZE = 8
LOGGER = logging.getLogger(__name__)
_HF_AUTH_WARNING = "You are sending unauthenticated requests to the HF Hub"


@contextmanager
def _quiet_native_stderr():
    """Hide native runtime probes that write expected diagnostics to fd 2."""
    saved_stderr = os.dup(2)
    null_stderr = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_stderr, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(null_stderr)


def _silence_cuda_probe_stderr() -> bool:
    """Query CUDA availability without leaking native ROCm/DRM probe chatter."""
    import torch

    with _quiet_native_stderr():
        return torch.cuda.is_available()


class ProPlexError(ValueError):
    """A user-correctable input or scoring error."""


@dataclass(frozen=True)
class ResidueScore:
    """Prediction for one one-based sequence position."""

    position: int
    residue: str
    log_probability: float
    probability: float
    top_candidate: str
    top_candidate_probability: float


@dataclass(frozen=True)
class SequenceResult:
    """Aggregate and per-residue ProPlex score for one protein sequence."""

    sequence: str
    sequence_id: str | None
    model_name: str
    length: int
    mean_log_probability: float
    pseudo_perplexity: float
    psn: float
    residue_scores: tuple[ResidueScore, ...]


@dataclass(frozen=True)
class _ModelBundle:
    tokenizer: Any
    model: Any
    device: Any
    accepted_residues: frozenset[str]
    residue_token_ids: dict[str, int]


@dataclass(frozen=True)
class _TokenizerBundle:
    tokenizer: Any
    accepted_residues: frozenset[str]
    residue_token_ids: dict[str, int]


def plan_windows(length: int) -> tuple[tuple[int, int], ...]:
    """Plan deterministic one-based inclusive windows for a sequence."""
    if length < 1:
        raise ProPlexError("sequence length must be positive")
    if length <= WINDOW_LENGTH:
        return ((1, length),)
    count = math.ceil((length - MINIMUM_OVERLAP) / (WINDOW_LENGTH - MINIMUM_OVERLAP))
    starts = tuple(1 + (index * (length - WINDOW_LENGTH)) // (count - 1) for index in range(count))
    windows = tuple((start, start + WINDOW_LENGTH - 1) for start in starts)
    if windows[0][0] != 1 or windows[-1][1] != length:
        raise ProPlexError("window plan is not anchored at both sequence termini")
    if any(
        right[0] - left[1] - 1 > WINDOW_LENGTH - MINIMUM_OVERLAP
        for left, right in pairwise(windows)
    ):
        raise ProPlexError("window plan violates the minimum overlap")
    return windows


def select_window(position: int, windows: tuple[tuple[int, int], ...]) -> int:
    """Return the index of the covering window nearest its center."""
    candidates = [
        (abs(2 * position - (start + end)), start, index)
        for index, (start, end) in enumerate(windows)
        if start <= position <= end
    ]
    if not candidates:
        raise ProPlexError(f"no inference window covers residue position {position}")
    return min(candidates)[2]


def parse_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Read FASTA records as (first header token, concatenated sequence)."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    except (OSError, UnicodeError) as exc:
        raise ProPlexError(f"cannot read FASTA file: {source}") from exc

    records: list[tuple[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []

    def finish_record() -> None:
        if current_id is None:
            return
        sequence = "".join(chunks)
        if not sequence:
            raise ProPlexError(f"FASTA record {current_id!r} has no sequence")
        records.append((current_id, sequence))

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if line == "":
            continue
        if line.startswith(">"):
            finish_record()
            header = line[1:].strip()
            if not header:
                raise ProPlexError(f"FASTA header on line {line_number} is empty")
            current_id = header.split()[0]
            chunks = []
        else:
            if current_id is None:
                raise ProPlexError(f"sequence text before first FASTA header on line {line_number}")
            chunks.append(line)
    finish_record()
    if not records:
        raise ProPlexError("FASTA file contains no records")
    return records


def _load_tokenizer(model_name: str) -> _TokenizerBundle:
    """Load the tokenizer and derive accepted residues from single-token vocabulary entries."""
    try:
        for logger_name in ("httpx", "httpcore"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        for logger_name in ("huggingface_hub", "transformers"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        from transformers import AutoTokenizer
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=f"{_HF_AUTH_WARNING}.*")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        residue_token_ids: dict[str, int] = {}
        special_ids = set(tokenizer.all_special_ids)
        unk_id = tokenizer.unk_token_id
        vocabulary = tokenizer.get_vocab()
        for symbol in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            token_id = vocabulary.get(symbol)
            if token_id is None or token_id in special_ids or token_id == unk_id:
                continue
            encoded = tokenizer.encode(symbol, add_special_tokens=False)
            if encoded == [token_id]:
                residue_token_ids[symbol] = token_id
        leading_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
        if (
            tokenizer.mask_token_id is None
            or tokenizer.eos_token_id is None
            or leading_token_id is None
            or tokenizer.pad_token_id is None
            or not residue_token_ids
        ):
            raise ProPlexError(
                f"tokenizer for {model_name} lacks required residue or special tokens"
            )
        return _TokenizerBundle(
            tokenizer=tokenizer,
            accepted_residues=frozenset(residue_token_ids),
            residue_token_ids=residue_token_ids,
        )
    except ProPlexError:
        raise
    except Exception as exc:
        raise ProPlexError(f"could not initialize tokenizer for {model_name}: {exc}") from exc


def _load_model(model_name: str, tokenizer_bundle: _TokenizerBundle | None = None) -> _ModelBundle:
    """Load ESM-2 weights after input has passed tokenizer-derived validation."""
    try:
        with _quiet_native_stderr():
            import torch
        from transformers import AutoModelForMaskedLM

        if tokenizer_bundle is None:
            tokenizer_bundle = _load_tokenizer(model_name)
        LOGGER.info("Loading ESM-2 model %s", model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)

        device = torch.device("cuda" if _silence_cuda_probe_stderr() else "cpu")
        model.eval()
        model.to(device)
        LOGGER.info("ESM-2 model ready on %s", device)
        return _ModelBundle(
            tokenizer=tokenizer_bundle.tokenizer,
            model=model,
            device=device,
            accepted_residues=tokenizer_bundle.accepted_residues,
            residue_token_ids=tokenizer_bundle.residue_token_ids,
        )
    except ProPlexError:
        raise
    except Exception as exc:
        raise ProPlexError(f"could not initialize ESM-2 model {model_name}: {exc}") from exc


def _prepare_sequence(sequence: str, *, trim: bool) -> str:
    if not isinstance(sequence, str):
        raise ProPlexError("sequence must be provided as text")
    normalized = (sequence.strip() if trim else sequence).upper()
    if not normalized:
        raise ProPlexError("sequence must not be empty")
    invalid = sorted(set(normalized) - set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    if invalid:
        symbols = ", ".join(repr(item) for item in invalid)
        raise ProPlexError(f"sequence contains invalid character(s): {symbols}")
    return normalized


def _normalize_sequence(sequence: str, accepted: frozenset[str]) -> str:
    invalid = sorted(set(sequence) - accepted)
    if invalid:
        symbols = ", ".join(repr(item) for item in invalid)
        raise ProPlexError(f"sequence contains unsupported residue symbol(s): {symbols}")
    return sequence


def _top_amino_acid_candidates(
    log_probs: Any, candidate_items: list[tuple[str, int]]
) -> list[tuple[str, float]]:
    """Select the best amino-acid token, breaking equal scores alphabetically."""
    import torch

    candidate_items = sorted(candidate_items, key=lambda item: item[0])
    candidate_token_ids = [token_id for _, token_id in candidate_items]
    candidate_ids = torch.tensor(candidate_token_ids, dtype=torch.long, device=log_probs.device)
    candidate_log_probs = log_probs.index_select(-1, candidate_ids)
    best_index = candidate_log_probs.argmax(dim=-1)
    best_log_probs = candidate_log_probs.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    best_indexes = best_index.detach().cpu().tolist()
    best_probabilities = best_log_probs.exp().detach().cpu().tolist()
    return [
        (candidate_items[index][0], float(probability))
        for index, probability in zip(best_indexes, best_probabilities)
    ]


def _score_normalized(
    sequence: str,
    bundle: _ModelBundle,
    model_name: str,
    sequence_id: str | None,
) -> SequenceResult:
    import torch

    windows = plan_windows(len(sequence))
    targets: list[tuple[int, str, int, int]] = []
    for position, residue in enumerate(sequence, start=1):
        window_index = select_window(position, windows)
        start, end = windows[window_index]
        targets.append((position, residue, start, end))

    log_probabilities: list[float] = []
    top_candidates: list[tuple[str, float]] = []
    candidate_items = sorted(bundle.residue_token_ids.items())
    with warnings.catch_warnings(), torch.inference_mode():
        warnings.filterwarnings(
            "ignore",
            message="Ignoring invalid value for boolean flag AMD_SERIALIZE_KERNEL.*",
        )
        for offset in range(0, len(targets), INFERENCE_BATCH_SIZE):
            batch = targets[offset : offset + INFERENCE_BATCH_SIZE]
            encoded_rows: list[list[int]] = []
            target_token_positions: list[int] = []
            target_ids: list[int] = []
            for position, residue, start, end in batch:
                window_sequence = sequence[start - 1 : end]
                local_position = position - start
                residue_ids = [bundle.residue_token_ids[char] for char in window_sequence]
                if len(residue_ids) != len(window_sequence):
                    raise ProPlexError("tokenizer did not produce one token per sequence residue")
                masked_ids = list(residue_ids)
                masked_ids[local_position] = bundle.tokenizer.mask_token_id
                leading_token_id = getattr(bundle.tokenizer, "bos_token_id", None)
                if leading_token_id is None:
                    leading_token_id = getattr(bundle.tokenizer, "cls_token_id", None)
                trailing_token_id = getattr(bundle.tokenizer, "eos_token_id", None)
                if leading_token_id is None or trailing_token_id is None:
                    raise ProPlexError("tokenizer must provide leading and trailing special tokens")
                encoded_rows.append([leading_token_id, *masked_ids, trailing_token_id])
                target_token_positions.append(local_position + 1)
                target_ids.append(bundle.residue_token_ids[residue])

            width = max(map(len, encoded_rows))
            pad_id = bundle.tokenizer.pad_token_id
            if pad_id is None:
                raise ProPlexError("tokenizer has no padding token required for batched inference")
            input_ids = torch.full(
                (len(encoded_rows), width), pad_id, dtype=torch.long, device=bundle.device
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, ids in enumerate(encoded_rows):
                input_ids[row_index, : len(ids)] = torch.tensor(
                    ids, dtype=torch.long, device=bundle.device
                )
                attention_mask[row_index, : len(ids)] = 1

            try:
                logits = bundle.model(input_ids=input_ids, attention_mask=attention_mask).logits
            except Exception as exc:
                raise ProPlexError(f"ESM-2 inference failed: {exc}") from exc
            rows = torch.arange(len(batch), device=bundle.device)
            positions = torch.tensor(target_token_positions, device=bundle.device)
            target_tensor = torch.tensor(target_ids, device=bundle.device)
            log_probs = torch.log_softmax(logits[rows, positions].float(), dim=-1)
            selected = log_probs[rows, target_tensor].detach().cpu().tolist()
            log_probabilities.extend(float(value) for value in selected)
            top_candidates.extend(_top_amino_acid_candidates(log_probs, candidate_items))
            completed = min(offset + len(batch), len(targets))
            LOGGER.info(
                "Residue scoring progress: %s / %s residues",
                f"{completed:,}",
                f"{len(targets):,}",
            )

    if len(log_probabilities) != len(sequence):
        raise ProPlexError("inference did not return exactly one probability per residue")
    mean_log_probability = math.fsum(log_probabilities) / len(sequence)
    scores = tuple(
        ResidueScore(
            position=position,
            residue=residue,
            log_probability=log_probability,
            probability=math.exp(log_probability),
            top_candidate=top_candidates[position - 1][0],
            top_candidate_probability=top_candidates[position - 1][1],
        )
        for position, (residue, log_probability) in enumerate(
            zip(sequence, log_probabilities), start=1
        )
    )
    return SequenceResult(
        sequence=sequence,
        sequence_id=sequence_id,
        model_name=model_name,
        length=len(sequence),
        mean_log_probability=mean_log_probability,
        pseudo_perplexity=math.exp(-mean_log_probability),
        psn=math.exp(mean_log_probability),
        residue_scores=scores,
    )


def score_sequence(
    sequence: str,
    *,
    model_name: str = DEFAULT_MODEL,
    sequence_id: str | None = None,
) -> SequenceResult:
    """Score one amino-acid sequence with ESM-2 masked residue predictions."""
    normalized = _prepare_sequence(sequence, trim=True)
    # The tokenizer defines rare/ambiguous residue support; load it before weights,
    # then reject unsupported letters without initializing the model.
    tokenizer_bundle = _load_tokenizer(model_name)
    _normalize_sequence(normalized, tokenizer_bundle.accepted_residues)
    bundle = _load_model(model_name, tokenizer_bundle)
    return _score_normalized(normalized, bundle, model_name, sequence_id)


def score_fasta(
    path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
) -> list[SequenceResult]:
    """Score each record in a FASTA file independently."""
    records = parse_fasta(path)
    prepared_records = [
        (record_id, _prepare_sequence(sequence, trim=False)) for record_id, sequence in records
    ]
    tokenizer_bundle = _load_tokenizer(model_name)
    for _, sequence in prepared_records:
        _normalize_sequence(sequence, tokenizer_bundle.accepted_residues)
    bundle = _load_model(model_name, tokenizer_bundle)
    return [
        _score_normalized(sequence, bundle, model_name, record_id)
        for record_id, sequence in prepared_records
    ]
