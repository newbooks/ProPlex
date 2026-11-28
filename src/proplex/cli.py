"""Typer command line interface for ProPlex."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from proplex.scoring import ProPlexError, SequenceResult, score_fasta, score_sequence

LOG_FORMAT = "[%(levelname)s]: %(message)s"
LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    name="proplex",
    help="Assess protein sequence predictability with a protein language model.",
    invoke_without_command=True,
    add_completion=False,
    context_settings={"help_option_names": ["--help", "-h"]},
)


def _default_output_path(fasta: Path | None) -> Path:
    if fasta is not None:
        return fasta.with_name(f"{fasta.stem}.proplex.json")
    return Path("proplex-results.json")


def _write_results(path: Path, results: list[SequenceResult]) -> None:
    payload = {
        "schema_version": 1,
        "results": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ProPlexError(f"could not write result JSON to {path}: {exc}") from exc


def render_residue_dots(result: SequenceResult) -> str:
    """Render native and top-candidate amino-acid probabilities per position."""
    lines = [
        f">{result.sequence_id or 'sequence'}",
        "Position  AA Probability  Percent    Candidate   Probability",
    ]
    for score in result.residue_scores:
        dot_count = min(10, int(score.probability * 10))
        dots = "." * dot_count or " "
        lines.append(
            f"{score.position:8d}  {score.residue}   {dots:<10}  "
            f"{score.probability * 100:6.2f}%     {score.top_candidate:<9} "
            f"{score.top_candidate_probability * 100:6.2f}%"
        )
    return "\n".join(lines)


@app.callback()
def main(
    ctx: typer.Context,
    seq: Annotated[str | None, typer.Option("--seq", help="Protein sequence string.")] = None,
    fasta: Annotated[Path | None, typer.Option("--fasta", help="FASTA input file.")] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON result path; defaults beside the FASTA or to ./proplex-results.json.",
        ),
    ] = None,
) -> None:
    """Score a sequence supplied with --seq or records supplied with --fasta."""
    if ctx.invoked_subcommand is None and seq is None and fasta is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)
    if seq is not None and fasta is not None:
        typer.echo("Options --seq and --fasta are mutually exclusive.", err=True)
        raise typer.Exit(code=2)
    if seq is None and fasta is None:
        typer.echo("Option --seq or --fasta is required.", err=True)
        raise typer.Exit(code=2)

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    result_path = output or _default_output_path(fasta)
    try:
        LOGGER.info("Starting ProPlex scoring")
        results = [score_sequence(seq)] if seq is not None else score_fasta(fasta)  # type: ignore[arg-type]
        _write_results(result_path, results)
    except ProPlexError as exc:
        LOGGER.error("%s", exc)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        LOGGER.error("Scoring failed: %s", exc)
        raise typer.Exit(code=1) from exc

    typer.echo("sequence_id\tlength\tmean_log_probability\tpseudo_perplexity\tpsn")
    for result in results:
        sequence_id = result.sequence_id or "sequence"
        typer.echo(
            f"{sequence_id}\t{result.length}\t{result.mean_log_probability:.8g}\t"
            f"{result.pseudo_perplexity:.8g}\t{result.psn:.8g}"
        )
        typer.echo(render_residue_dots(result))
    LOGGER.info("Wrote %s result(s) to %s", len(results), result_path)
