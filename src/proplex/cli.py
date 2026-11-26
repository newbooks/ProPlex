"""Typer command line interface for ProPlex."""

from typing import Annotated
from pathlib import Path

import typer

app = typer.Typer(
    name="proplex",
    help="Assess protein sequence predictability with a protein language model.",
    invoke_without_command=True,
    add_completion=False,
    context_settings={"help_option_names": ["--help", "-h"]},
)


@app.callback()
def main(
    ctx: typer.Context,
    seq: Annotated[
        str | None,
        typer.Option("--seq", help="Protein sequence string (scoring is not implemented yet)."),
    ] = None,
    fasta: Annotated[
        Path | None,
        typer.Option("--fasta", help="FASTA input file (scoring is not implemented yet)."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Request residue-level probability graph output."),
    ] = False,
) -> None:
    """Score a sequence supplied with --seq or records supplied with --fasta."""
    if ctx.invoked_subcommand is None and seq is None and fasta is None and not verbose:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)
    if seq is not None and fasta is not None:
        typer.echo("Options --seq and --fasta are mutually exclusive.", err=True)
        raise typer.Exit(code=2)
    if seq is not None or fasta is not None or verbose:
        typer.echo(
            "This CLI is a placeholder: sequence scoring and verbose output are not implemented yet.",
            err=True,
        )
        raise typer.Exit(code=1)
