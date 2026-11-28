# 002: Typer command and importable API entry point

Status: implemented; later sequence-scoring behavior is specified in 003.
Prerequisite: [001: ProPlex core scoring and CLI implementation](001-core-scoring.md).

## 1. Goal

Provide a Typer root command and an importable Python package/API. The `query`
subcommand owns sequence scoring; the root remains available for future
commands such as `root`.

## 2. Package and public entry points

1. Create the package under `src/proplex/` and expose the Typer application as
   `proplex.cli:app` for the existing `[project.scripts]` entry in
   `pyproject.toml`.
2. Make `proplex` importable as a Python module/package.
3. Define and export a public `score_sequence` function from `proplex`, so users
   can call it directly as `from proplex import score_sequence`. Also make its
   implementation module importable. The function may be a documented stub in
   this step; the scoring implementation is defined in spec 003.
4. Keep CLI wiring thin and call the same public library API that Python users
   will call once scoring is implemented. Do not put the scoring algorithm in
   the Typer command callback.

## 3. Initial command behavior

1. `proplex --help` and `proplex -h` display root help and exit successfully.
2. Running `proplex` with no arguments displays root help by default and exits
   successfully. It must not try to load ESM-2, read input, or start scoring.
3. The root help describes ProPlex and lists the `query` subcommand. Query help
   describes the planned input/output options:
   - `--seq SEQUENCE`: sequence string input.
   - `--fasta PATH`: FASTA file input, potentially containing multiple records.
   - `--output PATH`, `-o`: JSON result destination; a default is selected
     based on the input when omitted.
4. `--seq` and `--fasta` are mutually exclusive. Scoring and JSON output are
   implemented as specified in 003.
5. Preserve help at both root and query levels. Bare `proplex` shows root
   help; bare `proplex query` shows query help. Neither loads model assets.

## 4. Help content

Root help must communicate that ProPlex evaluates protein sequence
predictability with a protein language model, and show:

```text
proplex query --seq SEQUENCE
proplex query --fasta PATH
proplex query --output results.json
```

Query help explains the sequence inputs and JSON output. Root help lists
subcommands and leaves room for future commands. Typer controls exact help
formatting.

## 5. Out of scope

Detailed scoring, input handling, and output semantics are specified in 003.

## 6. Acceptance checks

1. `proplex --help` exits zero and displays the root description and `query`
   subcommand.
2. `proplex -h` behaves equivalently; `proplex query --help` lists the
   `--seq`, `--fasta`, and `--output` options.
3. `proplex` with no arguments exits zero and displays root help; `proplex
   query` with no arguments exits zero and displays query help. Neither loads
   model weights or performs inference.
4. `import proplex` succeeds, and `from proplex import score_sequence`
   resolves the documented public callable.
5. Placeholder input invocation never reports a successful score.
6. Write pytest coverage for the CLI help/default behavior and public import;
   run relevant pytest tests and report results per `AGENTS.md`.

