# 002: Typer command and importable API entry point

Status: implementation spec; initial CLI shell only.
Prerequisite: [001: ProPlex core scoring and CLI implementation](001-core-scoring.md).

## 1. Goal

Provide a usable `proplex` command and an importable Python package/API. This
step establishes the public entry points and help text. It does not implement
sequence parsing, model loading, scoring, or residue-level visualization.

## 2. Package and public entry points

1. Create the package under `src/proplex/` and expose the Typer application as
   `proplex.cli:app` for the existing `[project.scripts]` entry in
   `pyproject.toml`.
2. Make `proplex` importable as a Python module/package.
3. Define and export a public `score_sequence` function from `proplex`, so users
   can call it directly as `from proplex import score_sequence`. Also make its
   implementation module importable. The function may be a documented stub in
   this step; the scoring contract is defined in spec 001 and should be
   implemented in a later step.
4. Keep CLI wiring thin and call the same public library API that Python users
   will call once scoring is implemented. Do not put the scoring algorithm in
   the Typer command callback.

## 3. Initial command behavior

1. `proplex --help` and `proplex -h` display root help and exit successfully.
2. Running `proplex` with no arguments displays the same help by default and
   exits successfully. It must not try to load ESM-2, read input, or start
   scoring.
3. The root help describes ProPlex and lists the planned initial options:
   - `--seq SEQUENCE`: sequence string input.
   - `--fasta PATH`: FASTA file input, potentially containing multiple records.
   - `--verbose`, `-v`: request residue-level probabilities and a text graph.
4. Register the options with useful help text, but leave their scoring and
   output behavior as explicit placeholders. Invoking an input or verbose
   option must not silently claim to have scored a sequence. If placeholder
   invocation is handled, report clearly that the feature is not implemented
   and exit nonzero.
5. `--seq` and `--fasta` are mutually exclusive once input handling is
   implemented. The option names and semantics should be established in the
   CLI now; input validation is out of scope for this spec.
6. Preserve Typer's normal help behavior at the root. No subcommands are
   required yet; use a Typer root application that can accept sub-applications
   later.

## 4. Help content

Root help must communicate that ProPlex evaluates protein sequence
predictability with a protein language model, and show:

```text
proplex --seq SEQUENCE
proplex --fasta PATH
proplex --verbose
```

Explain that verbose output is intended to include residue-level prediction
probabilities and a text graph. Do not imply that scoring or graph rendering
already works in this shell-only step. Typer may control exact formatting.

## 5. Out of scope

1. ESM-2 model/tokenizer loading, dependency setup, and inference.
2. Implementing `score_sequence`; it is a public API placeholder in this step.
3. Sequence alphabet validation, FASTA parsing, and long-sequence windowing.
4. PPPL/PSN calculations and final result formatting.
5. Detailed residue graph design and rendering.

## 6. Acceptance checks

1. `proplex --help` exits zero and displays the description and planned
   `--seq`, `--fasta`, and `--verbose` options.
2. `proplex -h` behaves equivalently.
3. `proplex` with no arguments exits zero and displays root help without model
   loading or inference.
4. `import proplex` succeeds, and `from proplex import score_sequence`
   resolves the documented public callable.
5. Placeholder input invocation never reports a successful score.
6. Write pytest coverage for the CLI help/default behavior and public import;
   run relevant pytest tests and report results per `AGENTS.md`.

