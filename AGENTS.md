# Repository ground rules

## Branches

- Check the current branch before editing files.
- Do not edit files on `main`. Create or switch to a working branch before making changes.
- On branches other than `main`, proceed with work within the user's requested scope without asking for additional approval. This standing authorization does not override tool permissions or higher-priority safety requirements.

## Tests

- Write pytest tests when implementing new code or changing existing code behavior.
- After implementing code, run the relevant pytest tests and fix any failures caused by the changes.
- Report the test results, including any tests that could not be run and the reason.

## Implementation specifications

- Give each new implementation specification a sequential number in its filename and title, for example `docs/001-cli.md` with the title `001: CLI implementation`.
- Choose the next unused number in implementation order. Keep assigned numbers stable so they can be referenced later; do not renumber existing specifications.
- Number implementation steps within each specification and state any prerequisite specifications so the implementation sequence is clear.
- Maintain a numbered specification index in `docs/README.md` when adding specifications, with links and implementation status.
- Apply this convention to new specifications going forward; existing unnumbered documents may remain as references.
