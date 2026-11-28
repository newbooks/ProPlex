# 003: Sequence loading, window planning, and ESM-2 residue scoring

Status: implementation in progress; JSON persistence and console progress follow the updated contract below.
Prerequisites: [001: ProPlex core scoring and CLI implementation](001-core-scoring.md),
[002: Typer command and importable API entry point](002-cli-entrypoint.md).

## 1. Goal and scope

Implement the first usable scoring pipeline for a single sequence or FASTA
records: validate input, select ESM-2 windows, score each residue with the
masked language modeling head, and calculate PPPL and PSN. The README sample
code is explanatory only; it does not constrain production structure.

This specification uses “model inference” for the ESM-2 forward pass. ESM-2
internally computes contextual hidden representations for every masked input,
but ProPlex does not extract, return, or save those embeddings. Its target
quantity is the MLM head's log-probability for the original amino acid.

## 2. Public API and result contract

1. Provide `score_sequence(sequence: str, *, model_name: str =
   "facebook/esm2_t30_150M_UR50D") -> SequenceResult`.
   Export it from `proplex`; it must not require calling the CLI.
2. Define a documented result type with these fields:
   - `sequence`: normalized sequence string;
   - `sequence_id`: `None` for direct sequence input, FASTA record ID otherwise;
   - `model_name`: checkpoint identifier used;
   - `length`: number of input residues;
   - `mean_log_probability`, `pseudo_perplexity`, and `psn`: finite floats;
   - `residue_scores`: ordered per-residue records containing one-based
     `position`, `residue`, `log_probability`, `probability`,
     `top_candidate`, and `top_candidate_probability`.
3. Keep model/tokenizer loading lazy. Importing `proplex`, calling CLI help,
   parsing FASTA, or planning windows must not download or initialize model
   weights. Load the default model only when a valid scoring request is made.
4. Provide an internal model-loader/inference boundary that tests can replace
   with deterministic fake tokenizer/model objects. Do not require network
   access or model weights in unit tests.

## 3. Input handling

1. Normalize a sequence by trimming leading/trailing whitespace and converting
   letters to uppercase. Do not remove internal whitespace or punctuation;
   reject them as invalid input. Reject empty input.
2. Accept exactly the amino-acid residue symbols supported by the selected
   ESM-2 model's residue alphabet, including its supported ambiguous/rare
   residue symbols. Derive/validate the set against the model tokenizer's
   residue vocabulary, excluding special tokens and gap/padding symbols. Keep
   this policy in one validator and test every accepted symbol plus nearby
   invalid characters. Do not assume that successful tokenization alone means
   an input character is a valid residue.
3. For direct CLI input, `--seq` supplies one sequence. For FASTA input,
   `--fasta PATH` accepts one or more records and preserves the first
   whitespace-delimited token of each header as the record ID; empty headers,
   empty records, sequence text before the first header, and a file with no
   records are errors. Ignore blank lines between sequence lines; concatenate
   sequence lines within a record before sequence validation.
4. Require exactly one of `--seq` and `--fasta`. With no arguments, preserve
   the help behavior defined in spec 002.
5. Implement FASTA parsing with a small isolated parser and focused tests; do
   not add a dependency solely for this simple format unless implementation
   evidence requires one.

## 4. Window plan

Use the same deterministic policy as ProStruct's `plan_windows` and
`select_window` in `src/prostruct/esm2.py`:

1. The window length is **1,021 sequence residues**, not 1,021 total model
   tokens. A sequence with `L <= 1021` has one window `[1, L]`. Every window
   for `L > 1021` has exactly 1,021 sequence residues.
2. For `L > W`, use `W=1021`, minimum overlap `O=100`, and
   `K = ceil((L - O) / (W - O))` windows. In one-based inclusive coordinates,
   for `i = 0..K-1`, set
   `start_i = 1 + floor(i * (L - W) / (K - 1))`,
   `end_i = start_i + W - 1`. This anchors the first and last windows and
   distributes the intermediate starts evenly. Adjacent windows overlap by at
   least 100 residues.
3. For every source residue position `p` (one-based), consider all windows
   containing `p`. Select the one minimizing its distance to the window center,
   using `abs(2*p - (start + end))` as an integer distance measure. Break ties
   by smaller window start. A 1,021-residue window has a unique center residue.
4. The selected window determines the full context for that residue's masked
   prediction. Do not score the same target in every overlapping window, blend
   predictions, or count a target more than once.
5. Validate that windows are ordered, terminally anchored, full length when
   long, meet the overlap requirement, cover every source residue, and assign
   exactly one selected window to each residue. Fail clearly if any invariant
   is violated.

## 5. Tokenization and masked inference

1. For each target residue, slice its selected window from the normalized
   source sequence, replace only that target residue with the model's mask
   token, and preserve all other residues in the selected window.
2. Tokenize without truncation. Require exactly one token per sequence
   position, plus exactly one leading and one trailing special token. Check
   that the target residue maps to the expected token position after the
   leading special token. If token count or alignment is wrong, raise a clear
   inference/input error; do not guess, shift, or truncate.
3. Pass the masked token sequence to the selected ESM-2 masked language model
   (MLM) head. Read logits at the target's masked token position. Compute
   `log_softmax(logits, dim=-1)` across the model's complete vocabulary and
   select the original residue token's log-probability. Do not renormalize
   across only canonical amino acids. Special tokens stay in the model input
   and denominator but are never target residues; padding is excluded from
   inference/selection.
4. Use inference/evaluation mode and disable gradient tracking. Batch masked
   inputs to improve throughput, but bound batch size to avoid unbounded memory
   growth. Expose no new user-facing batch-size option in this implementation.
   Preserve deterministic output ordering independent of batch boundaries.
   Log completed/total target residues to the console after each inference
   batch using PClass's `[INFO]: message` style.
5. Store one log-probability for every original position, in sequence order.
   Compute the reported probability as `exp(log_probability)`; do not derive
   aggregate scores by multiplying residue probabilities. Also select the
   highest-probability amino-acid residue token at that masked position from
   the full-vocabulary log-softmax values. Store its symbol and its original
   (not amino-acid-renormalized) probability. Ties resolve to the first symbol
   in alphabetical order.

## 6. Aggregate scores and Python result

For the `L` ordered residue log-probabilities `lp_i`, compute:

```text
mean_log_probability = math.fsum(lp_i) / L
pseudo_perplexity = exp(-mean_log_probability)
psn = exp(mean_log_probability)
```

Return a `SequenceResult` satisfying the API fields in section 2. Preserve
full-precision floats in the API; formatting/rounding belongs only to the CLI.
For a FASTA record, associate its ID with the result without changing the
sequence scoring calculation. Each FASTA record is scored independently.

## 7. CLI behavior, progress, and JSON output

1. Keep the installed Typer root command `proplex`, implement scoring as
   `proplex query`, and keep the public `from proplex import score_sequence`
   API. The CLI must call the public scoring function, not duplicate inference
   logic. Leave the root open for future commands such as `proplex root`.
2. `proplex query --seq` and `proplex query --fasta` invoke the scoring pipeline for their respective
   inputs. Supplying both is an explicit usage error. Model/input/inference
   errors print a concise `[ERROR]:` message to stderr and exit nonzero; no
   routine traceback is printed.
3. Log startup, model loading/readiness, residue progress, and result
   publication to the console using PClass's compact format:
   `[INFO]: message`. Progress messages report completed and total residues
   after each inference batch.
4. Always write a JSON file containing all results, including every residue's
   position, observed symbol/probability/log-probability, and top candidate
   symbol/probability. The top-level JSON
   object contains `schema_version: 1` and a `results` array of complete
   `SequenceResult` objects. Write UTF-8, indented JSON with a final newline;
   publish through a temporary file and atomic replacement so interrupted
   writes do not leave a partial result at the destination.
5. Add `--output PATH` / `-o PATH` to choose the JSON destination. Without it,
   FASTA input writes `<input-stem>.proplex.json` beside the input file;
   `--seq` writes `proplex-results.json` in the current directory. Log the
   final destination. A write failure is an operational error and exits
   nonzero.
6. Standard output prints one summary row per input sequence with ID when
   present, length, mean log probability, PPPL, and PSN. For `--seq`, use
   `sequence` as the display ID. Also print one residue row with one-based
   position, observed amino-acid symbol, a dot bar, observed probability
   percentage, top candidate amino-acid symbol, and its probability percentage.
   Use ten dot levels for the observed residue: `floor(probability * 10)` dots,
   capped at ten. A score below 10% displays zero dots; 100% displays ten.
   Exact values remain in JSON.
7. Root and query `--help`/`-h` plus bare `proplex` and `proplex query` help
   must work without loading model weights.

## 8. Testing and acceptance criteria

Use pytest with fake tokenizer/model outputs for unit tests. No unit test may
download model weights or require GPU/network access. Cover at least:

1. Normalization and validation: trimming/case normalization, empty sequence,
   all supported ESM-2 residue symbols, and invalid whitespace/punctuation.
2. FASTA parsing: one and multiple records, multiline sequences, identifiers,
   blank lines, empty/malformed records, and empty files.
3. Window planning and target assignment: lengths 1, 1,021, 1,022, exact
   boundaries, long multi-window sequences, overlap, anchoring, complete
   coverage, center choice, tie-break, and exactly one selected window per
   residue.
4. Masking/token alignment: correct target masking, one token per residue,
   special-token offsets, original target token lookup, full-vocabulary
   normalization, and errors on truncation/misalignment.
5. Scoring: hand-computable fake logits, ordered residue outputs, `math.fsum`
   aggregation, PPPL/PSN reciprocal identity within floating-point tolerance,
   and no duplicate residue contributions from overlap.
6. Public API and CLI: import without model loading; API result fields;
   `proplex query --seq`, multi-record `proplex query --fasta`, JSON
   content/default/custom destinations, PClass-style
   progress messages, ten-level residue dot bars, candidate probabilities,
   mutually exclusive inputs, concise invalid-input/model-
   load/write errors, successful help and bare-command help without model
   loading, and absence of the removed `--verbose` option.
7. Run relevant pytest tests, fix regressions, and report results per
   `AGENTS.md`.

