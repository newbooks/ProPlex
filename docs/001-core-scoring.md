# 001: ProPlex core scoring and CLI implementation

Status: decisions captured; implementation pending.
Prerequisites: none.

## 1. Goal and scope

ProPlex assesses how contextually predictable a protein sequence is under a
pretrained masked protein language model (PLM). Its first implementation uses
ESM-2 and reports pseudo-perplexity (PPPL) and Protein Sequence Naturalness
(PSN). The score describes model-dependent sequence predictability. It is not a
calibrated probability that a sequence is natural, nor a measure of fitness,
function, or evolutionary distance.

The code sample in the repository README is explanatory only. It is not
production code or a required implementation structure. Implement the behavior
specified here in production-quality code and tests.

## 2. Model and scoring method

1. Default to ESM-2 150M, checkpoint
   `facebook/esm2_t30_150M_UR50D`.
2. For every residue in the input sequence, create an input where that residue
   is replaced by the model's mask token and all other sequence residues are
   unchanged.
3. Use the masked language modeling head to obtain the probability of the
   original residue at the masked position. Use the model's full vocabulary
   softmax; do not renormalize over only the 20 canonical amino acids.
4. For residue probabilities `p_i` over `L` input residues, compute
   `mean_log_probability = (1/L) * sum(log(p_i))`,
   `PPPL = exp(-mean_log_probability)`, and
   `PSN = exp(mean_log_probability) = 1/PPPL`.
5. Exclude special and padding tokens from the set of scored residues. Special
   tokens remain in the model input and the model's vocabulary normalization.
6. Report PPPL and PSN for each sequence. Preserve the mean log probability as
   useful output metadata. Verbose mode also reports per-residue probabilities
   using a text graph; graph design is intentionally deferred.

## 3. Input sequence policy

1. Accept sequence symbols according to the residue alphabet supported by the
   selected ESM-2 tokenizer/model. Validate input before inference and give a
   useful error for unsupported symbols. Do not copy the README sample's
   hard-coded residue policy without confirming it against ESM-2.
2. The CLI accepts either a sequence string via `--seq` or sequence record(s)
   from a FASTA file via `--fasta`.
3. FASTA files may contain multiple records; produce one result per record and
   retain each record's identifier in output.

## 4. Windowing for long sequences

Follow the overlapping-window policy used by ProStruct in
`src/prostruct/esm2.py` (`WINDOW_LENGTH`, `MINIMUM_OVERLAP`, `plan_windows`,
and `select_window`):

1. A window contains at most 1,021 sequence residues. Sequences of length at
   most 1,021 use one complete window.
2. Longer sequences use the minimum number of exactly 1,021-residue windows
   that cover the full sequence with at least 100 residues of overlap between
   each adjacent pair. Anchor the first window at the N terminus and the last
   at the C terminus; distribute intermediate windows as evenly as integer
   coordinates allow. ProStruct computes the count as
   `ceil((L - 100) / (1021 - 100))`.
3. A residue covered by multiple windows is evaluated in the window where it
   is closest to that window's center. Break equal-distance ties in favor of
   the window with the smaller start coordinate. This keeps the residue of
   interest away from a window edge when possible.
4. Every source residue must receive exactly one score, from its selected
   window. Do not average or combine duplicate predictions from overlapping
   windows.
5. The 1,021 limit counts sequence residues. Tokenization must also fit the
   model's context limit after adding special tokens. Do not silently truncate.

The window policy originates in ProStruct's residue-embedding workflow. In
ProPlex, apply it to the masked prediction for each target residue.

## 5. ESM-2 token alignment and special tokens

Follow ProStruct's token-alignment safeguards:

1. Tokenize without truncation. Require exactly one token per input residue,
   with exactly one leading and one trailing special token.
2. In padded batches, verify right-padding and use the attention mask so padding
   cannot be mistaken for residues.
3. Map sequence residue index `j` to its aligned token position after accounting
   for the leading special token. Score only the masked target position.
4. Never score special tokens or padding as sequence residues. Do not shift or
   infer alignment if token counts differ; fail with a clear error.

## 6. Initial command-line interface

The installed top-level command is `proplex`, implemented with Typer using a
root application that can support subcommands as the project grows. The initial
interface decisions are:

1. `proplex --seq SEQUENCE` scores a sequence supplied on the command line.
2. `proplex --fasta PATH` scores sequence record(s) from a FASTA file.
3. `proplex --help` and `proplex -h` show help. Invoking `proplex` with no
   arguments also shows help by default.
4. `proplex --verbose` and `proplex -v` include residue-level probabilities
   using a text graph. The graph's appearance and scale are not decided yet;
   leave the detailed graph design for a later decision.
5. Treat `--seq` and `--fasta` as mutually exclusive input modes.

## 7. Decisions still open

1. Define the verbose residue-probability text graph.
2. Define final result formatting and output-file support, if needed.
3. Decide whether CPU/GPU selection needs a user-facing option; no such option
   is currently specified.
4. Pin or otherwise define model/tokenizer revisions if reproducible scoring
   across model updates is required.

