# ProPlex

**Protein sequence naturalness assessment using protein language model pseudo-perplexity.**

ProPlex quantifies how consistent a protein sequence is with the statistical patterns learned by a protein language model (PLM) trained on naturally occurring protein sequences. It requires only a query amino acid sequence and a pretrained masked protein language model, such as ESM-2. No sequence alignment, reference protein, functional annotation, or additional model training is required.

## Scientific motivation

Naturally occurring protein sequences are shaped by mutation, natural selection, genetic drift, and other evolutionary processes. These processes leave statistical signatures in amino acid composition, residue ordering, structural constraints, and functional motifs.

Protein language models learn these patterns from large protein sequence databases. They therefore implicitly encode information reflecting evolutionary constraints, without explicitly reconstructing evolutionary history.

ProPlex asks a simple question: **How consistent is a query sequence with the protein sequence patterns learned by a PLM?**

A naturally occurring protein may be more predictable in its sequence context than a randomly generated or shuffled sequence. Pseudo-perplexity provides a quantitative way to test this hypothesis. It measures sequence plausibility under the model, not the probability of natural origin, evolutionary fitness, or evolutionary distance.

## Method

Given a protein sequence $S=(a_1,a_2,\ldots,a_L)$ of length $L$, mask each residue individually and use the PLM to predict the probability of the original amino acid given the remaining sequence context:

$$
p_i=P_\theta(a_i\mid S_{\setminus i})
$$

Here, $S_{\setminus i}$ denotes the sequence with position $i$ replaced by a mask token, and $\theta$ denotes the pretrained model parameters.

### 1. Pseudo-likelihood

Combine the conditional residue probabilities:

$$
PL(S)=\prod_{i=1}^{L}p_i
$$

This is a pseudo-likelihood, not a normalized joint probability of the sequence. The product typically decreases with sequence length, making direct comparisons between different lengths difficult.

### 2. Length-normalized probability

Take the geometric mean of the residue probabilities:

$$
G(S)=\left(\prod_{i=1}^{L}p_i\right)^{1/L}
$$

This removes the direct multiplicative dependence on sequence length. For numerical stability, compute it in log space:

$$
\ell(S)=\frac{1}{L}\sum_{i=1}^{L}\ln p_i
$$

$$
G(S)=\exp(\ell(S))
$$

### 3. Pseudo-perplexity

Pseudo-perplexity (PPPL) is the reciprocal of the geometric mean probability:

$$
\boxed{\operatorname{PPPL}(S)=\exp\left(-\frac{1}{L}\sum_{i=1}^{L}\ln p_i\right)=\frac{1}{G(S)}}
$$

Lower pseudo-perplexity indicates greater consistency with learned protein sequence patterns.

### 4. Protein Sequence Naturalness score

For intuitive interpretation, ProPlex proposes reporting the reciprocal of pseudo-perplexity as a Protein Sequence Naturalness (PSN) score:

$$
\boxed{\operatorname{PSN}(S)=\exp\left(\frac{1}{L}\sum_{i=1}^{L}\ln p_i\right)=\frac{1}{\operatorname{PPPL}(S)}}
$$

PSN is bounded between 0 and 1, with higher values indicating greater contextual predictability. It is a proposed reporting name for the geometric mean probability, not a new mathematical quantity or a calibrated probability of natural origin.

## Worked example

Consider the illustrative sequence `MKLAG`. Mask each position independently and retrieve the probability of the original residue from the model's softmax distribution.

| Position | Masked sequence | Target | Probability | Natural log probability |
|---:|---|---|---:|---:|
| 1 | `[MASK]KLAG` | M | 0.80 | -0.2231 |
| 2 | `M[MASK]LAG` | K | 0.60 | -0.5108 |
| 3 | `MK[MASK]AG` | L | 0.90 | -0.1054 |
| 4 | `MKL[MASK]G` | A | 0.40 | -0.9163 |
| 5 | `MKLA[MASK]` | G | 0.70 | -0.3567 |

These probabilities are hypothetical, not actual ESM-2 predictions.

**Pseudo-likelihood:**

$$
PL=0.8\times0.6\times0.9\times0.4\times0.7=0.12096
$$

**Mean log probability:**

$$
\ell=\frac{-0.2231-0.5108-0.1054-0.9163-0.3567}{5}\approx-0.4225
$$

**Protein Sequence Naturalness:**

$$
PSN=0.12096^{1/5}=e^{-0.4225}\approx0.6553
$$

**Pseudo-perplexity:**

$$
PPPL=e^{0.4225}=\frac{1}{0.6553}\approx1.526
$$

The arithmetic mean probability is 0.68. Unlike the arithmetic mean, the geometric mean penalizes very low residue probabilities more strongly.

## Interpretation

| PSN | PPPL | Interpretation |
|---:|---:|---|
| 1.00 | 1 | Perfect contextual prediction |
| 0.50 | 2 | High observed-residue probability |
| 0.25 | 4 | Lower observed-residue probability |
| 0.10 | 10 | Low observed-residue probability |
| 0.05 | 20 | Uniform 20-amino-acid prediction baseline |
| 0.01 | 100 | Very low observed-residue probability |

These are mathematical reference points, not empirically established thresholds. The uniform baseline assumes a distribution restricted to 20 canonical amino acids; a model's full vocabulary may include additional tokens. PPPL can exceed 20 when observed residues receive probabilities below 0.05.

A PSN of 0.6553 means that the geometric mean of the conditional residue probabilities is 0.6553, not that the sequence has a 65.53% probability of being natural.

## Model selection

ESM-2 is the recommended starting point because its pretrained masked language modeling (MLM) head directly predicts missing amino acids from bidirectional sequence context. No additional training or embedding-based classifier is required.

| Model | Hugging Face checkpoint (prefix `facebook/`) | Layers | Hidden dimensions | Suggested use |
|---|---|---:|---:|---|
| ESM-2 8M | `esm2_t6_8M_UR50D` | 6 | 320 | Lightweight baseline |
| ESM-2 35M | `esm2_t12_35M_UR50D` | 12 | 480 | High-throughput screening |
| **ESM-2 150M** | `esm2_t30_150M_UR50D` | 30 | 640 | **Initial default** |
| ESM-2 650M | `esm2_t33_650M_UR50D` | 33 | 1280 | Higher-capacity comparison |
| ESM-2 3B | `esm2_t36_3B_UR50D` | 36 | 2560 | Optional research benchmark |
| ESM-2 15B | `esm2_t48_15B_UR50D` | 48 | 5120 | Optional research benchmark |

Start with the 150M model as a compromise between capacity and inference cost. Benchmark 35M, 150M, and 650M on natural and composition-matched shuffled sequences before selecting a production default. Larger models may predict natural residues better without necessarily improving natural-versus-random discrimination. Scores should be calibrated separately for each checkpoint.

### How the model computes residue probabilities

For `MK<mask>AG`, the transformer contextualizes the masked token using self-attention over the available sequence. The pretrained MLM head transforms the resulting hidden representation into one logit per vocabulary token. Softmax converts these logits into conditional probabilities:

$$
p_i=\frac{\exp(z_{a_i})}{\sum_{k\in V}\exp(z_k)}
$$

Here, $V$ is the model vocabulary and $z_{a_i}$ is the logit for the original residue. ProPlex uses the full vocabulary for normalization, retaining the model's native probabilities rather than renormalizing over only the 20 canonical amino acids. Special tokens are excluded from the positions being scored, not from the softmax denominator.

## Proposed implementation

1. Load a pretrained masked PLM and tokenize the query sequence.
2. Construct one masked input per amino acid position, preserving all other residues.
3. Evaluate masked inputs in batches and extract the log-softmax probability of each original residue at its masked position.
4. Average the log probabilities over actual amino acid positions, excluding special and padding tokens.
5. Report PPPL, PSN, and optionally residue-level log probabilities.

Exact scoring requires $L$ masked examples, which can be batched to improve throughput. For long sequences exceeding the model context limit, a windowing policy must be defined and documented because restricted context changes the score. Approximate single-pass scoring may be explored separately and benchmarked against exact scoring.

### Reference implementation (Python / PyTorch)

Install the dependencies:

```bash
pip install torch transformers
```

The following example calculates exact masked pseudo-perplexity using ESM-2 150M. Each batch contains independently masked copies of the same sequence, with exactly one residue masked per copy. GPU acceleration is used when CUDA is available (including compatible ROCm PyTorch builds).

```python
import math
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

MODEL = "facebook/esm2_t30_150M_UR50D"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AA = set("ACDEFGHIKLMNPQRSTVWY")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = EsmForMaskedLM.from_pretrained(MODEL).to(DEVICE).eval()


@torch.inference_mode()
def score_sequence(sequence, batch_size=8):
    sequence = sequence.strip().upper()
    if not sequence or set(sequence) - AA:
        raise ValueError("Expected a nonempty canonical amino acid sequence")
    if len(sequence) > 1022:
        raise ValueError("Reference implementation supports up to 1022 residues")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    inputs = tokenizer(sequence, return_tensors="pt",
                       return_special_tokens_mask=True)
    tokens = inputs["input_ids"].to(DEVICE)
    attention = inputs["attention_mask"].to(DEVICE)
    positions = torch.where(inputs["special_tokens_mask"][0] == 0)[0].to(DEVICE)
    if positions.numel() != len(sequence):
        raise ValueError("Unexpected tokenization")

    log_probs = []
    for pos in positions.split(batch_size):
        n = pos.numel()
        rows = torch.arange(n, device=DEVICE)
        masked = tokens.repeat(n, 1)
        targets = masked[rows, pos].clone()
        masked[rows, pos] = tokenizer.mask_token_id

        logits = model(input_ids=masked,
                       attention_mask=attention.repeat(n, 1)).logits
        masked_logits = logits[rows, pos].float()
        log_p = torch.log_softmax(masked_logits, dim=-1)[rows, targets]
        log_probs.extend(log_p.cpu().tolist())

    mean_log_p = math.fsum(log_probs) / len(sequence)
    return {
        "model": MODEL,
        "length": len(sequence),
        "mean_log_probability": mean_log_p,
        "pseudo_perplexity": math.exp(-mean_log_p),
        "psn": math.exp(mean_log_p),
        "residue_probabilities": [math.exp(x) for x in log_probs],
        "residue_surprisal": [-x for x in log_probs],
    }


if __name__ == "__main__":
    result = score_sequence("MKLAG")
    for key in ("model", "length", "mean_log_probability",
                "pseudo_perplexity", "psn"):
        print(f"{key}: {result[key]}")
    for i, (aa, prob) in enumerate(
        zip("MKLAG", result["residue_probabilities"]), start=1
    ):
        print(f"{i}\t{aa}\t{prob:.6f}")
```

The example uses a conservative 1022-residue limit (1024 tokens including boundary tokens), rejects ambiguous residues, and does not truncate sequences silently. Longer sequences require a documented windowing strategy. Reduce `batch_size` if GPU memory is insufficient. The code is a reference example, not a tested production implementation; actual model predictions for `MKLAG` will differ from the hypothetical worked example above.

Exact scoring requires one masked example per residue, even when batched. For future acceleration, compare residue subsampling or the One Fell Swoop approximation against exact scores before adopting either method.

### Proposed command-line interface

```bash
proplex query --fasta query.fasta --output scores.json
```

Example output (illustrative):

```tsv
sequence_id	length	mean_log_probability	pseudo_perplexity	psn
example	5	-0.4225	1.526	0.6553
```

The interface is a design proposal, not an implemented command.

## Validation plan

Evaluate the ability of PPPL and PSN to distinguish natural proteins from increasingly challenging synthetic controls:

| Sequence class | Purpose |
|---|---|
| Natural Swiss-Prot sequences | Biological reference distribution |
| Uniform random sequences | Basic random baseline |
| Frequency-matched random sequences | Control global amino acid frequencies and length |
| Individually shuffled natural sequences | Preserve exact length and composition |
| Dipeptide-preserving shuffles | Preserve adjacent-residue statistics |
| Block-shuffled sequences | Test sensitivity to longer-range organization |
| Mutated natural sequences | Test sensitivity to sequence perturbation |
| Experimentally validated designed proteins | Test whether plausible artificial sequences score well |

Compare ESM-2 model sizes, score distributions, AUROC, and computational cost. Stratify by sequence length, composition, disorder, and protein family. Keep source sequences and their synthetic derivatives together in any data split used for calibration or evaluation. Use homology-separated evaluation where appropriate, and acknowledge possible overlap with PLM pretraining data.

## Applications

- Assess sequence plausibility before protein function classification.
- Flag atypical queries for additional investigation in PClass and PClass-Dark.
- Compare natural proteins with randomized, mutated, or designed sequences.
- Investigate relationships between sequence naturalness, functional specificity (GO information content), mean pairwise sequence similarity, and similarity-distribution entropy.
- Identify locally unexpected residues using residue-level prediction scores.

## Limitations

- The score measures model-dependent contextual predictability, not evolutionary distance, biological fitness, folding stability, or function directly.
- Natural proteins with unusual composition, intrinsic disorder, or rare sequence patterns may score poorly; artificial proteins and repetitive sequences may score well.
- Length normalization does not eliminate all length-dependent effects.
- Scores from different models, tokenization conventions, or context-window policies are not necessarily comparable.
- PSN and PPPL are reciprocal transformations and therefore provide equivalent rankings, not independent evidence.
- Low naturalness should not be used as an automatic rejection criterion for novel proteins.

## References

- Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*. https://doi.org/10.1126/science.ade2574
- Salazar et al. (2020). Masked Language Model Scoring. *Proceedings of ACL*. https://aclanthology.org/2020.acl-main.240/
- Kantroo et al. Pseudo-perplexity in One Fell Swoop for Protein Fitness Estimation. https://arxiv.org/abs/2407.07265
- ESM-2 150M checkpoint: https://huggingface.co/facebook/esm2_t30_150M_UR50D
- ESM model repository: https://github.com/facebookresearch/esm

## Project status

ProPlex is in the initial design stage. The mathematical scoring method is defined above; implementation, benchmarking, calibration, and empirical interpretation remain to be completed.
