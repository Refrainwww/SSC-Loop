# SSC-Loop

Official implementation for **Signed-Graph Recommendation as Structural Consistency Maximization**.

SSC-Loop jointly addresses three forms of consistency in signed social recommendation:

- **Structural consistency:** ESA-DA edits the signed graph using model-derived confidence, degree/connectivity guards, and sparse structural-balance checks.
- **Propagation consistency:** P/N/O propagation keeps positive, negative, and ambiguous social signals in separate channels.
- **Semantic consistency:** a signed contrastive objective aligns user representations with trust and distrust relations.

The training procedure alternates graph refinement and representation learning, with validation-based early stopping.

## Repository layout

```text
train.py              Main Epinions training and ablation entry point
models.py             SSC-Loop model, P/N/O propagation, edge scoring, losses
modules/esa_da.py     ESA-DA graph refinement and structural guards
data.py               Epinions parser and deterministic 70/10/20 split
slashdot_data.py      Auxiliary Slashdot data loader
baselines.py          Graph-based recommendation baselines
sigformer.py          Sign-aware baseline implementation
metrics.py            Rating and ranking metrics
scripts/smoke_test.py Synthetic end-to-end sanity check (no dataset required)
```

Training outputs, checkpoints, generated candidate edges, logs, and datasets are intentionally excluded.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy
- SciPy
- tqdm
- Optional: `faiss-cpu` or `faiss-gpu` for faster kNN candidate generation

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation may be adjusted for the CUDA version on your machine. Without FAISS, ESA-DA uses a blockwise PyTorch kNN fallback.

## Data preparation

Datasets are **not** distributed in this repository. Obtain Epinions or Slashdot from their original providers and follow their terms of use.

The Epinions interaction file is whitespace-separated with three columns:

```text
user_id item_id rating
1 10 5
1 18 3
```

The signed social-network file is whitespace-separated and uses signs in `{+1, -1}`:

```text
source_id target_id sign
1 2 1
1 3 -1
```

Headers are optional. User and item identifiers must be integers. The loader remaps them to contiguous internal indices and deterministically creates a 70%/10%/20% train/validation/test split for each seed.

Suggested local layout (ignored by Git):

```text
data/
  epinions/
    epinions.inter
    epinions.net
```

## Quick verification

Run the synthetic smoke test before using the full dataset:

```bash
python scripts/smoke_test.py
```

It builds a tiny signed recommendation problem in a temporary directory, evaluates the complete loss, and performs one ESA-DA refinement step.

## Training SSC-Loop

```bash
python train.py \
  --inter_path data/epinions/epinions.inter \
  --net_path data/epinions/epinions.net \
  --ckpt_dir runs/epinions/full \
  --device cuda:0 \
  --outer_loops 3 \
  --inner_epochs 5 \
  --patience 5 \
  --alpha 0.5 \
  --repeat 5
```

For CPU execution, use `--device cpu`. Exact kNN construction on a large graph is expensive; installing FAISS is strongly recommended for full-scale Epinions experiments.

Important graph-refinement arguments:

| Argument | Default | Meaning |
|---|---:|---|
| `--delete_ratio` | 0.05 | Fraction of low-confidence observed edges considered for deletion |
| `--add_ratio` | 0.05 | Edge-addition budget relative to the current edge count |
| `--knn_k` | 200 | Embedding-neighborhood size for candidate generation |
| `--top_k_candidate` | 20000 | Retained candidate count per polarity |
| `--d_pos_max` / `--d_neg_max` | 50 | Positive/negative degree caps |
| `--delta_min` | 1 | Minimum degree preserved by deletion |
| `--balance_guard` | 1 | Enable sparse structural-balance filtering |
| `--tau` | 1.0 | Temperature used for polarity confidence |

Run `python train.py --help` for the complete option list.

## Ablations

Use the same command with one of the following values:

```bash
--ablation full
--ablation no_esa_da
--ablation no_pno
--ablation no_contrastive
--ablation one_shot
```

- `no_esa_da`: keeps the observed graph fixed.
- `no_pno`: replaces P/N/O propagation with unsigned social propagation.
- `no_contrastive`: removes signed contrastive alignment.
- `one_shot`: performs one graph-refinement round instead of closed-loop refinement.

## Reproducibility notes

- Validation data are used only for early stopping and checkpoint selection.
- Test metrics are computed after restoring the best validation checkpoint.
- `--seed` controls data shuffling, initialization, and sampled contrastive edges.
- `--repeat N` uses seeds `seed, seed+1, ..., seed+N-1`.
- Generated checkpoints are written under `runs/`, which is ignored by Git.

## Citation

If this repository is useful in your research, please cite the accompanying paper. Bibliographic venue information can be added to `CITATION.cff` after publication.

## Contact

Zifan Wang, Northeast Normal University.
