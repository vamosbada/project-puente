# Meta-learner re-evaluation (cross-validation)

The paper's Late Fusion result (67.15%) trains the MLP meta-learner on the validation-set logits
and evaluates it on the **same** validation set. This folder checks how much that inflates the number.

## ⚠️ Which run these files come from

The `.npy` files here are **not** from the 67.15% run. They come from
**Experiment 34** in [`docs/experiment_log.md`](../../docs/experiment_log.md)
(same architecture, XLM-R lr 2e-5, 6 epochs), whose original evaluation gave **65.71%**.

Both runs saved their logits to the same Google Drive folder (`late_fusion_results/`), so the
67.15% run's logits were overwritten by later runs and are no longer available.

| File | Contents |
|---|---|
| `mbert_val_logits.npy` | mBERT validation logits, shape (1114, 3) |
| `xlmr_val_logits.npy` | XLM-R validation logits, shape (1114, 3) |
| `val_labels.npy` | Validation labels (0 = negative, 1 = neutral, 2 = positive) |
| `train_indices.npy`, `val_indices.npy` | Row indices of the train/validation split |
| `fusion_eval.py` | Re-evaluation script |

## How to run

No GPU needed; runs in a few seconds.

```bash
pip install numpy scipy scikit-learn
python experiments/meta_learner_cv/fusion_eval.py experiments/meta_learner_cv
```

The meta-learner uses the same settings as the original notebook:
`MLPClassifier(hidden_layer_sizes=(6,), random_state=42, max_iter=1000)` on the raw concatenated
logits (no scaling).

## Results (Experiment 34 logits)

| Method | Accuracy |
|---|---|
| mBERT solo | 57.54% |
| XLM-R solo | 63.73% |
| 1) Original method — fit and evaluate on the same validation set | **65.71%** (matches the original notebook) |
| 2) 5-fold stratified CV (fold seed 42) | **65.17% ± 3.81%** |
| 3) Simple average of softmax probabilities (no training) | **62.12%** |

5-fold CV with other fold seeds (mean ± sample std):

| Fold seed | 42 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| CV accuracy | 65.17 ± 3.81 | 64.73 ± 3.27 | 64.81 ± 1.06 | 64.63 ± 2.04 | 65.26 ± 3.21 |

**Takeaways**
- Evaluating on the training data inflates the number by about **0.4–1.1 pp** (65.71% vs. 64.6–65.3%).
- The ensemble gain comes from **training the MLP**: under CV it still beats XLM-R solo by about
  +0.9–1.5 pp, while a simple softmax average is *below* XLM-R solo.
- With only 1,114 validation samples (~223 per fold), fold-to-fold variance is large.

## Environment

Python 3.12.0 · numpy 2.4.4 · scipy 1.17.1 · scikit-learn 1.8.0
