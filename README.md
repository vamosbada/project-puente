[한국어 버전](README_ko.md)

# Project PUENTE — Relabeling for Spanish-English Code-Switching Sentiment Analysis

[![KSC 2025](https://img.shields.io/badge/KSC_2025-Honorable_Mention-gold)](docs/paper.pdf)
[![HF Dataset](https://img.shields.io/badge/🤗_Dataset-badashin%2Flince--sa--refined-yellow)](https://huggingface.co/datasets/badashin/lince-sa-refined)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **"A Relabeling Approach for Spanish-English Code-Switching Sentiment Analysis:
> Impact Analysis of Data Quality Improvement"**
> KSC 2025 (Korean Software Congress) · Honorable Mention, Undergraduate Division

*PUENTE* means "bridge" in Spanish — reflecting the goal of bridging languages through code-switching analysis.

<p align="center">
  <img src="docs/award.png" alt="KSC 2025 Honorable Mention" width="400"/>
</p>

## 🔍 Overview

LINCE SA is the standard benchmark for Spanish-English code-switching sentiment analysis, but its labels are noisy.
In a manual audit of 100 random samples, **17 were mislabeled**, and 65% of those errors were Positive ↔ Neutral
confusions — often tweets whose sentiment is clear once you read them with Hispanic cultural context.

This is a **data-centric** project, done as a second-year undergraduate. Instead of building a more complex model,
I reviewed and corrected the labels myself, built a 5,567-sample Refined Dataset, and showed that cleaning the labels
helped more than any of the multi-task architectures I had tried.

**Core contributions:**
- Found a 17% label error rate in LINCE SA (estimated from a 100-sample audit)
- Built a 5,567-sample Refined Dataset by hand (763 labels corrected)
- Relabeling alone raised accuracy by 4.0 pp — more than any of the 9 multi-task experiments on the original labels
- Reached 67.15% accuracy with a late-fusion ensemble (mBERT + XLM-R) on the validation set — see [Limitations](#limitations)

## 📊 Key Results

| Stage | Model | Accuracy | Improvement |
|---|---|---|---|
| Baseline | mBERT (original labels) | 56.6% | — |
| Data-Centric | mBERT (Refined Dataset) | 60.6% | +4.0 pp |
| Final | Late Fusion Ensemble | **67.15%**\* | **+10.55 pp** |

\* The meta-learner was trained and evaluated on the same validation set. A cross-validated check
on a comparable run is in [Limitations](#limitations).

**Where each number comes from** — the notebooks are the original Colab run records, with outputs:

| Paper number | Notebook | Experiment |
|---|---|---|
| 56.6% | [`notebooks/01_baseline_mbert.ipynb`](notebooks/01_baseline_mbert.ipynb) | Baseline |
| 60.6% | [`notebooks/02_refined_dataset_mbert.ipynb`](notebooks/02_refined_dataset_mbert.ipynb) | Exp. 11 |
| 67.15% | [`notebooks/03_late_fusion_ensemble.ipynb`](notebooks/03_late_fusion_ensemble.ipynb) | Exp. 29 |

## 🗂️ Dataset

The original LINCE SA dataset is available on Hugging Face:
[`lince-benchmark/lince` / `sa_spaeng`](https://huggingface.co/datasets/lince-benchmark/lince)

I do not redistribute the original text. Instead, this repo includes:

- **`data/label_mapping.json`** — 763 corrected labels in `{sample_id: {original, corrected}}` format
- **`data/build_refined_dataset.py`** — reproduces the Refined Dataset from the Hugging Face source

**To build the Refined Dataset:**
```bash
cd data
python build_refined_dataset.py
# outputs: data/refined_dataset.json (5,567 samples)
```

**What was corrected and why:**
LINCE SA was annotated without explicit guidelines for Hispanic cultural context.
Many tweets use Spanish words or expressions that are clearly positive in that context
(e.g., about food, family, or music) but were labeled neutral or negative.
I relabeled them using the same criteria throughout. Of the 763 corrections, the most common were
Positive → Neutral (194), Neutral → Positive (156), Neutral → Negative (133), and Positive → Negative (114).

## 🏗️ Architecture

The final model combines mBERT and XLM-R with **late fusion**: each model is trained
separately, then their raw output logits (3 + 3) are concatenated and fed to a small
MLP meta-learner.

```
         Refined Dataset (5,567 samples)
        /                              \
   mBERT                           XLM-R
(bert-base-multilingual-cased)  (xlm-roberta-base)
   + 2-layer head                 + 2-layer head
   (768 → 256 → 3)               (768 → 256 → 3)
   lr = 5e-6                      lr = 3e-5
        \                              /
         concatenate logits (dim = 6)
                      |
              MLP meta-learner
               (6 → 6 → 3)
                      |
              final prediction
```

**What I observed:** mBERT was trained with a low learning rate (`5e-6`), which lowers its
solo accuracy (57.36% vs 61.04% for XLM-R). When I switched to settings that favored each model's own
accuracy (a higher mBERT learning rate and/or accuracy-based early stopping, Exp. 30 and 32), the ensemble dropped to 66.16% and 66.25%. My reading was that the ensemble gains more
from the two models making *different* errors than from each model being strong. This is an
observation from single runs, not a proven effect — see [Limitations](#limitations).

## 🔬 Experiment Journey

This result came from **34 experiments** across 5 phases.
Full details are in [`docs/experiment_log.md`](docs/experiment_log.md).

| Phase | Experiments | Direction | Outcome |
|---|---|---|---|
| 1 | Baseline | mBERT baseline | **56.6%** — reference point |
| 2 | Exp. 1–9, 15–19 | Multi-task learning (hard sharing + adapters) | None beat single-task (negative transfer) |
| 3 | Exp. 10, 20–28 | Single-task with a 2-layer head + tuning | **61.49%** peak — marginal gain |
| 4 | Exp. 11–14 | Data-centric: Refined Dataset | **60.6%** — beats all MTL approaches |
| 5 | Exp. 29–34 | Late Fusion ensemble | **67.15%** — final result |

**The central finding:** On the original labels, 9 multi-task experiments and a tuned single-task model
(Exp. 1–10) all stayed below the 56.6% baseline. Switching to the relabeled data gave the largest
single jump (+4.0 pp). Architecture tuning on the new data afterwards added less than 1 pp (61.49%),
and the ensemble built on top of that.

## ⚡ Reproduction

**What is in `notebooks/`:** the original Google Colab notebooks that produced the paper's numbers,
kept as run records. Code and outputs are unchanged; only download progress bars, Drive mount
messages and library warnings were removed, and an English overview cell was added at the top.
Code comments are in Korean.

**What you can reproduce directly**

| What | How | GPU |
|---|---|---|
| Refined Dataset (5,567 samples) | `cd data && python build_refined_dataset.py` | No |
| Meta-learner re-evaluation (65.71%, CV, softmax average) | See [`experiments/meta_learner_cv/`](experiments/meta_learner_cv/) | No |
| Model training (notebooks 01–03) | See below | Yes |

**Re-running the notebooks** requires a few changes, because they were written for Colab with Google Drive:
- They read data from Google Drive paths (`/content/drive/...`). Change these to local paths.
- Notebook 01 reads `lince_sa_preprocessed.json` (6,319 samples, original labels), which is not in this repo.
- Notebooks 02 and 03 read `lince_sa_golden_cleaned.json`. `data/build_refined_dataset.py` builds the
  Refined Dataset from Hugging Face, but stores the text in a `text` field instead of `cleaned_text`.
  We have not verified that the two files are byte-for-byte identical.
- Even with identical data and seeds, GPU non-determinism can shift accuracy by around ±1 pp.

**Requirements:** Python 3.10+, CUDA GPU for model training (originally run on Google Colab Pro, T4)

```bash
git clone https://github.com/vamosbada/project-puente.git
cd project-puente
pip install -r requirements.txt
cd data && python build_refined_dataset.py && cd ..
```

**Recorded training times (T4 GPU, Colab Pro):** ~16 min (01), ~13 min (02), ~13 min for mBERT in 03
(the XLM-R training output in 03 was not preserved).

<a id="limitations"></a>
## ⚠️ Limitations

I did this project alone — labeling, experiments, and writing — as a second-year undergraduate.
Looking back, these are the parts that were not good enough, and what I would do differently next time.

- **One main annotator.** I made all 763 corrections myself. For tweets I was unsure about,
  I asked native Spanish-speaking friends from Mexico, but this was informal — not a systematic
  second annotation — so there is no agreement score like Cohen's kappa.
  *Next time:* write the labeling rules down first, have a second annotator label a random
  sample (100–200 tweets) independently, and report Cohen's kappa.
- **No held-out test set.** LinCE does not release test-set labels, so every accuracy here is on my own
  stratified 20% validation split, and I chose hyperparameters by looking at that same split.
  *Next time:* set aside a test split at the start and use it only once, at the end.
- **67.15% evaluates the meta-learner on its own training data.** I trained the MLP meta-learner on the
  validation-set logits and evaluated it on the same set. The logits of that run were later overwritten, so it
  cannot be re-checked directly ([`docs/experiment_log.md`](docs/experiment_log.md) records that re-running the
  Exp. 29 settings gave 67.15% again, with the same evaluation method). As a check, I ran 5-fold stratified CV
  on the saved logits of a comparable run (Exp. 34: same architecture, XLM-R lr 2e-5, 6 epochs; 65.71% with the
  original method). CV gives 65.17% ± 3.81%, and across 5 fold seeds the original method is about
  **0.4–1.1 pp too high**. Under CV the ensemble still beats XLM-R alone by about +1–1.5 pp, while a simple average
  of the two models' softmax outputs (62.12%) is *below* XLM-R alone (63.73%) — so the gain comes from training
  the MLP. Details: [`experiments/meta_learner_cv/`](experiments/meta_learner_cv/).
  *Next time:* train the meta-learner with cross-validation from the start, and save the logits of every run.
- **56.6% and 60.6% are not on the same validation samples.** Removing duplicate IDs during cleaning changed the
  dataset, so the baseline uses 1,264 validation samples and the Refined Dataset run uses 1,114.
  *Next time:* fix the split by sample ID before cleaning, so both are compared on the same samples.
- **Small validation set, single runs.** With 1,114 validation samples (~223 per fold), fold-to-fold standard
  deviation reaches ~3.8 pp. Most results here are one run with one seed, and some comparisons changed more than
  one setting at once (Exp. 30 changed the learning rate and the early-stopping metric together). So the
  "weaker mBERT helps the ensemble" observation (a ~1 pp difference) is within noise.
  *Next time:* change one thing at a time and report mean ± std over several seeds.
- **The paper misstates the metric and two ensemble settings.** The paper says it uses weighted F1, but every
  number in it (56.6%, 60.6%, 67.15%) is accuracy, as the notebooks show. Its training settings list mBERT lr 1e-5
  and XLM-R 6 epochs; 1e-5 matches notebooks 01–02, but the 67.15% run (notebook 03) used mBERT lr 5e-6 and
  5 epochs for both models. Weighted F1 shows the same trend (0.56 → 0.61 → 0.65).
  *Next time:* build the results and settings tables directly from the run logs, not by hand.

## 📄 Citation

```bibtex
@inproceedings{shin2025puente,
  title     = {스페인어-영어 코드스위칭 감성분석을 위한 재라벨링 접근법:
               데이터 품질 개선의 영향 분석},
  author    = {신바다 and 김선오},
  booktitle = {한국소프트웨어종합학술대회 (KSC 2025)},
  year      = {2025},
  note      = {학부생논문경진대회 장려상}
}
```

## 📜 License

- **Code** — MIT License (see [LICENSE](LICENSE))
- **Data** — The label corrections in `data/label_mapping.json` are released under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The underlying text
  belongs to the original LINCE dataset; please refer to their license for usage terms.

## 🙏 Acknowledgments

- Advised by Prof. Sunoh Kim, Dept. of Computer Engineering, Dankook University
- Supported by the SW-oriented University Program (SW중심대학사업)
- Original LINCE dataset: [LinCE Benchmark](https://ritual.uh.edu/lince/)

