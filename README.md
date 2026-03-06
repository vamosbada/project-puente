[한국어 버전](README_ko.md)

# Project PUENTE — Relabeling for Spanish-English Code-Switching Sentiment Analysis

[![KSC 2025](https://img.shields.io/badge/KSC_2025-Honorable_Mention-gold)](docs/paper.pdf)
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

The LINCE SA benchmark is the standard dataset for Spanish-English code-switching sentiment analysis, but its labels are noisy. We found that **17% of samples contain annotation errors** — cases where the original label does not match the sentiment expressed in the tweet when read with Hispanic-American cultural context in mind.

This project takes a **data-centric approach**: instead of building a more complex model, we manually reviewed and corrected the labels, constructed a 5,567-sample Refined Dataset, and showed that data quality improvements alone outperform every architectural change we tried.

**Core contributions:**
- Identified and documented a 17% labeling error rate in the LINCE SA benchmark
- Built a 5,567-sample Refined Dataset with human-verified labels (763 corrections)
- Showed +4.0 pp accuracy gain from data cleaning alone, exceeding 9 multi-task learning experiments
- Achieved 67.15% accuracy with a Late Fusion ensemble (mBERT + XLM-R)

## 📊 Key Results

| Stage | Model | Accuracy | Improvement |
|---|---|---|---|
| Baseline | mBERT (original labels) | 56.6% | — |
| Data-Centric | mBERT (Refined Dataset) | 60.6% | +4.0 pp |
| Final | Late Fusion Ensemble | **67.15%** | **+10.55 pp** |

## 🗂️ Dataset

The original LINCE SA dataset is available on Hugging Face:
[`lince-benchmark/lince` / `sa_spaeng`](https://huggingface.co/datasets/lince-benchmark/lince)

We do not redistribute the original text. Instead, we release:

- **`data/label_mapping.json`** — 763 corrected labels in `{sample_id: {original, corrected}}` format
- **`data/build_refined_dataset.py`** — reproduces the Refined Dataset from the Hugging Face source

**To build the Refined Dataset:**
```bash
cd data
python build_refined_dataset.py
# outputs: data/refined_dataset.json (5,567 samples)
```

**What was corrected and why:**
The LINCE SA dataset was annotated without explicit cultural guidelines for
Hispanic-American English. Many tweets use Spanish words or expressions that carry
positive connotations in that cultural context (e.g., references to food, family,
music) but were labeled as neutral or negative. We re-annotated these with a
consistent cultural framework.

## 🏗️ Architecture

The final model combines mBERT and XLM-R via **Late Fusion**: each model is trained
independently, then their output probability distributions are concatenated and passed
to a small MLP meta-learner.

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

**Key design insight:** mBERT is trained with a deliberately low learning rate (`5e-6`),
which lowers its solo accuracy but increases its prediction diversity relative to XLM-R.
Experiments showed that individually optimized models (higher solo accuracy) actually
*reduced* ensemble performance — the ensemble benefits more from complementary errors
than from raw individual strength.

## 🔬 Experiment Journey

This result came from **34 systematic experiments** across 5 phases.
Full details are in [`docs/experiment_log.md`](docs/experiment_log.md).

| Phase | Experiments | Direction | Outcome |
|---|---|---|---|
| 1 | Exp. 1 | mBERT baseline | **56.6%** — reference point |
| 2 | Exp. 2–9, 15–19 | Multi-task learning (hard sharing + adapters) | All failed; negative transfer confirmed |
| 3 | Exp. 10, 20–28 | Enhanced single-task (2-layer head, tuning) | **61.49%** peak — marginal gain |
| 4 | Exp. 11–14 | Data-centric: Refined Dataset | **60.6%** — beats all MTL approaches |
| 5 | Exp. 29–34 | Late Fusion ensemble | **67.15%** — final result |

**The central finding:** After exhausting model-centric approaches (9 multi-task learning
experiments, 10 architecture/hyperparameter experiments), switching to data quality
improvement produced the largest single jump (+4.0 pp). The ensemble then built on
that stronger foundation.

## ⚡ Reproduction

**Requirements:** Python 3.10+, CUDA GPU recommended (tested on Google Colab Pro)

```bash
# 1. Clone the repository
git clone https://github.com/vamosbada/project-puente.git
cd project-puente

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the Refined Dataset
cd data && python build_refined_dataset.py && cd ..

# 4. Run notebooks in order
jupyter notebook notebooks/
```

**Notebook order:**
1. `01_baseline_mbert.ipynb` — establishes the 56.6% baseline (uses Hugging Face directly)
2. `02_refined_dataset_mbert.ipynb` — requires `data/refined_dataset.json`
3. `03_late_fusion_ensemble.ipynb` — requires `data/refined_dataset.json`

**Expected runtimes (on a single T4 GPU):**
- Notebooks 01 & 02: ~30–40 minutes each
- Notebook 03: ~90–120 minutes (trains two models sequentially)

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

