[English Version](README.md)

# Project PUENTE — 스페인어-영어 코드스위칭 감성분석을 위한 재라벨링 접근법

[![KSC 2025](https://img.shields.io/badge/KSC_2025-장려상-gold)](docs/paper.pdf)
[![HF Dataset](https://img.shields.io/badge/🤗_Dataset-badashin%2Flince--sa--refined-yellow)](https://huggingface.co/datasets/badashin/lince-sa-refined)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **"스페인어-영어 코드스위칭 감성분석을 위한 재라벨링 접근법: 데이터 품질 개선의 영향 분석"**
> KSC 2025 (한국소프트웨어종합학술대회) · 학부생논문경진대회 장려상

*PUENTE*는 스페인어로 "다리(bridge)"를 뜻합니다 — 코드스위칭 분석을 통해 언어 간의 다리를 잇는다는 의미를 담았습니다.

<p align="center">
  <img src="docs/award.png" alt="KSC 2025 장려상" width="400"/>
</p>

## 🔍 개요

LINCE SA는 스페인어-영어 코드스위칭 감성분석의 표준 벤치마크지만, 라벨에 오류가 많습니다.
무작위로 뽑은 100개를 직접 검토해 보니 **17개가 잘못 라벨링**되어 있었고, 그중 65%가 Positive ↔ Neutral 혼동이었습니다.
히스패닉 문화권 맥락을 알고 읽으면 감성이 분명한 트윗이 많았습니다.

학부 2학년 때 진행한 **데이터 중심(data-centric)** 프로젝트입니다. 더 복잡한 모델을 만드는 대신 라벨을 직접 검토·수정해
5,567개짜리 Refined Dataset을 만들었고, 라벨 정제가 그동안 시도한 어떤 멀티태스크 구조보다 효과가 컸다는 것을 보였습니다.

**핵심 기여:**
- LINCE SA의 라벨 오류율 17% 확인 (100개 샘플 검토로 추정)
- 직접 검수한 5,567개 Refined Dataset 구축 (라벨 763개 수정)
- 라벨 정제만으로 정확도 +4.0%p — 원본 라벨로 한 멀티태스크 실험 9개보다 모두 높음
- mBERT + XLM-R late fusion 앙상블로 검증 세트 기준 67.15% — [한계](#limitations) 참고

## 📊 핵심 성과

| 단계 | 모델 | 정확도 | 향상폭 |
|---|---|---|---|
| 베이스라인 | mBERT (원본 라벨) | 56.6% | — |
| 데이터 중심 | mBERT (Refined Dataset) | 60.6% | +4.0%p |
| 최종 | Late Fusion 앙상블 | **67.15%**\* | **+10.55%p** |

\* 메타러너를 같은 검증 세트로 학습하고 평가한 값입니다. 비슷한 실행으로 교차검증한 결과는
[한계](#limitations)에 있습니다.

**수치별 근거 노트북** — 노트북은 결과 출력이 남아 있는 원본 Colab 실행 기록입니다:

| 논문 수치 | 노트북 | 실험 |
|---|---|---|
| 56.6% | [`notebooks/01_baseline_mbert.ipynb`](notebooks/01_baseline_mbert.ipynb) | 베이스라인 |
| 60.6% | [`notebooks/02_refined_dataset_mbert.ipynb`](notebooks/02_refined_dataset_mbert.ipynb) | 실험 11 |
| 67.15% | [`notebooks/03_late_fusion_ensemble.ipynb`](notebooks/03_late_fusion_ensemble.ipynb) | 실험 29 |

## 🗂️ 데이터셋

원본 LINCE SA 데이터셋은 Hugging Face에서 제공합니다:
[`lince-benchmark/lince` / `sa_spaeng`](https://huggingface.co/datasets/lince-benchmark/lince)

원본 텍스트는 재배포하지 않고, 대신 다음 두 가지를 공개합니다:

- **`data/label_mapping.json`** — 수정된 763개 라벨 (`{sample_id: {original, corrected}}` 형식)
- **`data/build_refined_dataset.py`** — Hugging Face 원본에 라벨 매핑을 적용해 Refined Dataset 생성

**Refined Dataset 생성 방법:**
```bash
cd data
python build_refined_dataset.py
# 출력: data/refined_dataset.json (5,567개 샘플)
```

**무엇을 왜 고쳤나:**
LINCE SA는 히스패닉 문화권 맥락에 대한 라벨링 기준 없이 만들어졌습니다. 음식, 가족, 음악처럼 그 문화권에서
분명히 긍정적인 스페인어 표현이 중립이나 부정으로 라벨링된 경우가 많았습니다. 처음부터 끝까지 같은 기준으로 다시 라벨링했습니다.
763개 수정 중 가장 많았던 유형은 Positive → Neutral(194), Neutral → Positive(156), Neutral → Negative(133),
Positive → Negative(114)입니다.

## 🏗️ 아키텍처

최종 모델은 **late fusion**으로 mBERT와 XLM-R을 합칩니다. 두 모델을 따로 학습한 뒤, 각 모델의 출력 로짓(3 + 3)을
이어 붙여 작은 MLP 메타러너에 넣습니다.

```
         Refined Dataset (5,567개 샘플)
        /                              \
   mBERT                           XLM-R
(bert-base-multilingual-cased)  (xlm-roberta-base)
   + 2층 헤드                     + 2층 헤드
   (768 → 256 → 3)               (768 → 256 → 3)
   lr = 5e-6                      lr = 3e-5
        \                              /
         로짓 이어 붙이기 (6차원)
                      |
             MLP 메타러너 (6 → 6 → 3)
                      |
                  최종 예측
```

**관찰한 것:** mBERT는 낮은 학습률(`5e-6`)로 학습해 단독 정확도가 낮습니다(57.36%, XLM-R은 61.04%).
각 모델의 단독 정확도에 유리한 설정(mBERT 학습률 상향, 정확도 기준 조기 종료 — 실험 30, 32)으로 바꾸면 앙상블은
66.16%, 66.25%로 떨어졌습니다. 저는 이를 "두 모델이 각자 강한 것보다 서로 *다르게* 틀리는 것이 앙상블에 더 도움이 된다"고
해석했습니다. 다만 단일 실행에서 본 관찰이지 증명된 효과는 아닙니다 — [한계](#limitations) 참고.

## 🔬 실험 과정

5단계에 걸쳐 **34번 실험**했습니다.
전체 기록은 [`docs/experiment_log.md`](docs/experiment_log.md)에서 확인할 수 있습니다.

| 단계 | 실험 번호 | 방향 | 결과 |
|---|---|---|---|
| 1단계 | 베이스라인 | mBERT 베이스라인 | **56.6%** — 기준점 |
| 2단계 | 실험 1–9, 15–19 | 멀티태스크 학습 (하드 공유 + 어댑터) | 단일 태스크보다 모두 낮음 (부정적 전이) |
| 3단계 | 실험 10, 20–28 | 단일 태스크 + 2층 헤드 + 튜닝 | **61.49%** 최고 — 소폭 향상 |
| 4단계 | 실험 11–14 | 데이터 중심 — Refined Dataset | **60.6%** — 모든 MTL 실험보다 높음 |
| 5단계 | 실험 29–34 | Late Fusion 앙상블 | **67.15%** — 최종 결과 |

**정리하면:** 원본 라벨로는 멀티태스크 실험 9개와 튜닝한 단일 태스크 모델(실험 1–10)이 모두 베이스라인 56.6%보다 낮았습니다.
라벨을 정제한 데이터로 바꾸자 가장 큰 향상(+4.0%p)이 나왔습니다. 그 뒤 새 데이터에서 구조를 튜닝한 효과는 1%p 미만(61.49%)이었고,
앙상블은 그 위에 쌓은 결과입니다.

## ⚡ 재현 방법

**`notebooks/`에 있는 것:** 논문 수치를 만든 원본 Google Colab 노트북을 실행 기록으로 보존했습니다.
코드와 결과는 그대로이며, 다운로드 진행바·Drive 마운트 메시지·라이브러리 경고만 출력에서 지우고
맨 위에 영어 개요 셀을 추가했습니다. 코드 주석은 한국어입니다.

**바로 재현할 수 있는 것**

| 대상 | 방법 | GPU |
|---|---|---|
| Refined Dataset (5,567개) | `cd data && python build_refined_dataset.py` | 불필요 |
| 메타러너 재검증 (65.71%, CV, softmax 평균) | [`experiments/meta_learner_cv/`](experiments/meta_learner_cv/) 참고 | 불필요 |
| 모델 학습 (노트북 01–03) | 아래 참고 | 필요 |

**노트북을 다시 돌리려면** Colab + Google Drive 기준으로 작성되어 있어 몇 가지를 바꿔야 합니다:
- 데이터를 Google Drive 경로(`/content/drive/...`)에서 읽습니다. 로컬 경로로 바꿔 주세요.
- 노트북 01은 `lince_sa_preprocessed.json`(6,319개, 원본 라벨)을 읽는데, 이 파일은 레포에 없습니다.
- 노트북 02, 03은 `lince_sa_golden_cleaned.json`을 읽습니다. `data/build_refined_dataset.py`는
  Hugging Face에서 Refined Dataset을 만들지만, 텍스트를 `cleaned_text`가 아닌 `text` 필드에 저장합니다.
  두 파일이 완전히 같은지는 확인하지 않았습니다.
- 데이터와 시드가 같아도 GPU 연산 특성상 정확도가 ±1%p 정도 달라질 수 있습니다.

**요구 사항:** Python 3.10 이상, 모델 학습에는 CUDA GPU 필요 (원래 Google Colab Pro T4 환경에서 실행)

```bash
git clone https://github.com/vamosbada/project-puente.git
cd project-puente
pip install -r requirements.txt
cd data && python build_refined_dataset.py && cd ..
```

**기록된 학습 시간 (T4 GPU, Colab Pro):** 01 약 16분, 02 약 13분, 03의 mBERT 약 13분
(03의 XLM-R 학습 출력은 남아 있지 않습니다).

<a id="limitations"></a>
## ⚠️ 한계

이 프로젝트는 학부 2학년 때 라벨링, 실험, 논문 작성까지 혼자 진행했습니다.
돌아보면 부족했던 점과, 다음에 한다면 어떻게 할지를 정리했습니다.

- **라벨링은 주로 혼자 했습니다.** 763개를 모두 직접 수정했고, 판단이 애매한 트윗은 멕시코인 스페인어 원어민 친구들에게
  물어봤습니다. 다만 비공식적으로 물어본 것이라 체계적인 이중 라벨링이 아니어서 Cohen's kappa 같은 일치도 지표는 없습니다.
  *다음에는:* 라벨링 기준을 먼저 문서로 정리하고, 두 번째 평가자가 무작위 샘플(100–200개)을 독립적으로 라벨링하게 해서
  Cohen's kappa를 보고하겠습니다.
- **별도의 테스트 세트가 없습니다.** LinCE는 테스트 세트 라벨을 공개하지 않아, 모든 정확도는 라벨 비율을 유지해 나눈(stratified)
  20% 검증 세트에서 쟀고, 하이퍼파라미터도 같은 검증 세트를 보고 골랐습니다.
  *다음에는:* 처음부터 테스트 세트를 따로 떼어 두고, 마지막에 한 번만 쓰겠습니다.
- **67.15%는 메타러너를 학습에 쓴 데이터로 평가한 값입니다.** MLP 메타러너를 검증 세트 로짓으로 학습하고 같은 세트로 평가했습니다.
  그 실행의 로짓은 이후 실행에 덮어써져 직접 다시 확인할 수 없습니다([`docs/experiment_log.md`](docs/experiment_log.md)에는
  실험 29 설정을 다시 돌렸을 때 같은 평가 방식으로 67.15%가 그대로 나왔다는 기록이 있습니다). 대신 비슷한 실행(실험 34: 같은 구조,
  XLM-R lr 2e-5, 6 에포크; 원래 방식으로 65.71%)의 저장된 로짓으로 5-fold stratified CV를 해보니 65.17% ± 3.81%였고,
  fold seed 5개 기준으로 원래 방식이 **약 0.4–1.1%p 높게** 나왔습니다. CV에서도 앙상블은 XLM-R 단독보다 약 +1–1.5%p 높았지만,
  두 모델의 softmax 출력을 단순 평균하면(62.12%) XLM-R 단독(63.73%)보다 낮았습니다. 즉 앙상블 효과는 MLP를 학습시킨 데서 나옵니다.
  자세한 내용: [`experiments/meta_learner_cv/`](experiments/meta_learner_cv/)
  *다음에는:* 메타러너를 처음부터 교차검증으로 학습하고, 모든 실행의 로짓을 저장하겠습니다.
- **56.6%와 60.6%는 같은 검증 샘플에서 잰 값이 아닙니다.** 정제하면서 중복 ID를 지워 데이터셋이 달라졌기 때문에, 베이스라인은
  검증 샘플 1,264개, Refined Dataset 실행은 1,114개를 썼습니다.
  *다음에는:* 정제 전에 샘플 ID 기준으로 split을 고정해서, 같은 샘플로 비교하겠습니다.
- **검증 세트가 작고, 대부분 한 번씩만 돌렸습니다.** 검증 샘플이 1,114개(fold당 약 223개)라 fold 간 표준편차가 최대 약 3.8%p입니다.
  대부분의 결과는 시드 하나로 한 번 돌린 값이고, 일부 비교는 설정을 두 개 이상 동시에 바꿨습니다(실험 30은 학습률과 조기 종료 기준을
  함께 바꿈). 그래서 "약한 mBERT가 앙상블에 도움이 된다"는 관찰(약 1%p 차이)은 노이즈 범위 안입니다.
  *다음에는:* 한 번에 하나만 바꾸고, 여러 시드의 평균 ± 표준편차로 보고하겠습니다.
- **논문의 지표와 앙상블 설정 두 가지가 실제와 다릅니다.** 논문은 지표를 weighted F1이라고 적었지만, 논문의 모든 수치
  (56.6%, 60.6%, 67.15%)는 노트북에 나온 대로 정확도(accuracy)입니다. 훈련 설정도 mBERT lr 1e-5, XLM-R 6 에포크로
  적었는데, 1e-5는 노트북 01–02와 맞지만 67.15% 실행(노트북 03)은 mBERT lr 5e-6, 두 모델 모두 5 에포크였습니다.
  weighted F1로 봐도 추세는 같습니다 (0.56 → 0.61 → 0.65).
  *다음에는:* 결과표와 설정표를 손으로 옮기지 않고 실행 기록에서 바로 만들겠습니다.

## 📄 인용

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

## 📜 라이선스

- **코드** — MIT License ([LICENSE](LICENSE) 참조)
- **데이터** — `data/label_mapping.json`의 라벨 수정본은 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)으로 공개합니다. 원본 텍스트의 저작권은 LINCE 데이터셋 원저자에게 있으며, 사용 시 해당 라이선스를 확인하시기 바랍니다.

## 🙏 감사의 말

- 지도: 단국대학교 컴퓨터공학과 김선오 교수님
- 지원: 단국대학교 SW중심대학사업단

