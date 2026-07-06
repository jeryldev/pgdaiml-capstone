# Predicting Student Dropout — with a Fairness Audit

Capstone (PGD AI/ML, Pillar 5). Predicts whether a student will **drop out** of higher
education using enrollment-time data, then **audits the model for bias** across gender,
age, and socioeconomic status.

## Problem framing
- **Domain:** Education — predict student dropout risk (a listed domain).
- **Task:** binary classification (Dropout vs Graduate); `Enrolled` (~18%) dropped.
- **Label:** `dropout = 1` if `Target == "Dropout"`, else `0` (Graduate).
- **Class balance:** ~39% dropout — mildly imbalanced, not severe.
- **Metrics:** recall on dropout + PR-AUC + ROC-AUC. Accuracy reported, not relied on.
- **Business KPI:** intervention value vs cost, anchored to Portuguese tuition (~€2,100/student).
- **Fairness goal:** comparable error rates across gender, age band, and SES groups
  (scholarship / debtor), via demographic parity, equalized odds, disparate impact.

## Dataset
UCI *Predict Students' Dropout and Academic Success* (Realinho et al., 2021; Instituto
Politécnico de Portalegre, Portugal). 4,424 students × 36 features + target.
See `data/README.md` for provenance, and *Data & citation* below.

## Rubric map (100 pts + 5 bonus)
| Step | Deliverable | Where |
|---|---|---|
| 1 Problem framing (10) | problem + task + metrics | `notebooks/01_*` |
| 2 Data understanding (10) | overview + data dictionary | `notebooks/02_*`, `data/` |
| 3 Preprocessing/EDA/FE (10) | cleaning, EDA, features, PCA | `notebooks/03_*` |
| 4 Modeling (20) | tuned models + comparison | `notebooks/04_*`, `models/` |
| 5 Ethics & bias (20) | SHAP + fairness audit + mitigation | `notebooks/05_*` |
| 6 Presentation (10) | technical + business decks | `presentations/` |
| 7 GitHub & report (15) | this repo, clean + reproducible | — |
| Bonus (+5) | GenAI / deployment | `notebooks/09_*`, `src/` |

## Build checklist
- [ ] **0** Repo scaffolded, on GitHub, first commit
- [ ] **1** Problem framing notebook + metrics + KPI
- [ ] **2** Data understanding + complete data dictionary + disparity preview
- [ ] **3** Preprocessing, EDA (right test per feature type), FE, selection, PCA
- [ ] **4** ≥4 models tuned, metrics table, saved artifacts, justified choice
- [ ] **5** SHAP + limitations + fairness audit (4 attrs) + ≥2 mitigations measured
- [ ] **6** Technical deck + business deck (genuinely different)
- [ ] **7** Final report PDF, clean-env rebuild verified, repo public + resolves incognito
- [ ] **Bonus** GenAI notebook / Phase-2 comparison / deployment

## Setup

Requires **Python ≥ 3.11**. [uv](https://docs.astral.sh/uv/) is recommended (faster) but optional —
`setup.sh` uses uv if present and falls back to `venv` + `pip` otherwise. One command builds the
environment **and** fetches the dataset.

```bash
# optional: curl -LsSf https://astral.sh/uv/install.sh | sh
./setup.sh                    # build .venv + install deps + fetch dataset
source .venv/bin/activate     # activate once per shell
jupyter lab
```

If `./setup.sh` reports `permission denied`, mark it executable first: `chmod +x setup.sh`
(or run it as `bash setup.sh`).

Refresh the dataset later, without rebuilding the environment:

```bash
bash data/download_data.sh
```

Pure pip, by hand: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && bash data/download_data.sh`

## Data & citation
Licensed **CC BY 4.0** (reuse permitted with attribution). Full attribution in `data/README.md`.
- **Dataset** — Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).
  *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository.
  DOI: [10.24432/C5MC89](https://doi.org/10.24432/C5MC89).
- **Introductory paper** — Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021).
  Early prediction of student's performance in higher education: a case study. *Trends and Applications
  in Information Systems and Technologies* (WorldCIST 2021), AISC vol. 1365, Springer, pp. 166–175.
  DOI: [10.1007/978-3-030-72657-7_16](https://doi.org/10.1007/978-3-030-72657-7_16).
