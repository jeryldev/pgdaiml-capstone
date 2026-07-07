# Predicting Student Dropout, with a Fairness Audit

A capstone for the PGD AI/ML program, Pillar 5. It predicts whether a student will **drop out** of higher education using enrollment-time data, then **audits the model for bias** across gender, age, and socioeconomic status.

## Problem framing
- **Domain.** Education, predicting student dropout risk, a listed domain.
- **Task.** Binary classification, Dropout vs Graduate, with `Enrolled` (about 18 percent) dropped.
- **Label.** `dropout = 1` when `Target == "Dropout"`, else `0` for Graduate.
- **Class balance.** About 39 percent dropout, mildly uneven, not severe.
- **Metrics.** Recall on dropout, PR AUC, and ROC AUC. Accuracy is reported but not relied on.
- **Business value.** Intervention value against cost, anchored to Portuguese tuition, about 2,100 euros per student.
- **Fairness goal.** Comparable error rates across gender, age band, and socioeconomic groups (scholarship, debtor), using demographic parity, equalized odds, and disparate impact.

## Dataset
The UCI set *Predict Students' Dropout and Academic Success* (Realinho et al., 2021, Instituto Politécnico de Portalegre, Portugal). 4,424 students, 36 features plus the target. See `data/README.md` for provenance, and Data and citation below.

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
- [ ] **4** At least 4 models tuned, metrics table, saved artifacts, justified choice
- [ ] **5** SHAP + limitations + fairness audit (4 attrs) + at least 2 mitigations measured
- [ ] **6** Technical deck + business deck (genuinely different)
- [ ] **7** Final report PDF, clean-env rebuild verified, repo public + resolves incognito
- [ ] **Bonus** GenAI notebook / Phase-2 comparison / deployment

## Setup

Requires Python 3.11 or newer. [uv](https://docs.astral.sh/uv/) is recommended for speed but optional. `setup.sh` uses uv if present and falls back to venv and pip otherwise. One command builds the environment and fetches the dataset.

```bash
# optional, install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
./setup.sh                    # build .venv, install deps, fetch dataset
source .venv/bin/activate     # activate once per shell
jupyter lab
```

If `./setup.sh` reports permission denied, mark it executable first with `chmod +x setup.sh`, or run it as `bash setup.sh`.

To refresh the dataset later, without rebuilding the environment.

```bash
bash data/download_data.sh
```

Pure pip, by hand. `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && bash data/download_data.sh`

## Data and citation
Licensed **CC BY 4.0**, reuse permitted with attribution. Full attribution in `data/README.md`.
- **Dataset.** Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. DOI [10.24432/C5MC89](https://doi.org/10.24432/C5MC89).
- **Introductory paper.** Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: a case study. *Trends and Applications in Information Systems and Technologies* (WorldCIST 2021), AISC vol. 1365, Springer, pp. 166 to 175. DOI [10.1007/978-3-030-72657-7_16](https://doi.org/10.1007/978-3-030-72657-7_16).
