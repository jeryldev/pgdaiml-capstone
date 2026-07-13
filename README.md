# Predicting Student Dropout, with a Fairness Audit

A capstone for the PGD AI/ML program, Pillar 5. It predicts whether a student will **drop out** of higher education using enrollment-time data, then **audits the model for bias** across gender, age, and socioeconomic status.

The interesting part of this project is not the model. It is the three things I got wrong on the first pass and had to go back and fix, each of which changed a conclusion. They are written up honestly in the notebooks rather than quietly corrected.

## Problem framing
- **Domain.** Education, predicting student dropout risk.
- **Task.** Binary classification, Dropout vs Graduate, with `Enrolled` (about 18 percent, no settled outcome) dropped.
- **Label.** `dropout = 1` when `Target == "Dropout"`, else `0` for Graduate.
- **Class balance.** About 39 percent dropout, mildly uneven, handled with class weights.
- **Metrics.** **PR AUC and ROC AUC choose the model**, because they hold across every cutoff. **Recall judges the deployed system**, once a cutoff exists — and the cutoff is chosen deliberately, not inherited from a library default. Accuracy is reported and not trusted.
- **Business value.** 2,100 euros per retained student against 50 to 300 euros per outreach contact. These numbers are not decoration: they pick the operating threshold in Step 4.
- **Fairness goal.** Comparable **error rates** across gender, age band, scholarship, and debtor. Demographic parity is reported but is not the verdict — for three of the four attributes it asks the wrong question.

## Headline results
- Logistic regression, chosen over a tied XGBoost on interpretability. The 0.005 gap between them sits inside a 0.020 fold-to-fold swing.
- **Recall 0.796** at an operating threshold of 0.445, chosen from out-of-fold training scores at 45 percent staff capacity.
- SHAP, summed back onto features rather than left scattered across one-hot columns, puts **Course first** and **mother's occupation third**.
- The real fairness harm is a **recall gap: 0.877 for men, 0.700 for women**. Three in ten women who drop out are never flagged.
- An exponentiated-gradient reduction closes that gap to 0.009 for eight points of recall. The threshold optimizer closes it by helping fewer people, which is not a fix.

## Dataset
The UCI set *Predict Students' Dropout and Academic Success* (Realinho et al., 2021, Instituto Politécnico de Portalegre, Portugal). 4,424 students, 36 features plus the target. Provenance in `data/README.md`, full column reference in `data/data_dictionary.md`.

## Repository map
| Step | Deliverable | Where |
|---|---|---|
| 1 Problem framing | problem, task, metrics, KPI | `notebooks/01_problem_framing.ipynb` |
| 2 Data understanding | overview, data dictionary, disparity preview | `notebooks/02_data_understanding.ipynb`, `data/data_dictionary.md` |
| 3 Preprocessing, EDA, FE | cleaning, EDA, features, selection, PCA | `notebooks/03_eda_feature_engineering.ipynb`, `src/features.py` |
| 4 Modeling | 7 tuned models, comparison, operating threshold | `notebooks/04_modeling.ipynb`, `models/` |
| 5 Ethics and bias | SHAP, fairness audit, 2 mitigations | `notebooks/05_ethics_bias_audit.ipynb` |
| 6 Presentation | technical and business decks | `presentations/` |
| 7 Report | full write-up | `reports/final_report.md` |

## Build checklist
- [x] **0** Repo scaffolded, on GitHub, first commit
- [x] **1** Problem framing notebook, metrics, business KPI
- [x] **2** Data understanding, complete data dictionary, disparity preview
- [x] **3** Preprocessing, EDA (right test per feature type), FE, selection applied, PCA
- [x] **4** Seven models tuned, metrics table, saved artifacts and hyperparameters, justified choice, operating threshold chosen
- [x] **5** SHAP, limitations, fairness audit (4 attributes), two mitigations measured and seeded
- [ ] **6** Technical deck + business deck (genuinely different)
- [ ] **7** Final report PDF, clean-env rebuild verified, repo public + resolves incognito
- [ ] **Bonus** GenAI explanation layer or deployment

## Setup

Requires Python 3.11 or newer. [uv](https://docs.astral.sh/uv/) is recommended for speed but optional — `setup.sh` uses it if present and falls back to venv and pip otherwise.

```bash
./setup.sh                    # build .venv, install pinned deps, fetch dataset
source .venv/bin/activate     # once per shell
```

If `./setup.sh` reports permission denied, run it as `bash setup.sh`.

At this point you can open the notebooks and read them. **They are committed with their outputs**, so every table, chart, and SHAP plot is already there — nothing needs running to see the results.

```bash
jupyter lab
```

## Reproducing every result

```bash
bash run_notebooks.sh
```

I wrote this script because the manual version kept going wrong. It runs notebooks 01 to 05 **in order, each on a fresh kernel**, writes the outputs back into each `.ipynb`, and then checks that they actually landed.

The order is not a nicety. **04 and 05 read files that 03 and 04 write.** Running 05 on its own does not throw an error — it silently reads a stale `model.joblib` from an earlier run and produces numbers that look entirely reasonable and are wrong. Nothing tells you. That is the failure the script exists to prevent.

It takes a while. Notebook 04 tunes seven models inside cross-validation, and the SVM is slow.

### Two reproducibility traps, if you extend this

Both of these produced numbers that changed between runs while nothing else changed, and neither announced itself.

- **`shap.LinearExplainer`**, handed a bare array as background, silently subsamples it to 100 rows, **unseeded**. Tuition status came out at 0.90 on one run and 0.68 on the next. Pass an explicit `shap.maskers.Independent(Ztr, max_samples=Ztr.shape[0])`.
- **`ExponentiatedGradient.predict`** returns a draw from a **randomized mixture** of classifiers, not a single fitted model. Off one fitted object the recall gap read 0.012 on one call and 0.001 on the next. It needs `random_state`.

Everything else — the split, the CV folds, mutual information, PCA, and all seven estimators — is seeded at 42. A full re-run reproduces every number in the report.

To refresh the dataset without rebuilding the environment: `bash data/download_data.sh`

## Data and citation
Licensed **CC BY 4.0**, reuse permitted with attribution. Full attribution in `data/README.md`.
- **Dataset.** Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. DOI [10.24432/C5MC89](https://doi.org/10.24432/C5MC89).
- **Introductory paper.** Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: a case study. *Trends and Applications in Information Systems and Technologies* (WorldCIST 2021), AISC vol. 1365, Springer, pp. 166–175. DOI [10.1007/978-3-030-72657-7_16](https://doi.org/10.1007/978-3-030-72657-7_16).
