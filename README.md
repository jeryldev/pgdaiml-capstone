# Predicting Student Dropout, with a Fairness Audit

A capstone for the PGD AI/ML program, Pillar 5. It predicts whether a student will **drop out** of higher education using enrollment-time data, then **audits the model for bias** across gender, age, and socioeconomic status.

The interesting part of this project is not the model. It is the mistakes I made, found, and had to go back and fix, each of which changed a conclusion. A feature that broke the model's coefficients without touching its accuracy. A SHAP chart that ranked encoded columns and buried the strongest predictor. A fairness metric that asked the wrong question of three attributes out of four. And then, having written a whole section about not reading noise, reading noise. They are written up honestly in the notebooks rather than quietly corrected.

## Problem framing
- **Domain.** Education, predicting student dropout risk.
- **Task.** Binary classification, Dropout vs Graduate, with `Enrolled` (about 18 percent, no settled outcome) dropped.
- **Label.** `dropout = 1` when `Target == "Dropout"`, else `0` for Graduate.
- **Class balance.** About 39 percent dropout, mildly uneven, handled with class weights.
- **Metrics.** **PR AUC and ROC AUC choose the model**, because they hold across every cutoff. **Recall judges the deployed system**, once a cutoff exists, and the cutoff is chosen deliberately, not inherited from a library default. Accuracy is reported and not trusted.
- **Business value.** 2,100 euros per retained student against 50 to 300 euros per outreach contact. These numbers are not decoration. They pick the operating threshold in Step 4.
- **Fairness goal.** Comparable **error rates** across gender, age band, scholarship, and debtor. Demographic parity is reported but is not the verdict. For three of the four attributes it asks the wrong question.

## Headline results
- Logistic regression, chosen over a tied XGBoost on interpretability. The 0.005 gap between them sits inside a 0.020 fold-to-fold swing.
- **Recall 0.796** unmitigated at an operating cutoff of 0.445 (a 45 percent staff-capacity target that lands at 48.3 percent flagged on the test set). The **shipped** system adds the exponentiated-gradient fairness fix, catching **71.8 percent** with the gender recall gap taken from a real 0.177 down to 0.009, which the bootstrap cannot tell apart from zero, at a cost of about 8,000 EUR per 1,000 students.
- SHAP, summed back onto features rather than left scattered across one-hot columns, puts **Course** far ahead of everything, and puts **a parent's occupation** level with the money signals rather than below them.
- The real fairness harm is a **recall gap, 0.877 for men, 0.700 for women**. Three in ten women who drop out are never flagged. A bootstrap confirms it, 95% interval [0.082, 0.274], never crossing zero.
- Both mitigations reach the **same fairness**, 0.009 and 0.026 are one number inside the noise, on 130 female dropouts. They part on **cost**, eight points of recall against nineteen, so the exponentiated gradient ships.
- The shipped fair model makes the decision but **cannot rank**. Its own vote collapses to five coarse bands, PR-AUC 0.617 against the base model's 0.822. So the fair model picks who is on the list and the base model sets the order. That holds because the fair list of **410 per 1,000 fits inside the 450 the team can staff**, where the unmitigated 483 overshot it. The fairness fix is what first pulled the list within reach, and it only fails if the list is not worked to the end.
- **Bonus, generative AI.** A SHAP-to-sentence explanation layer, built against a rules-engine baseline and behind a verifier. A naive prompt failed all five students, printing numbers and naming gender; the guarded prompt passed all five. It runs for a grader with no key off a committed response cache.

## Dataset
The UCI set *Predict Students' Dropout and Academic Success* (Realinho et al., 2021, Instituto Politécnico de Portalegre, Portugal). 4,424 students, 36 features plus the target. Provenance in `data/README.md`, full column reference in `data/data_dictionary.md`.

## Repository map
| Step | Deliverable | Where |
|---|---|---|
| 1 Problem framing | problem, task, metrics, KPI | `notebooks/01_problem_framing.ipynb` |
| 2 Data understanding | overview, data dictionary, disparity preview | `notebooks/02_data_understanding.ipynb`, `data/data_dictionary.md` |
| 3 Preprocessing, EDA, FE | cleaning, EDA, features, selection, PCA | `notebooks/03_eda_feature_engineering.ipynb`, `src/features.py` |
| 4 Modeling | 7 tuned models, comparison, operating threshold | `notebooks/04_modeling.ipynb`, `models/` |
| 5 Ethics and bias | SHAP, fairness audit, 2 mitigations, shipped model | `notebooks/05_ethics_bias_audit.ipynb`, `models/` |
| 6 Presentation | technical and business decks | `presentations/` |
| 7 Report and repo | full write-up, open-source structure, reproducible code | `reports/final_report.md`, `reports/final_report.pdf`, this repo |
| 9 Generative AI (bonus) | explanation layer, naive vs guarded behind a verifier | `notebooks/09_genai.ipynb`, `models/genai_cache.json` |

## Build checklist
- [x] **0** Repo scaffolded, on GitHub, first commit
- [x] **1** Problem framing notebook, metrics, business KPI
- [x] **2** Data understanding, complete data dictionary, disparity preview
- [x] **3** Preprocessing, EDA (right test per feature type), FE, selection applied, PCA
- [x] **4** Seven models tuned, metrics table, saved artifacts and hyperparameters, justified choice, operating threshold chosen
- [x] **5** SHAP, limitations, fairness audit (4 attributes), two mitigations measured and seeded, shipped model saved
- [x] **6** Technical deck + business deck (genuinely different)
- [x] **7** Final report PDF, clean-env rebuild verified, repo public + resolves incognito
- [x] **Bonus** GenAI explanation layer, naive vs guarded prompts behind a mechanical verifier, runs off a committed cache with no key

## Setup

Managed with [uv](https://docs.astral.sh/uv/). Python **3.13**, pinned in `.python-version`. Packages pinned exactly in `requirements.txt`.

```bash
./setup.sh                    # uv venv + uv pip install + fetch dataset
source .venv/bin/activate     # once per shell
```

uv reads `.python-version` and will download 3.13 if the machine does not have it, so there is nothing to arrange. If `./setup.sh` reports permission denied, run it as `bash setup.sh`.

No uv? `setup.sh` falls back to `venv` + `pip` and warns if it had to build on a different interpreter. Or by hand.

```bash
uv venv && uv pip install -r requirements.txt && bash data/download_data.sh
```

At this point you can just read the notebooks. **They are committed with their outputs**. Every table, chart, and SHAP plot is already there, and nothing needs running to see the results.

```bash
jupyter lab
```

## Reproducing every result

```bash
./run_notebooks.sh
```

I wrote this script because the manual version kept going wrong. It runs notebooks 01 to 05 **in order, each on a fresh kernel**, writes the outputs back into each `.ipynb`, and checks that they landed.

Before it runs anything it pins the interpreter to `.venv` and **refuses to fall back to system Python**, then verifies the installed versions against the pins. Both guards exist because the same failure kept recurring in this project. A wrong environment does not crash, it *finishes*, and writes numbers that quietly disagree with the report.

The order is not a nicety either. **04 and 05 read files that 03 and 04 write.** Running 05 on its own does not throw. It silently reads a stale `model.joblib` and produces numbers that look entirely reasonable and are wrong.

It takes a while. Notebook 04 tunes seven models inside cross-validation, and the SVM is slow.

### Two models are saved, on purpose

`models/` holds both, and confusing them would be easy.

- **`model.joblib`** is the plain logistic regression. Every number in Step 4 is read off it, and it is the *unmitigated* model.
- **`model_shipped.joblib`** is the one the report recommends deploying. Same model under an equalized-odds constraint, fit by `ExponentiatedGradient`. It carries the fitted preprocessor with it, because the reduction was fit on the encoded matrix rather than on raw columns, so scoring a new student needs both. One caveat, and it is the design of the system rather than a defect. This artifact produces the fair decision, not a ranking. Its vote collapses to five coarse bands, PR-AUC 0.617 against the base model's 0.822. So the fair model decides who is on the list and `model.joblib` sets the order within it. The fair list of 410 per 1,000 fits inside the 450 the team can staff, so the whole list gets worked and the order is a triage sequence rather than a gate. Notebook 05 measures all of this and the residual-risk section states the one condition it depends on.

```python
import joblib
art = joblib.load("models/model_shipped.joblib")
pred = art["estimator"].predict(art["preprocessor"].transform(X), random_state=42)
```

`metrics.json` carries a `shipped` block naming which is which, with recall before and after, the bootstrap interval on the gender gap before and after the fix, and what the fix costs. The after-interval is the one worth having, since it is what shows the remaining 0.009 cannot be told apart from zero. Saving only the first model would have left `models/` describing a system Step 5 says should not be deployed.

### Three reproducibility traps, if you extend this

Every one of these produced numbers that changed while nothing else changed, and not one of them announced itself.

- **`shap.LinearExplainer`**, handed a bare array as background, silently subsamples it to 100 rows, **unseeded**. Tuition status came out at 0.90 on one run and 0.68 on the next. Pass an explicit `shap.maskers.Independent(Ztr, max_samples=Ztr.shape[0])`.
- **`ExponentiatedGradient.predict`** returns a draw from a **randomized mixture** of classifiers, not a single fitted model. Off one fitted object the recall gap read 0.012 on one call and 0.001 on the next. It needs `random_state`.
- **The environment itself.** `requirements.txt` was once rewritten against versions from a different machine, pinning five packages *behind* the env that had produced every number in the report. Nothing would have errored. `run_notebooks.sh` now checks the pins on every run.

Everything else, the split, the CV folds, mutual information, PCA, all seven estimators, and the Step 5 bootstrap, is seeded at 42. A full re-run on the pinned environment reproduces every number in the report.

To refresh the dataset without rebuilding the environment, run `bash data/download_data.sh`

### The bonus notebook runs off a cache, not the network

`notebooks/09_genai.ipynb` is the Step 9 explanation layer, and it is deliberately **not** in `run_notebooks.sh`, because a script that reaches a network can half-succeed. Every prompt and response is committed to `models/genai_cache.json`, so the notebook runs and renders **with no API key at all**. To refresh the cache after changing a prompt, set a free Gemini key (`export GEMINI_API_KEY=...`, from `https://aistudio.google.com/apikey`) and re-run; otherwise it reads the committed cache. The key is never written to the notebook, and a pre-commit hook refuses any commit that carries one.

## Data and citation
Licensed **CC BY 4.0**, reuse permitted with attribution. Full attribution in `data/README.md`.
- **Dataset.** Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. DOI [10.24432/C5MC89](https://doi.org/10.24432/C5MC89).
- **Introductory paper.** Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: a case study. *Trends and Applications in Information Systems and Technologies* (WorldCIST 2021), AISC vol. 1365, Springer, pp. 166–175. DOI [10.1007/978-3-030-72657-7_16](https://doi.org/10.1007/978-3-030-72657-7_16).
