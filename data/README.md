# Data

## Source and citation
- **Dataset.** The UCI set *Predict Students' Dropout and Academic Success*.
- **Authors.** Realinho, V., Vieira Martins, M., Machado, J., and Baptista, L. (2021), Instituto Politécnico de Portalegre (IPP), Portugal.
- **DOI** 10.24432/C5MC89. **License** CC BY 4.0, free reuse with attribution.
- **Introductory paper.** Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., and Realinho, V. (2021). Early prediction of student's performance in higher education: a case study. *Trends and Applications in Information Systems and Technologies* (WorldCIST 2021), AISC vol. 1365, Springer, pp. 166 to 175. DOI 10.1007/978-3-030-72657-7_16. The authors request this citation for scientific use.
- **Funding.** SATDAP, Capacitação da Administração Pública, grant POCI-05-5762-FSE-000191.
- **Vintage.** Published 2021, with the underlying enrollment and outcome data running through about 2019.

## Provenance
The file gathers several disjoint institutional databases into one student-level table. Each row is one student. The columns cover information known at enrollment (academic path, demographics, socioeconomic factors) plus academic performance at the end of the first and second semesters. The providers state they cleaned the file for anomalies, unexplainable outliers, and missing values before release, so the clean state here is curated by them, not a natural property of raw institutional data.

## How to obtain
The raw data is not committed to git, since it is re-fetchable and CC BY. After cloning, run this.

```bash
bash data/download_data.sh
```

It downloads from UCI, with no account needed, and writes `data/raw/dropout.csv`.

## File
| file | rows | cols | sep | label | missing |
|---|---|---|---|---|---|
| `raw/dropout.csv` | 4,424 | 37 | `;` | `Target` | none |

The binary framing is Dropout vs Graduate, with Enrolled dropped, which leaves 3,630 rows at 39.1 percent dropout.

## Sensitive attributes, for the Step 5 fairness audit
Four attributes are **audited**, and they are the four in `SENSITIVE` in `src/data.py`.

| Attribute | Why |
|---|---|
| Gender | Protected. The one attribute the model turns out to *amplify* rather than report. |
| Age at enrollment | Banded into four groups for the audit. Carries the largest genuine base-rate gap. |
| Scholarship holder | Socioeconomic proxy. |
| Debtor | Socioeconomic proxy. |

All four show real dropout-rate disparities before any model exists, quantified in `notebooks/02`. That matters: a gap the model *reports* is not the same as a gap the model *creates*, and Step 5 sets the two side by side rather than treating demographic parity as a verdict.

A wider set of proxies — tuition status, parental education, parental occupation — is **discussed** in Step 5 but is not part of the audited four. Tuition status is handled separately as a watched near-outcome signal (Step 3). Parental occupation matters for a different reason: once SHAP values are grouped back onto features, mother's occupation ranks third among the model's drivers, which makes it a proxy worth naming even though the audit does not run on it.

## Leakage note
The second-semester curricular-unit fields are near-outcome signals. The model trains on enrollment-time features only, with the curricular-progress columns excluded. This is narrated as leakage mitigation in Steps 3 and 5.
