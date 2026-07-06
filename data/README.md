# Data

## Source & citation
- **Dataset:** UCI *Predict Students' Dropout and Academic Success*.
- **Authors:** Realinho, V.; Vieira Martins, M.; Machado, J.; Baptista, L. (2021),
  Instituto Politécnico de Portalegre (IPP), Portugal.
- **DOI:** 10.24432/C5MC89 · **License:** CC BY 4.0 (free reuse *with attribution*).
- **Introductory paper:** Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: a case study. *Trends and Applications in Information Systems and Technologies* (WorldCIST 2021), AISC vol. 1365, Springer, pp. 166–175. DOI: 10.1007/978-3-030-72657-7_16. Requested by the authors for scientific use.
- **Funding:** SATDAP – Capacitação da Administração Pública, grant POCI-05-5762-FSE-000191.
- **Vintage:** published 2021; underlying enrollment/outcome data runs through ~2019.

## Provenance
Aggregated from **several disjoint institutional databases** into one student-level table.
Each row is one student. Columns cover information known **at enrollment** (academic path,
demographics, socioeconomic factors) plus academic performance at the end of the 1st and
2nd semesters. The providers state they performed rigorous preprocessing to remove
anomalies, unexplainable outliers, and missing values — so the clean state of this file is
*curated by them*, not a natural property of raw institutional data.

## How to obtain
Raw data is **not committed** to git (re-fetchable and CC BY). After cloning:

```bash
bash data/download_data.sh
```

Downloads from UCI (no account needed) and writes `data/raw/dropout.csv`.

## File
| file | rows | cols | sep | label | missing |
|---|---|---|---|---|---|
| `raw/dropout.csv` | 4,424 | 37 | `;` | `Target` | none |

Binary framing used in this project: Dropout vs Graduate (Enrolled dropped) →
3,630 rows, 39.1% dropout.

## Sensitive attributes (drive the Step 5 fairness audit)
Gender, Age at enrollment, and socioeconomic proxies (Scholarship holder, Debtor,
Tuition fees up to date, parental education/occupation). All show real dropout-rate
disparities — quantified in `notebooks/02`.

## Leakage note
Second-semester curricular-unit fields are near-outcome signals. The model is trained on
**enrollment-time features only**; curricular-progress columns are excluded. Narrated as
leakage mitigation in Steps 3 and 5.
