# Data

## Source and citation
- **Dataset.** The UCI set *Predict Students' Dropout and Academic Success*.
- **Authors.** Realinho, V., Vieira Martins, M., Machado, J., and Baptista, L. (2021), Instituto Politécnico de Portalegre (IPP), Portugal.
- **DOI** 10.24432/C5MC89. **License** CC BY 4.0, free reuse with attribution.
- **Introductory paper.** Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., and Realinho, V. (2021). Early prediction of student's performance in higher education: a case study. *Trends and Applications in Information Systems and Technologies* (WorldCIST 2021), AISC vol. 1365, Springer, pp. 166 to 175. DOI 10.1007/978-3-030-72657-7_16. The authors request this citation for scientific use.
- **Funding.** SATDAP, Capacitação da Administração Pública, grant POCI-05-5762-FSE-000191.
- **Vintage.** Published 2021, with the underlying enrollment and outcome data running through about 2019.

## Provenance
The file gathers several disjoint institutional databases into one student-level table. Each row is one student. The columns cover information known at enrollment (academic path, demographics, socioeconomic factors) plus academic performance at the end of the first and second semesters. The providers state they cleaned the file for anomalies, unexplainable outliers, and missing values before release, so the clean state I see is curated by them, not a natural property of raw institutional data.

## How to obtain
I do not commit the raw data to git, since it is re-fetchable and CC BY. After cloning, I run this.

```bash
bash data/download_data.sh
```

It downloads from UCI, with no account needed, and writes `data/raw/dropout.csv`.

## File
| file | rows | cols | sep | label | missing |
|---|---|---|---|---|---|
| `raw/dropout.csv` | 4,424 | 37 | `;` | `Target` | none |

My binary framing is Dropout vs Graduate, with Enrolled dropped, which leaves 3,630 rows at 39.1 percent dropout.

## Sensitive attributes, for the Step 5 fairness audit
Gender, Age at enrollment, and socioeconomic proxies (Scholarship holder, Debtor, Tuition fees up to date, parental education and occupation). All show real dropout-rate disparities, which I quantify in `notebooks/02`.

## Leakage note
The second-semester curricular-unit fields are near-outcome signals. I train the model on enrollment-time features only, and I exclude the curricular-progress columns. I narrate this as leakage mitigation in Steps 3 and 5.
