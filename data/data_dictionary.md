# Data Dictionary

UCI *Predict Students' Dropout and Academic Success*, id 697. Realinho, Vieira Martins,
Machado and Baptista, 2021. DOI 10.24432/C5MC89, licensed CC BY 4.0.

4,424 rows, 37 columns. Semicolon separated, UTF-8 with a byte order mark, which is why
`src/data.py` strips the BOM and a trailing tab from the header on load. **No missing
cells anywhere in the file.** I checked rather than assumed, because a clean file is a
claim worth verifying before building on it.

The 37 columns fall into six groups, and the group a column lands in decides how the
model is allowed to treat it.

| group | columns | goes into the model |
|---|---|---|
| continuous | 6 | yes, scaled |
| binary flag | 8 | yes, passed through, minus 3 dropped in Step 3 |
| coded category | 9 | one-hot encoded, minus 1 dropped and 5 replaced in Step 3 |
| count | 1 | yes, scaled |
| curricular outcome | 12 | **no. leakage.** |
| target | 1 | this is the thing being predicted |

The twelve curricular columns are the reason this dataset needs reading carefully. They
record how many units a student enrolled in, passed and was graded on, which is
information the school does not have at the moment it wants a prediction. A model given
those columns scores beautifully and is useless, because by the time you can fill them in
the student has already left. Notebook 02, Section 2 works through why.

---

## Continuous

| column | range | units | missing |
|---|---|---|---|
| `Previous qualification (grade)` | 95 to 190 | grade points, Portuguese 0 to 200 scale | 0 |
| `Admission grade` | 95 to 190 | grade points, Portuguese 0 to 200 scale | 0 |
| `Age at enrollment` | 17 to 70 | years | 0 |
| `Unemployment rate` | 7.6 to 16.2 | percent, national, at time of enrolment | 0 |
| `Inflation rate` | -0.8 to 3.7 | percent, national, at time of enrolment | 0 |
| `GDP` | -4.06 to 3.51 | percent change, national, at time of enrolment | 0 |

The last three describe the country, not the student. Every student enrolling in the same
year carries the same three values, so they act as a coarse cohort marker rather than an
individual signal.

## Binary Flags

| column | meaning of 1 | meaning of 0 | missing |
|---|---|---|---|
| `Daytime/evening attendance` | daytime | evening | 0 |
| `Displaced` | studying away from home | local | 0 |
| `Educational special needs` | yes | no | 0 |
| `Debtor` | owes the school money | does not | 0 |
| `Tuition fees up to date` | paid up | behind | 0 |
| `Gender` | male | female | 0 |
| `Scholarship holder` | yes | no | 0 |
| `International` | yes | no | 0 |

Three of these leave in Step 3. `International` is `Nacionality != 1` on every single row,
so it is the same column twice. `Daytime/evening attendance` is decided entirely by
`Course`, since no course runs both a day and an evening class. `Educational special needs`
sits at the bottom of every ranking and applies to a handful of students.

## Coded Categories

Integer codes with no order to them. A code of 44 is not larger than a code of 2, it is
just different, which is why every one of these gets one-hot encoded rather than scaled.

| column | range | distinct codes | missing |
|---|---|---|---|
| `Marital status` | 1 to 6 | 6 | 0 |
| `Application mode` | 1 to 57 | 18 | 0 |
| `Course` | 33 to 9991 | 17 | 0 |
| `Previous qualification` | 1 to 43 | 17 | 0 |
| `Nacionality` | 1 to 109 | 21 | 0 |
| `Mother's qualification` | 1 to 44 | 29 | 0 |
| `Father's qualification` | 1 to 44 | 34 | 0 |
| `Mother's occupation` | 0 to 194 | 32 | 0 |
| `Father's occupation` | 0 to 195 | 46 | 0 |

Five of these get replaced in Step 3 rather than dropped. The four parental columns carry
141 distinct codes between them, which is 141 dummy columns saying nothing about order or
grouping. They fold down into two education ladders and two ISCO occupational groups.
`Application mode` folds into five admission routes. `Nacionality` leaves entirely.

`Nacionality` is spelled that way in the source file. I kept the typo rather than rename
it, so the column names in this repo match the column names in the download.

## Counts

| column | range | units | missing |
|---|---|---|---|
| `Application order` | 0 to 9 | **0 = first choice**, 9 = last choice | 0 |

The one column where the number means something ordered, so it gets scaled like a
continuous feature rather than one-hot encoded. Note the direction. Zero is the *best*
outcome here, not the worst.

## Curricular Outcomes, Excluded as Leakage

| column | range | units | missing |
|---|---|---|---|
| `Curricular units 1st sem (credited)` | 0 to 20 | units | 0 |
| `Curricular units 1st sem (enrolled)` | 0 to 26 | units | 0 |
| `Curricular units 1st sem (evaluations)` | 0 to 45 | assessments | 0 |
| `Curricular units 1st sem (approved)` | 0 to 26 | units | 0 |
| `Curricular units 1st sem (grade)` | 0 to 18.875 | grade points, 0 to 20 scale | 0 |
| `Curricular units 1st sem (without evaluations)` | 0 to 12 | units | 0 |
| `Curricular units 2nd sem (credited)` | 0 to 19 | units | 0 |
| `Curricular units 2nd sem (enrolled)` | 0 to 23 | units | 0 |
| `Curricular units 2nd sem (evaluations)` | 0 to 33 | assessments | 0 |
| `Curricular units 2nd sem (approved)` | 0 to 20 | units | 0 |
| `Curricular units 2nd sem (grade)` | 0 to 18.5714 | grade points, 0 to 20 scale | 0 |
| `Curricular units 2nd sem (without evaluations)` | 0 to 12 | units | 0 |

Dropping twelve of the strongest columns in the file was the hardest call in Step 2, and
it is the one I would defend first. A student who passed zero units in the second semester
has effectively already dropped out. Predicting that they will drop out is not prediction.

## Target

| column | values | missing |
|---|---|---|
| `Target` | `Dropout`, `Enrolled`, `Graduate` | 0 |

`Enrolled` means the student was still studying when the data was collected, so their
outcome is genuinely unknown rather than missing. Those 794 rows are dropped, leaving
3,630 students with a settled outcome and a 39.1 percent dropout rate.
