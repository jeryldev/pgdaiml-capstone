# Predicting Student Dropout, with a Fairness Audit

**Capstone report, PGD AI/ML, Pillar 5.**
An end-to-end machine learning project that predicts which students are at risk of dropping out of higher education, using only what is known at enrollment, and audits the model for bias before trusting it.

Repository: **https://github.com/jeryldev/pgdaiml-capstone**

The full code lives there, one notebook per step, with outputs saved so every number in this report can be checked without running anything. This report tells the whole story, from framing to fairness, including the parts I got wrong on the first pass.

---

## Executive Summary

Every year some students leave higher education before finishing their degree. The cost falls on three sides, the student, the institution, and society. A tool that spots who is at risk early, around the time a student enrolls, lets a support team reach those students before they leave.

This project builds that tool on real data from a Portuguese polytechnic, and then does the harder work of checking whether it is fair.

The final model is a logistic regression. At a cutoff chosen deliberately against the school's own staffing capacity, it catches **79.6 percent** of the students who go on to drop out, using only enrollment-time information. It is simple enough to explain itself, which turns out to matter more than any accuracy figure in this project.

Four findings are worth leading with, and every one of them came from correcting a mistake.

**The model is not mostly a money model.** Once SHAP values are summed back onto the features they came from, rather than left scattered across one-hot columns, the strongest driver is **which course a student enrolled in**, far ahead of everything else. Tuition status sits second, and **what their mother does for a living** sits level with scholarship status right behind it. So the model substantially recognises students who started from further back, not students in trouble.

**The fairness verdict is not "everything fails."** A first audit ran demographic parity across all four sensitive attributes, found every one below the 0.8 disparate impact line, and called it a failure. That was the wrong question for three of them. Debtors drop out at 76 percent against 34 percent, so a model that flags them equally would be ignoring a real 42-point gap on purpose. Set the model's flagging gap next to the gap in the world and the picture separates cleanly. For debtor and scholarship the model **reports** a disparity. For **gender it amplifies one**.

**The real harm is a recall gap.** The model catches **87.7 percent of male dropouts and 70.0 percent of female ones**. Three in ten women who go on to leave are never flagged at all. That does not appear in any flagging-rate metric. It only appears once you stop counting flags and start counting errors.

**And rigor does not generalize on its own.** I spent Step 4 refusing to call a 0.005 lead real against a 0.020 fold swing, then walked into Step 5 and quoted fairness numbers to three decimals off 130 female dropouts. A bootstrap shows the two mitigations are **tied on fairness**, statistically indistinguishable, and that the only difference surviving resampling is what they cost, eight points of recall against nineteen. The harm survived the check. My reason for picking a fix did not.

The honest conclusion is that the model works and should not be trusted blindly. It belongs in the hands of a support team, as a ranked prompt to start a conversation, not as a verdict attached to a student's record.

---

## Step 1. Problem Framing

### The problem, and who it helps

The data comes from the Polytechnic Institute of Portalegre (IPP), a public higher education institution in Portugal. The institution and its research team built this dataset to reduce academic dropout and failure, by identifying students at risk at an early stage so support can be put in place. This project mirrors that goal.

The model serves the tutoring and support staff. They cannot watch every student closely, so they need to focus limited attention on those most at risk. The model turns a long list of students into a short ranked one, the students worth contacting first. A correct flag reaches a student who would otherwise slip away. A wrong one wastes staff time, or misses a student who needed help.

### The task, and why this kind

The original data has three outcomes, Dropout, Graduate, and Enrolled. Enrolled means the student was still studying when the record was taken, so the final outcome is not settled. A student with no settled outcome cannot teach the model what dropout looks like, so those 794 rows are removed. What remains is two settled outcomes, which fits a yes or no question. The task is **binary classification**, Dropout against Graduate, on 3,630 students at a 39.1 percent dropout rate.

Regression predicts a number, but the answer here is a category. Clustering groups unlabeled data, but the labels are known. Classification is the right choice.

### How success is measured

The model outputs a probability, not a hard label. Turning that into a decision needs a cutoff, and **the cutoff is a choice, not a given**. That distinction drives everything downstream, so it is worth being precise about which metrics survive it.

**Two metrics choose the model**, because neither depends on where the cutoff sits.

- **PR AUC**, from precision and recall. It ignores the graduates correctly left alone, so it stays honest on an uneven 39 to 61 split. This is what the tuning optimises.
- **ROC AUC**, the chance the model scores a random dropout above a random graduate. It reads as a ranking quality, which matches the real use.

**One metric judges the deployed system**, once a cutoff exists.

- **Recall on dropout.** Of the students who truly drop out, how many did the model flag? A missed student is the costly error. This is the number the school cares about, and the number I would report to them.

But recall cannot pick a model, and the reason matters. **Recall is not a property of a model, it is a property of a model and a cutoff.** Recall can be driven to 1.0 on any model by flagging every student. So PR AUC and ROC AUC pick the model, Step 4 picks the cutoff on purpose, and only then does recall mean something.

Accuracy is reported and not trusted. With a 39 percent dropout rate, a lazy model that calls everyone a graduate is right 61 percent of the time while catching no one.

### Business value

Value is students kept, minus the cost of reaching out to flagged students. Three stated assumptions keep the estimate honest.

- A kept student is worth about **2,100 euros**, from the Portuguese public tuition cap of roughly 697 euros a year over about three years. A deliberate low estimate. Tuition is citable. The wider loss to the student and society is larger but harder to defend.
- A flag starts a conversation. It does not keep the student on its own, so 2,100 euros is an upper bound per caught student, scaled by how often outreach works, taken here as 3 in 10.
- Outreach costs roughly **50, 150, or 300 euros** per flagged student, reported as a range since the real figure depends on the institution.

These are not decoration. Step 4 uses them directly to choose the probability cutoff, which is the one place in this project where cost and benefit actually meet.

### Scope

The model fits an IPP style Portuguese polytechnic, across 17 undergraduate degrees, from 2008-09 through 2018-19. It is not a general dropout predictor for any school in any country. The method carries even though the model does not.

---

## Step 2. Data Understanding

### Source and citation

The dataset is the UCI set *Predict Students' Dropout and Academic Success* (Realinho, Vieira Martins, Machado, and Baptista, 2021), donated December 2021 under CC BY 4.0, DOI 10.24432/C5MC89. Funded by the Portuguese program SATDAP, grant POCI-05-5762-FSE-000191. The creators joined several separate institutional databases into one student-level table and state they performed rigorous preprocessing for anomalies, outliers, and missing values before release. Full citation in `data/README.md`.

The degrees behind the course codes span a wide mix, from Nursing and Veterinary Nursing to Agronomy, Social Service, Journalism, Management, Communication Design, and Basic Education. That spread matters later, because course turns out to be the strongest predictor in the model.

### Overview

**4,424 students, 37 columns.** No missing cells and no duplicate rows, which is the providers' curation rather than luck, and which I verified rather than assumed. The outcome splits Graduate 2,209, Dropout 1,421, Enrolled 794.

A complete column-by-column data dictionary, with types, ranges, units, and missing counts, is committed at **`data/data_dictionary.md`** and reproduced from the data in notebook 02.

### Feature types

| Type | Count | Meaning |
|---|---|---|
| True numbers | 6 | Real quantities where order and distance matter, like admission grade and age |
| Yes or no flags | 8 | Stored as 1 or 0, like gender, scholarship holder, and debtor |
| Coded categories | 9 | Integer codes standing for categories, labels not amounts, like course and parental qualification |
| Ordered count | 1 | Application order, a rank from **0 = first choice** to 9 = last |
| Curricular, leakage | 12 | First and second semester performance, held out of the model |
| Target | 1 | Dropout, Graduate, Enrolled |

The coded categories are the trap. A course coded 9500 is not greater than one coded 33, it is a different course. Treating these as numbers, scaling them or measuring plain correlation on them, is wrong, so they are handled as categories throughout.

### First look at disparity

Four attributes drive the fairness work in Step 5. Against an overall rate of 39.1 percent, the gaps are already large before any model exists.

| Group | Dropout rate | n |
|---|---|---|
| Men | 56.1 percent | 1,249 |
| Women | 30.2 percent | 2,381 |
| No scholarship | 48.4 percent | 2,661 |
| Scholarship holder | 13.8 percent | 969 |
| Debtor | 75.5 percent | 413 |
| Not a debtor | 34.5 percent | 3,217 |
| Age 17 to 20 | 26.1 percent | 2,080 |
| Age 21 to 23 | 40.6 percent | 473 |
| Age 24 to 30 | 66.5 percent | 499 |
| Age 31 plus | 61.4 percent | 578 |

**These numbers come back in Step 5, and they are the reason the fairness audit had to be redone.** A gap this large in the world is not something a model invents. Whether the model *amplifies* it is a different question, and it needs a different metric to answer.

One column stands apart. Students **not up to date on tuition drop out 94.0 percent of the time**, against 30.7 percent for those who are. Only 486 students are in that state. That is not a normal background fact. A student who has stopped paying is often a student already leaving. It is a near-outcome signal, kept but watched, and Step 3 measures exactly what the model would lose without it.

---

## Step 3. Preprocessing, EDA, and Feature Engineering

### Cleaning and the leakage guard

The twelve curricular columns record how many courses a student took, passed, and failed in the first and second semesters. That performance predicts dropout almost perfectly, because a student failing courses is a student already leaving. A model built on it would score high and help no one. All twelve are dropped, so the model sees only what is known at enrollment. A deliberate trade of raw accuracy for a warning that leaves time to act.

### The split comes first

Train and test split before any feature work that looks at the outcome, stratified on dropout, fixed seed. 2,904 training students, 726 test.

**Every decision in this notebook is made on training folds, including the tuition check below.** My first version of that check compared models on the held-out test set, which was a quiet leak. Any decision made by looking at test performance puts test information into the model, however small the decision looks.

### What predicts dropout

**True numbers, Mann-Whitney U.** Grades and age are skewed, so the test assumes no normal shape. Age has the strongest link, older entrants drop out more. Admission grade and previous qualification grade follow. Unemployment and inflation show no real difference (p = 0.75 and 0.21).

![Continuous features by outcome](figures/continuous_by_dropout.png)

**Categories, chi-square and Cramér's V.** Chi-square answers whether a link is real. Cramér's V rescales it to a 0-to-1 strength. Used together they rank by a link that is both real and meaningful. On 3,630 rows almost everything is significant, so V does the real ranking.

| Feature | Cramér's V |
|---|---|
| Tuition fees up to date | 0.437 |
| Course | 0.346 |
| Application mode | 0.330 |
| Scholarship holder | 0.321 |
| Debtor | 0.269 |
| Gender | 0.255 |

**Multicollinearity.** Every VIF on the true numbers is under 2. The linear model is safe on that front, but VIF only looks at the six numerics. It says nothing about the one-hot columns, or about any feature built later. That gap turns out to matter enormously.

![Cramér's V between categories](figures/cramers_v_matrix.png)

The matrix above shows two pairs at a perfect **V = 1.000**. I noted that and moved on, which was a mistake.

### The tuition status check

Tuition status leads the ranking, and its 94 percent dropout rate makes it a near-outcome suspect. A logistic regression trained with and without it, scored on **cross-validation folds inside the training set**.

| Version | CV PR AUC | CV ROC AUC |
|---|---|---|
| With tuition status | 0.817 | 0.848 |
| Without tuition status | 0.765 | 0.826 |

Removing it costs five points of PR AUC. Too much to hand back for a risk that is a judgement call rather than a demonstrated leak, and the curricular columns are the demonstrated leak. The flag stays, with a note. A school acting on this model is acting, in large part, on who is behind on payments.

### The feature I threw away

This is the part of the project I would defend first, and it started as a failure.

I built a **socioeconomic pressure** score, one point each for owing money, falling behind on tuition, and holding no scholarship. The idea felt right and the numbers agreed. It ranked **first** on mutual information, ahead of every raw column in the file.

Then the coefficients came back wrong. Not weak. **Wrong.** The model said owing the school money makes a student *less* likely to drop out, and holding a scholarship makes them *more* likely. The data says the opposite, loudly.

The cause was an exact linear dependency. Rearranged, the score minus its three source flags returns `2` on every single training row. It was not new information. It was three columns the matrix already held, added together. So there was no unique answer to how much weight each column should carry, and the penalty picked one split at random. That arbitrary split was what I had been reading as a coefficient.

A rank check on the design matrix made it visible.

| | Old matrix | Fixed matrix |
|---|---|---|
| Columns | 216 | **84** |
| One-hot blocks | 9 | 6 |
| Exact linear dependencies | **19** | **6** |

Six blocks give six dependencies by construction. A one-hot block that keeps every level always sums to the intercept. The old matrix had **ten more than it should**. Here is what those ten did.

| Feature | Coefficient, score in matrix | Coefficient, score removed | Actual dropout rate |
|---|---|---|---|
| Debtor | **−0.465** | **+1.006** | 76% vs 34% |
| Scholarship holder | **+0.138** | **−1.344** | 14% vs 49% |
| Tuition fees up to date | −1.480 | −2.913 | 31% vs 94% |

Two signs flip. And the reason the feature had to go is not the one I expected. It barely moved accuracy at all, because it never carried information the model did not already have. What it did was make the model **unreadable**. Every explanation in Step 5 is read off these coefficients, so a sign flip here becomes a sentence that tells a student the opposite of the truth.

**A feature that costs nothing and explains nothing is not a harmless addition. It is a liability.**

The same rank check then caught two more repeats, both of which the Cramér's V matrix had already flagged at 1.000.

- `International` equals `(Nacionality != 1)` on **100 percent** of rows. The same column twice.
- `Daytime/evening attendance` is decided entirely by `Course`. **Zero of 17** courses run both a day and an evening class.

### Features that replace, instead of repeat

The rule that came out of this is short. **A feature built from columns that stay in the matrix adds nothing to a linear model.** The encoding already holds that information. A feature earns its place only by replacing its sources, or by saying something the raw columns cannot say alone.

| Feature | What it does | Replaces |
|---|---|---|
| `mother education tier`, `father education tier` | Folds ~30 nominal qualification codes onto a 0-to-3 ladder, keeping the order the codes throw away | Both parental qualification columns (63 dummies → 2) |
| `first generation` | On when neither parent reached higher education | Nothing. This is genuinely new, invisible until the two columns are read together |
| `mother isco`, `father isco` | Folds occupation codes to ISCO major groups | Both parental occupation columns (71 dummies → 12) |
| `application route` | Folds 18 application-mode codes into 5 routes an admissions officer would recognise | Application mode |
| `mature entry` | Age ≥ 23, Portugal's *Maiores de 23* route. Age enters as a straight line, but risk breaks at that route, and a straight line cannot bend | Nothing. It bends the age term |

All of it lives in `src/features.py`, so Steps 4 and 5 read one definition rather than passing copies around. Nothing in it reads the outcome.

### Feature selection, actually applied

Mutual information on the model's real feature set.

| Feature | MI |
|---|---|
| Tuition fees up to date | 0.1033 |
| Age at enrollment | 0.0663 |
| Course | 0.0623 |
| Scholarship holder | 0.0575 |
| Previous qualification (grade) | 0.0559 |
| **mature entry** | **0.0534** |

`mature entry` lands sixth, below the raw age column it bends. That is the honest order, since mutual information already sees the whole age signal. Whether the bend earns its place is a modeling question, not one this ranking settles.

Last time I printed a ranking like this, wrote a sentence about which features were weakest, and then trained on every single one of them anyway. **A ranking that changes nothing is not feature selection. It is a chart.** So this time four columns actually come out, `Nacionality`, `International`, `Daytime/evening attendance`, `Educational special needs`, and the cost gets measured.

| | CV PR AUC | Columns |
|---|---|---|
| Every raw feature | 0.817 | 215 |
| Engineered and selected | **0.819** | **84** |

Sixty percent of the columns gone, and the score went slightly **up**. The gain is inside the noise, and that is fine. Cutting 131 columns for free is the result.

### Dimensionality reduction

![PCA scree](figures/pca_scree.png)

PCA on the nine numerics takes **seven of nine** components to reach 90 percent of the variance. Nothing to compress, which matches the low VIF.

That is an honest negative result, and last time I left it there. **A scree plot that no model ever consumes is not a dimensionality reduction method used.** So PCA rides into Step 4 as a seventh model and has to earn its seat against the other six.

---

## Step 4. Modeling and Comparison

Seven models, tuned inside five-fold stratified cross-validation on the training set. Class weights handle the mild imbalance, a light touch that fits a 39/61 split, rather than synthetic oversampling. The test set is scored once, at the end. Every model's best hyperparameters are saved to `models/metrics.json`, so every row here can be rebuilt.

| Model | CV PR AUC | Test PR AUC | ROC AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| XGBoost | **0.824** | 0.828 | 0.857 | 0.746 | 0.665 | 0.703 |
| Random forest | 0.820 | 0.823 | 0.855 | 0.725 | 0.669 | 0.696 |
| **Logistic regression** | 0.819 | 0.822 | 0.851 | 0.750 | 0.670 | 0.708 |
| Logistic regression + PCA | 0.818 | 0.821 | 0.852 | 0.757 | 0.666 | 0.708 |
| SVM | 0.808 | 0.817 | 0.858 | 0.722 | 0.704 | 0.713 |
| MLP (neural net) | 0.752 | 0.753 | 0.798 | 0.669 | 0.667 | 0.668 |
| Decision tree | 0.723 | 0.699 | 0.762 | 0.683 | 0.601 | 0.639 |

**PCA lands fourth**, one thousandth behind plain logistic regression. Six components in place of nine numerics costs almost nothing and buys almost nothing. That is a measured answer, and it is the reason PCA does not ship. A principal component is a blend of nine columns, so a student flagged by it could never be told why in a sentence that means anything.

**The MLP loses**, below everything except the lone decision tree. That answers the question it was there to ask. Deep learning does not beat trees, or a straight line, on data this small and tabular.

### Is the XGBoost lead real?

XGBoost leads by 0.005. Before taking it, look at how much the five folds disagreed.

| Model | Mean | Std | Folds |
|---|---|---|---|
| Logistic regression | 0.819 | **0.020** | 0.796, 0.841, 0.812, 0.801, 0.846 |
| XGBoost | 0.824 | 0.015 | 0.800, 0.835, 0.822, 0.817, 0.844 |

**The gap is 0.005. The folds swing by 0.020.** The lead is inside the noise. Rerun with a different seed and the order could flip. XGBoost is not better than logistic regression on this data. It is tied with it, and it happened to land on top.

![Fold-to-fold spread](figures/model_fold_noise.png)

The chart is the argument. Each dot is one fold. The two ranges overlap almost completely, and the distance between the means disappears inside them.

**The choice is logistic regression.** Not because it scores highest, because it does not. Because when two models are tied, the tiebreak should be something that matters, and here that is whether a student can be told why they were flagged. A logistic coefficient is a number a tutor can read out loud. A boosted ensemble of 300 trees is not.

### The threshold nobody chose

Every recall figure above sits at a probability cutoff of 0.5. **Nobody picked 0.5.** It is a library default, and on this data it happens to flag 44 percent of the cohort, which makes recall-at-0.5 a fact about the default, not a fact about the model.

Step 1 wrote down a business case and then never used it again. The cutoff is exactly where it belongs. It is chosen on **out-of-fold predictions inside the training set**, then applied once to test.

| Rule | Threshold | Recall | Precision | Flagged |
|---|---|---|---|---|
| Default 0.5 | 0.50 | 0.750 | 0.670 | 43.8% |
| Capacity 50% | 0.38 | 0.838 | 0.609 | 53.9% |
| **Capacity 45%** | **0.45** | **0.796** | **0.644** | **48.3%** |
| Capacity 40% | 0.50 | 0.750 | 0.670 | 43.8% |
| Capacity 35% | 0.57 | 0.697 | 0.723 | 37.7% |
| Capacity 30% | 0.64 | 0.630 | 0.768 | 32.1% |

**The default is capacity 40 percent.** That is all it ever was, a staffing decision made by accident, by a library default. Move it to 45 percent and recall climbs to **0.796** on the same model and the same data.

One thing to be precise about, since this section says "probability" a great many times. `class_weight="balanced"` shifts the intercept to make the rarer class easier to reach, and in doing so it **decalibrates** the outputs. So 0.445 is not a claim that a student has a 44.5 percent chance of dropping out. It is a position in a ranking.

That is fine, and it is worth explaining why rather than waving at it. Nothing downstream needs the number to be a true probability. The capacity rule picks a **quantile** of the out-of-fold scores, which only needs the ranking to be right. The expected-value table below counts **realized outcomes** at each cutoff rather than trusting the scores to be probabilities. And Step 5 commits to never showing the score to a student at all. These are ranking scores, and only the ranking is ever used.

Then the Step 1 numbers get a say.

| Outreach cost | Best threshold | Flagged | Recall | Value per 1,000 students |
|---|---|---|---|---|
| 50 EUR | 0.08 | 93.1% | 0.996 | 199,022 EUR |
| 150 EUR | 0.34 | 59.4% | 0.870 | 125,289 EUR |
| 300 EUR | 0.61 | 34.4% | 0.658 | 58,967 EUR |

![Expected value against cutoff](figures/expected_value.png)

The economics say **flag broadly**. A saved student returns 630 euros in expectation against 150 to reach one, so at the middle cost the model wants to contact 59 percent of the cohort. That is a strange thing for a targeting tool to say, and it is worth sitting with rather than hiding. When a save is worth four times what a contact costs, being wrong is cheap and missing someone is not. There is barely any triage left to do.

**The chart caught me being sloppy, and the table had let me get away with it.** I had written that staff time runs out long before the economics do. Look at the 300 EUR curve. It peaks at **0.61, tighter than the capacity line**. At that cost the economics want *fewer* flags than the tutoring team could handle, and capacity is not the binding constraint at all.

So the honest version carries a condition. Staff capacity binds at 50 and 150 EUR per contact. At 300 EUR the money binds first. Which regime the school is in depends on a number I do not have, and three rows of a table let me write straight past that. The curve does not.

What survives either way. **The ranked list is the product, and the binary flag is not.** A tutor working down a list from most at risk to least does not need a cutoff, and does not care which constraint binds.

A cutoff is still needed to audit fairness and report one honest number, so it goes at **0.445**, the capacity-45-percent point, which assumes the capacity-limited regime, the case at the low and middle outreach costs.

And the cost of choosing capacity over the optimum is small, which is worth a number rather than a wave. The rows above are each cost's profit-optimal cutoff. At the deployed cutoff of 0.445 the value comes down a little, to **171,942, 123,595, and 51,074 EUR per 1,000 students** at 50, 150, and 300 EUR a contact. At the middle cost that is 1,694 below the optimum's 125,289. Fitting the tool to what the tutoring team can staff is very nearly free.

### At the operating point

![The operating point](figures/operating_point.png)

| | Not flagged | Flagged |
|---|---|---|
| **Actually graduated** | 317 | **125** |
| **Actually dropped out** | **58** | 226 |

226 leavers caught. 58 missed. And **125 graduates contacted who were never going to leave**. Those 125 are the real cost of this tool, and they are not free. A student pulled into a retention conversation they did not need has been told, in effect, that a system thinks they are failing.

Step 5 asks who those 125 are. The answer is not evenly spread.

---

## Step 5. Ethics, Bias, and Fairness

### How the model decides

![What actually drives the risk score](figures/shap_importance.png)

**This chart was wrong the first time, and wrong in a way that was hard to see.**

It plotted one bar per column of the design matrix. But `Course` is not a column. It is seventeen one-hot columns. So its importance got chopped into seventeen small bars, none big enough to notice, while a single flag like tuition status kept all its weight in one place. High-cardinality features were being buried by the encoding, and I was reading the result as if it meant something.

Summed back onto the original features.

| Feature | Mean absolute SHAP |
|---|---|
| **Course** | **1.126** |
| Tuition fees up to date | 0.680 |
| **mother isco** | 0.509 |
| Scholarship holder | 0.507 |
| mature entry | 0.436 |
| Gender | 0.301 |

*(A second bug lived here too. Handed a bare array, SHAP's LinearExplainer quietly subsamples the background to 100 rows, unseeded, and the numbers then drift on every run. Tuition came out at 0.90 one run and 0.68 the next. The explainer now uses the full 2,904-row background. An explanation that changes when nothing changed is not an explanation.)*

One caution on that ordering, because it is thinner than it looks. Mother's occupation reads 0.509 and scholarship holder reads 0.507, two thousandths apart. Step 4 refused to call a 0.005 lead real against a measured fold swing, and nobody has measured the noise on a SHAP ranking, so this report does not claim a rank there. It does not need one. Course at 1.126 is far ahead of everything, and a parent's occupation sits well above admission grade at 0.245 either way.

**That changes what this project is about.** The story I had was a money story, tuition, scholarship, debt. It is still there and still strong. But the thing above it is **which course a student enrolled in**, and sitting level with the money signals is **what their mother does for a living**.

Neither is a warning a student can act on. Nobody chooses their mother's occupation, and by the time the model runs, the course is already picked. So the model is not mostly finding students in trouble. **It is substantially finding students who started from further back.**

![Partial dependence with ICE curves](figures/pdp.png)

Risk climbs with age and falls with admission grade. The thin ICE lines run parallel, which is what a linear model with no interactions should look like. A check passing, not a finding, but worth having on the page, because an averaged partial-dependence line can hide a feature that pushes half the students one way and half the other.

### Limits, stated honestly

- **Imbalance.** 39 percent is mildly uneven. Class weights handle it. PR AUC leads.
- **Leakage.** The model uses enrollment-time features only. Tuition status is the softer, watched case, and its cost was measured **on training folds**, not on the test set.
- **Unreadable coefficients.** This limit is here because it nearly ruined the project. Every explanation is read off the model's weights, so a weight carrying the wrong sign becomes a sentence that tells a student the opposite of the truth. The rank check is now a standing guard. Six dependencies against six one-hot blocks, and not one more.
- **Overfitting.** Train PR AUC 0.840 against test 0.822, a gap of 0.018. The model learned the pattern, not the noise.
- **Scope.** IPP-style institution only.

### The fairness audit, and the metric I was using wrong

The first audit ran demographic parity on all four attributes, found every one below the 0.8 line, and concluded the model fails on all four. **That conclusion was too easy, and it was partly wrong.**

Demographic parity asks whether the model flags each group at the same rate. That is the right question when the groups do not really differ. It is the wrong question when they do. Debtors drop out at 76 percent and non-debtors at 34. Demanding equal flagging across that would be demanding the model ignore a 42-point difference it can plainly see. **That is not fairness. That is asking to be wrong on purpose.**

So the real-world gap goes into the table, next to the metric. If the model's flagging gap is roughly the size of the actual outcome gap, the model is **reporting** a disparity. If it is bigger, the model is **making** one.

| Attribute | Real gap in outcomes | Flagging gap (DP) | DI ratio | Error gap (EO) | Recall gap |
|---|---|---|---|---|---|
| **Gender** | 0.238 | **0.350** | 0.496 | 0.290 | **0.177** |
| Scholarship | 0.318 | 0.500 | 0.171 | 0.385 | 0.283 |
| **Debtor** | 0.392 | **0.369** | 0.545 | 0.199 | 0.199 |
| Age band | 0.433 | 0.558 | 0.369 | 0.572 | 0.253 |

**Debtor.** Real gap 0.392, flagging gap 0.369. The model's disparity is *smaller* than the one in the world. It is under-reporting a real difference, not inventing one. Its DI ratio of 0.545 sits well below 0.8, and the old audit called that a failure. It is not. Debt is not a protected class, and equal flagging across it is not something anyone should want.

**Gender.** Real gap 0.238, flagging gap **0.350**. The gap the model creates is *bigger* than the gap in the world. **That is amplification**, and it is the one attribute here where demographic parity asks exactly the right question and gets back a bad answer.

### The harm, which is not the one I expected

| Gender | Flag rate | Recall | False alarm |
|---|---|---|---|
| Male | 0.694 | **0.877** | 0.485 |
| Female | 0.345 | **0.700** | 0.195 |

The model flags men twice as often as women, and their false alarm rate is more than double, so a man who would have graduated fine is far more likely to be pulled into a conversation he did not need.

But look at recall. **0.877 for men. 0.700 for women.**

**Three in ten women who go on to drop out are never flagged at all.** For men it is closer to one in eight. The students this tool exists to reach are being missed, and they are being missed unevenly.

That is the finding. Not "every attribute fails disparate impact," which was true and useless. The model is quietly worse at its actual job for women than for men, and no amount of staring at flagging rates would have shown it. It only appears once you stop counting flags and start counting **errors**.

Whether the model should see gender at all is a separate question, and the answer is not the obvious one. Dropping the column does not remove gender, because the model still sees Course, the strongest thing it uses, and courses in this data are heavily gendered. **Removing a protected attribute from a model that keeps its proxies makes the bias harder to measure without making it any smaller.**

### Mitigation, and what each one charges

Both methods target **equalized odds**, not demographic parity. The harm is an error gap, so the constraint has to be on errors. Both are seeded, so the numbers hold still between runs.

| Approach | Recall | Precision | Accuracy | Flagging gap | Error gap | Recall gap |
|---|---|---|---|---|---|---|
| Baseline at 0.45 | **0.796** | 0.644 | 0.748 | 0.350 | 0.290 | 0.177 |
| Threshold optimizer | 0.609 | 0.783 | 0.781 | 0.100 | 0.026 | 0.026 |
| **Exponentiated gradient** | **0.718** | 0.685 | 0.760 | 0.131 | 0.027 | **0.009** |

Reading those numbers, my first instinct was to say the reduction closes the recall gap *better*, 0.009 against 0.026.

I should not have said that, and Step 4 is the reason why.

In Step 4 I refused to hand XGBoost the win for a 0.005 lead, because the folds swung by 0.020 and a gap smaller than the noise is not a gap. **Then I came here and quoted fairness numbers to three decimals off a single 726-row split without ever asking what the noise was.** The rigor was pointing in one direction only. So I went back and asked. 2,000 seeded bootstrap resamples of the test set.

| | Point estimate | 95% interval | |
|---|---|---|---|
| Baseline recall gap | +0.177 | [+0.082, +0.274] | **real** |
| Threshold optimizer recall gap | −0.026 | [−0.142, +0.090] | indistinguishable from zero |
| Exponentiated gradient recall gap | −0.009 | [−0.118, +0.101] | indistinguishable from zero |
| **TO gap minus EG gap** | +0.008 | [−0.057, +0.076] | **tied** |
| **EG recall minus TO recall** | **+0.109** | **[+0.075, +0.146]** | **real** |

Three things fall out of that.

**The harm is real.** The 0.177 gap never crosses zero across two thousand resamples. The model is worse at finding female dropouts, every time. This finding stands.

**But the third decimal was noise, and I was reading it.** The recall gap rests on **130 female dropouts and 154 male ones**, not 726 students, which is the size of the test set, not the size of the thing the metric depends on. **0.009 is not better than 0.026. They are the same number wearing different clothes.** Catching myself making the exact error I had refused to make one notebook earlier is a humbling way to learn that rigor is not a mood you can be in. It is a check you run.

**And exactly one difference survives.** The recall cost, at +0.109 with an interval nowhere near zero. Which means the decision was never really about fairness at all.

### The verdict, on the one thing that survives

**The two mitigations are tied on fairness.** Both close a gap that was genuinely there. Neither is measurably better at it.

**They are not tied on cost, and that is the entire decision.**

The **threshold optimizer** drops recall from 0.796 to **0.609**. Nineteen points. Look at how it got there. It closed the gap between men and women largely by catching fewer people, so it bought fairness by missing more students of both kinds. Accuracy went *up* while the tool got *worse* at its job. A useful reminder of how little accuracy is worth on a problem shaped like this.

The **exponentiated gradient** drops recall to **0.718**. Eight points. Same fairness, and it keeps eleven points of recall the other method throws away.

**Take the exponentiated gradient.** Eight points of recall is still a real cost, roughly thirty fewer leavers caught per thousand students, and it should be written down plainly rather than buried. What it buys is that those thirty are not disproportionately women.

The other option was to fix the gap by helping fewer people. That is not a fix.

**One honest note on how this choice was made.** I picked between these two by reading their **test-set** scores. In Step 3 I caught myself settling the tuition question on the test set and moved it onto cross-validation folds, and I was pleased with myself for it. This is the same act. It is more defensible, the final comparison of two candidates at the end, but by the standard I set for myself in Step 3, a stricter version selects the mitigation out-of-fold and touches test exactly once, to report. I am naming it rather than hoping nobody notices.

### Residual risk, and what I would tell the school

The model works, and it works on things students cannot change. Course first, far ahead of everything, and a parent's occupation level with the money signals rather than below them. Tuition and scholarship, the money problems the school could actually do something about, sit around them.

Which suggests the flag is the wrong shape for the intervention, and Step 4 reached the same conclusion from the economics. A ranked list, read from the top, with the reason attached, is something a tutor can work with. A binary at-risk label attached to a student's record is something that follows them around.

Gender is mitigated. Scholarship and debtor are not, and that is the right call. Their gaps are real and the model reports rather than invents them. **Age band is genuinely unresolved** and needs its own pass.

Three things I would insist on before this goes near a student.

1. **The score is never shown to the student as a number.** A person told they are 78 percent likely to fail has been handed a prediction, not help.
2. **Anything the score is built from that the school could change, the school should change.** Tuition status is the second strongest signal, and a payment plan is a cheaper intervention than a tutor.
3. **The recall gap gets re-measured every intake.** It was 0.177 on this cohort, with a bootstrap interval of roughly [0.08, 0.27]. Wide, because it rests on 130 female dropouts. It is not the kind of thing that stays fixed on its own, and one cohort is not enough to call it settled.

---

## Reproducibility

```
notebooks/          01 to 05, one per step, run in order, outputs committed
src/                data loader, path helpers, and the feature module
data/               raw (fetched, not committed), processed, and the data dictionary
models/             saved preprocessor, model, threshold, and full metrics with hyperparameters
reports/            this report and its figures
.python-version     3.13, read by uv
requirements.txt    exact pins, read off the environment that produced these outputs
```

```bash
./setup.sh                  # uv venv on Python 3.13, install pinned deps, fetch dataset
source .venv/bin/activate
./run_notebooks.sh          # my own script, runs 01 to 05 in order and saves the outputs
```

I wrote `run_notebooks.sh` because re-running five notebooks by hand, in the right order, on a clean kernel, and remembering to save each one, is a thing I got wrong more than once. The script does it in a single command. It also pins the interpreter to `.venv` and refuses to fall back to system Python, then checks the installed versions against the pins before it runs anything.

Order is not optional. 04 and 05 read files that 03 and 04 write, and running them out of sequence does not error loudly. It errors quietly against a stale artifact, which is worse.

Every number in this report is produced by running the notebooks in order against the real data with fixed seeds. **Three sources of silent drift were found and closed**, and all three are worth naming, because not one of them announced itself. In each case the code ran to completion and simply wrote down a different answer.

- **SHAP's `LinearExplainer`**, handed a bare array, silently subsamples its background to 100 rows, unseeded. Tuition status came out at 0.90 on one run and 0.68 on the next.
- **`ExponentiatedGradient.predict`** returns a draw from a *randomized mixture* of classifiers, not a single fitted model. Off one fitted object, the recall gap read 0.012 on one call and 0.001 on the next.
- **The environment itself.** `requirements.txt` was at one point rewritten against versions read off a different machine, which pinned five packages *behind* the environment that had produced every number here. Installing it would have downgraded pandas, numpy, scikit-learn, scipy and matplotlib, re-run cleanly, and quietly disagreed with this report.

The first two are seeded now. The third is checked on every run.

Everything else, the split, the CV folds, mutual information, PCA, all seven estimators, both mitigations, and the Step 5 bootstrap, is seeded at 42. The notebooks are committed **with their outputs**, so every table and chart here can be checked on GitHub without running anything.

---

## Conclusion

The project delivers a working early-warning model for student dropout that uses only enrollment-time information, catches **79.6 percent** of true dropouts at a cutoff chosen against the school's real staffing capacity, and stays simple enough to explain itself.

It also delivers four findings that only appeared after I went back and fixed something I had gotten wrong.

**A feature that adds no information cannot help a model, but it can still break it**, and it breaks it in the place that matters most, which is the model's ability to say why.

**The model's strongest signals are things students cannot change.** Course, and parental occupation. That is not a bug to be patched out. It is what the data says, and a school deploying this should know it before it starts flagging people.

**The harm was not where the standard fairness metric was pointing.** Demographic parity said all four attributes failed. The truth was that three of them were real gaps being reported, and one, gender, was a real gap being amplified, with the damage showing up as a recall gap that no flagging-rate metric could see.

**And rigor does not generalize on its own.** I spent Step 4 refusing to call a 0.005 lead real against a 0.020 fold swing, and then walked into Step 5 and quoted fairness numbers to three decimals off 130 female dropouts as though the noise had stayed behind in the other notebook. It had not. The bootstrap that fixed it did not overturn the finding, the recall gap is real, but it did overturn the *reason* I gave for choosing a mitigation. Skepticism is not a stance you adopt once and carry around. It is a check, and it has to be run everywhere it applies.

A dropout model is not a scoreboard. It is a decision about which students get help. The right way to use this one is as a ranked prompt for a support team, checked by a person, aimed at starting a conversation early enough to matter.

---

## References

- Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. DOI 10.24432/C5MC89. License CC BY 4.0.
- Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: a case study. *Trends and Applications in Information Systems and Technologies* (WorldCIST 2021), AISC vol. 1365, Springer, pp. 166–175. DOI 10.1007/978-3-030-72657-7_16.
- Funding: SATDAP, Capacitação da Administração Pública, grant POCI-05-5762-FSE-000191, Portugal.
