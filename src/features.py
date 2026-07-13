"""Feature engineering and selection for the capstone.

Step 3 builds the model's feature set here, so Steps 4 and 5 read one definition
instead of passing copies around.

Two rules shaped this file. Both came out of a mistake, and the mistake is worth
writing down, because it took me a while to see it.

My first attempt added a "socioeconomic pressure" score built from three flags
that stayed in the matrix right next to it. The score looked strong. It ranked
first on mutual information. Then the coefficients came back saying that owing
money lowers dropout risk, which is the exact opposite of what the data says. The
cause was an exact linear dependency. The score was only a sum of columns the
model already had, so the weight could be split between them any way at all, and
the penalty picked one split at random.

Rule one, then. A feature built from columns that stay in the matrix adds nothing
to a linear model. The one-hot encoding already holds that information. The
features below REPLACE their source columns instead of sitting beside them.

Rule two. Check the rank. `linear_dependencies` at the bottom counts the exact
dependencies in a design matrix. A one-hot block that keeps every level always
sums to the intercept, so six blocks give six dependencies and that is fine.
Anything above that means a column repeats what the matrix already has, and no
coefficient touching it can be trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Parental education, folded onto a four-rung ladder.
#
# The raw codes are thirty-odd nominal values running from "cannot read" all the
# way to a doctorate. One-hot encoded they cost 63 dummy columns, and they throw
# away the only thing that matters about them, which is their order. The ladder
# keeps the order and costs two columns.
# --------------------------------------------------------------------------
_HIGHER = {2, 3, 4, 5, 6, 40, 41, 42, 43, 44}                     # bachelor's to doctorate
_SECONDARY = {1, 9, 10, 12, 13, 14, 18, 20, 22, 25, 31, 33, 39}   # 10th to 12th year, technical
_BASIC = {11, 19, 26, 27, 29, 30, 37, 38}                         # up to 9th year
_NONE = {35, 36}                                                  # cannot read, or below 4th year

# Code 34 is "Unknown". It maps to nan and then fills to the rung most parents sit
# on, so a missing answer does not quietly read as a low one.
_UNKNOWN_FILL = 2.0


def education_tier(code: int) -> float:
    """Map a parental qualification code onto 0 (none) to 3 (higher education)."""
    if code in _HIGHER:
        return 3.0
    if code in _SECONDARY:
        return 2.0
    if code in _BASIC:
        return 1.0
    if code in _NONE:
        return 0.0
    return np.nan


def isco_group(code: int) -> str:
    """Fold a parental occupation code down to its ISCO major group.

    The source mixes one-digit major groups (0 to 10) with three-digit detailed
    codes, and in the detailed codes the first digit is the major group. Codes 90
    and 99 mean other or blank, so both land in one bucket.
    """
    if code in (90, 99):
        return "other"
    if code < 11:
        return str(code)
    return str(code)[0]


# Eighteen application-mode codes fold down to five routes an admissions officer
# would actually recognise. Cardinality drops, and the mature-entry route becomes
# a named thing instead of hiding as one code among many.
_ROUTE = {
    1: "general", 17: "general", 18: "general",
    39: "over 23",
    42: "transfer", 43: "transfer", 44: "transfer", 51: "transfer",
    7: "international or special", 15: "international or special",
    16: "international or special", 53: "international or special",
    57: "international or special",
}

# Portugal opens a separate admission route, "Maiores de 23", at this age.
MATURE_AGE = 23


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features.

    Every rule here is fixed and none of them read the outcome, so running this on
    train and on test applies the same mapping and the test set stays clean.
    """
    f = frame.copy()

    mother = f["Mother's qualification"].map(education_tier)
    father = f["Father's qualification"].map(education_tier)
    f["mother education tier"] = mother.fillna(_UNKNOWN_FILL)
    f["father education tier"] = father.fillna(_UNKNOWN_FILL)

    # First generation, meaning neither parent reached higher education. The
    # education research has treated this as a dropout risk factor for decades.
    # Neither parent column shows it on its own. It only appears once the two are
    # read together, which is the whole point of engineering it.
    f["first generation"] = (
        (f["mother education tier"] < 3) & (f["father education tier"] < 3)
    ).astype(int)

    f["mother isco"] = f["Mother's occupation"].map(isco_group)
    f["father isco"] = f["Father's occupation"].map(isco_group)
    f["application route"] = f["Application mode"].map(_ROUTE).fillna("other")

    # Mature entry. Age goes into the model as a straight line, but dropout risk
    # does not climb smoothly with age. It breaks at the mature-entry route. A
    # straight line cannot bend, so this flag catches the break for it.
    f["mature entry"] = (f["Age at enrollment"] >= MATURE_AGE).astype(int)
    return f


# --------------------------------------------------------------------------
# What the model sees, after engineering and selection.
# --------------------------------------------------------------------------

# Source columns the engineered features replace. Keeping both sides would put the
# same information into the matrix twice, which is the mistake described at the
# top of this file.
REPLACED = [
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Application mode",
]

# Columns dropped by feature selection. Each one has a reason.
#   Nacionality and International   the same column twice. International is
#                                   exactly (Nacionality != 1) on every row.
#   Daytime/evening attendance      Course decides it. No course runs both.
#   Educational special needs       near-zero Cramer's V, near-zero mutual
#                                   information, and only a handful of students.
DROPPED = [
    "Nacionality",
    "International",
    "Daytime/evening attendance",
    "Educational special needs",
]

MODEL_NUMERIC = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
    "Application order",
    "mother education tier",
    "father education tier",
]

MODEL_NOMINAL = [
    "Marital status",
    "Course",
    "Previous qualification",
    "application route",
    "mother isco",
    "father isco",
]

MODEL_FLAGS = [
    "Displaced",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "first generation",
    "mature entry",
]

MODEL_FEATURES = MODEL_NUMERIC + MODEL_NOMINAL + MODEL_FLAGS


def linear_dependencies(matrix) -> int:
    """Count the exact linear dependencies in a design matrix, intercept included.

    This is the check that caught the bad feature. A one-hot block that keeps every
    level always sums to the intercept, so a matrix with k blocks carries k
    dependencies and that is expected. If the count comes back higher than the
    number of blocks, some column is repeating information the matrix already has,
    and every coefficient touching it is unreadable.
    """
    Z = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    Zi = np.hstack([Z, np.ones((len(Z), 1))])
    return int(Zi.shape[1] - np.linalg.matrix_rank(Zi))


def group_shap_by_feature(shap_values, encoded_names, nominal=None) -> pd.Series:
    """Sum SHAP values back onto the original column they came from.

    One-hot encoding splits one categorical into many columns, so a chart drawn per
    encoded column chops that feature into a lot of small bars and buries it.
    Course has 17 levels and vanishes this way. Adding the parts back together puts
    it back where it belongs, which turned out to be first.
    """
    nominal = nominal or MODEL_NOMINAL
    mean_abs = np.abs(shap_values).mean(axis=0)

    def original(name: str) -> str:
        if name.startswith("cat__"):
            body = name[len("cat__"):]
            for col in sorted(nominal, key=len, reverse=True):
                if body.startswith(col + "_"):
                    return col
        return name.split("__", 1)[-1]

    grouped = pd.Series(mean_abs, index=[original(n) for n in encoded_names])
    return grouped.groupby(level=0).sum().sort_values(ascending=False)
