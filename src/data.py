"""Shared data loading for the capstone.

One place to read the dataset, so every notebook from Step 2 to Step 5 loads it
the same way and the binary framing is defined once, not copied.

The feature groups below record what kind of thing each column is. Each kind
needs different handling later, so naming them here prevents mistakes in Step 3,
above all treating an integer coded category as if it were a real number.
"""

from __future__ import annotations

import pandas as pd

from .paths import RAW_DIR

TARGET = "Target"

# Six true numbers. The value is a real quantity where order and distance mean
# something.
CONTINUOUS = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Eight yes or no flags, stored as 1 or 0.
BINARY_FLAGS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# Nine categories stored as integer codes. The number is a label, not an amount.
# A course coded 9500 is not greater than one coded 33, it is a different course.
NOMINAL_CODED = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# One ordered count. Application order runs 0 (first choice) to 9 (last).
COUNT_ORDINAL = ["Application order"]

# Twelve curricular records from the first and second semesters. Recorded after
# enrollment, so they leak the outcome. Held out of the model, per Step 1.
LEAKAGE = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

# Four attributes for the Step 5 fairness audit.
SENSITIVE = ["Gender", "Age at enrollment", "Scholarship holder", "Debtor"]


def load_raw() -> pd.DataFrame:
    """Read the raw dataset, with column names cleaned.

    The file is semicolon separated, and its header carries a byte order mark
    and a stray tab, so the names are stripped on load.
    """
    df = pd.read_csv(RAW_DIR / "dropout.csv", sep=";")
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    return df


def load_binary() -> pd.DataFrame:
    """Read the dataset in the binary frame used by the project.

    The Enrolled rows are dropped, since their outcome is not settled. A new
    'dropout' column holds 1 for Dropout and 0 for Graduate.
    """
    df = load_raw()
    df = df[df[TARGET].isin(["Dropout", "Graduate"])].copy()
    df["dropout"] = (df[TARGET] == "Dropout").astype(int)
    return df
