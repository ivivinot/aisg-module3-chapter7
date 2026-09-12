import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.impute import SimpleImputer


class DataCleaner:
    """Cleaning + imputation for the student score dataset.

    Every method returns ``self`` so steps can be chained; call ``get()`` for the
    resulting DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # -- Structural -----------------------------------------------------------

    def drop_columns(self, cols: list) -> "DataCleaner":
        existing = [c for c in cols if c in self.df.columns]
        if existing:
            self.df = self.df.drop(columns=existing)
            print(f"[fix] Dropped columns: {existing}")
        return self

    def clean_duplicate(self, subset: list = None, keep: str = "first") -> "DataCleaner":
        """Drop duplicate rows.

        For this dataset student_id is the natural key: 900 records are exact
        re-entries of the same student and would otherwise leak across the
        train/test split, inflating the scores.
        """
        subset = [c for c in (subset or []) if c in self.df.columns] or None
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        print(f"[fix] Removed {before - len(self.df):,} duplicate rows "
              f"(subset={subset or 'all columns'}) -> {len(self.df):,} remain")
        return self

    def drop_missing_target(self, target: str) -> "DataCleaner":
        """Drop rows with no target.

        These cannot be learned from and must not be imputed -- imputing the
        label would fabricate the very signal we are modelling.
        """
        before = len(self.df)
        self.df = self.df.dropna(subset=[target]).reset_index(drop=True)
        dropped = before - len(self.df)
        print(f"[fix] Dropped {dropped:,} rows with missing {target} "
              f"({dropped / before:.2%}) -> {len(self.df):,} remain")
        return self

    # -- Categorical text -----------------------------------------------------

    def standardise_categories(self, cols: list) -> "DataCleaner":
        """Strip whitespace and upper-case labels so Clubs and CLUBS collapse."""
        for col in cols:
            if col not in self.df.columns:
                continue
            before = self.df[col].nunique()
            self.df[col] = self.df[col].astype("object").str.strip().str.upper()
            print(f"[ok] {col}: {before} -> {self.df[col].nunique()} distinct levels")
        return self

    def fill_categorical(self, col: str, value: str = "NONE") -> "DataCleaner":
        """Fill missing categoricals with an explicit level.

        For CCA a missing value means no co-curricular activity was recorded,
        which is information rather than noise -- so it becomes its own category
        instead of being imputed with the mode.
        """
        if col not in self.df.columns:
            return self
        n = int(self.df[col].isna().sum())
        self.df[col] = self.df[col].fillna(value)
        print(f"[ok] {col}: filled {n:,} missing values with {value}")
        return self

    def normalise_yes_no(self, cols: list) -> "DataCleaner":
        """Collapse Yes/Y and No/N variants onto a single Y/N spelling."""
        for col in cols:
            if col not in self.df.columns:
                continue
            before = self.df[col].nunique()
            self.df[col] = self.df[col].astype("object").str.strip().str.upper().str[0]
            print(f"[fix] {col}: {before} -> {self.df[col].nunique()} levels "
                  f"{dict(self.df[col].value_counts())}")
        return self

    # -- Numeric --------------------------------------------------------------

    def clean_negative_values(self, include_cols: list) -> "DataCleaner":
        """Take the absolute value in columns that cannot physically be negative."""
        for col in include_cols:
            if col not in self.df.columns:
                continue
            neg = int((self.df[col] < 0).sum())
            if neg:
                self.df[col] = self.df[col].abs()
                print(f"[fix] {col}: converted {neg} negative value(s) to positive")
        return self

    def fix_age(self, col: str = "age", valid: tuple = (15, 16)) -> "DataCleaner":
        """Repair the age column.

        Two defects appear in the raw data: a handful of negatives (-4, -5) and
        ~450 single-digit values (5, 6) that are 15/16 with a dropped leading
        digit. Negatives are made positive, single digits get the 10 restored,
        and anything still outside the valid cohort range is set to NaN and
        imputed with the modal age.
        """
        if col not in self.df.columns:
            return self

        original = self.df[col].copy()
        self.df[col] = self.df[col].abs()
        small = self.df[col] < 10
        self.df.loc[small, col] = self.df.loc[small, col] + 10

        invalid = ~self.df[col].isin(valid) & self.df[col].notna()
        n_invalid = int(invalid.sum())
        if n_invalid:
            self.df.loc[invalid, col] = np.nan

        mode_age = self.df[col].mode(dropna=True)
        fill = float(mode_age.iloc[0]) if len(mode_age) else float(valid[0])
        n_filled = int(self.df[col].isna().sum())
        self.df[col] = self.df[col].fillna(fill)

        n_changed = int((original != self.df[col]).sum())
        print(f"[fix] {col}: repaired {n_changed} value(s) "
              f"({n_invalid} out-of-range -> NaN, {n_filled} imputed with mode={fill:g}); "
              f"now {dict(self.df[col].value_counts())}")
        return self

    def simple_impute(self, subset: list = None, method: str = "median") -> "DataCleaner":
        """Impute numeric columns with mean / median / most_frequent."""
        subset = [c for c in (subset or self.df.select_dtypes(include=[np.number]).columns)
                  if c in self.df.columns]
        missing_before = {c: int(self.df[c].isna().sum()) for c in subset}
        subset = [c for c in subset if missing_before[c] > 0]
        if not subset:
            print("[ok] No numeric missing values to impute")
            return self

        imputer = SimpleImputer(strategy=method)
        self.df[subset] = pd.DataFrame(
            imputer.fit_transform(self.df[subset]), columns=subset, index=self.df.index
        )
        for i, col in enumerate(subset):
            print(f"[ok] {col}: imputed {missing_before[col]:,} missing values "
                  f"with {method} = {imputer.statistics_[i]:.2f}")
        return self

    def impute_fill(self, method: str = "ffill", subset: list = None) -> "DataCleaner":
        """Forward / backward fill -- useful for time-ordered data."""
        subset = [c for c in (subset or self.df.columns) if c in self.df.columns]
        if method == "ffill":
            self.df[subset] = self.df[subset].ffill()
        elif method == "bfill":
            self.df[subset] = self.df[subset].bfill()
        else:
            raise ValueError("method must be ffill or bfill")
        print(f"[ok] Applied {method} on {subset}")
        return self

    def auto_transform_skewed(self, subset: list = None, skew_threshold: float = 0.5,
                              exclude: list = None) -> "DataCleaner":
        """Add log1p / reflect-log copies of skewed numeric columns."""
        exclude = exclude or []
        subset = [c for c in (subset or self.df.select_dtypes(include=[np.number]).columns)
                  if c in self.df.columns and c not in exclude]
        for col in subset:
            s = self.df[col].dropna()
            if s.nunique() < 10:
                continue
            sk = float(skew(s))
            if sk > skew_threshold and s.min() >= 0:
                self.df[col + "_log1p"] = np.log1p(self.df[col])
                print(f"[ok] {col}: right-skew {sk:.2f} -> added {col}_log1p")
            elif sk < -skew_threshold:
                self.df[col + "_reflectlog"] = np.log1p(self.df[col].max() + 1 - self.df[col])
                print(f"[ok] {col}: left-skew {sk:.2f} -> added {col}_reflectlog")
        return self

    # -- Result ---------------------------------------------------------------

    def get(self) -> pd.DataFrame:
        return self.df
