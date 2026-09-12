import pandas as pd
import numpy as np


class DataProfiler:
    """Read-only profiling of a DataFrame: shape, dtypes, missingness, cardinality."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def profile(self) -> pd.DataFrame:
        """Summary table: count, dtype, unique, missing, missing%."""
        return pd.DataFrame({
            "count":      self.df.count(),
            "dtype":      self.df.dtypes.astype(str),
            "n_unique":   self.df.nunique(),
            "n_missing":  self.df.isna().sum(),
            "n_missing%": (self.df.isna().sum() / len(self.df) * 100).round(2),
        })

    def duplicate_summary(self, subset: list = None) -> dict:
        """Duplicate counts for the whole row and for an optional key subset."""
        summary = {"n_rows": len(self.df), "n_duplicate_rows": int(self.df.duplicated().sum())}
        if subset:
            existing = [c for c in subset if c in self.df.columns]
            summary["subset"] = existing
            summary["n_duplicate_subset"] = int(self.df.duplicated(subset=existing).sum())
        return summary

    def value_counts_summary(self, exclude_cols: list = None) -> None:
        """Print value counts + percentage for every column (minus exclusions)."""
        exclude_cols = exclude_cols or []
        for col in self.df.columns:
            if col in exclude_cols:
                continue
            vc = self.df[col].value_counts(dropna=False)
            pct = self.df[col].value_counts(normalize=True, dropna=False).mul(100).round(2)
            print(f"\nValue counts - {col} (nunique={self.df[col].nunique()}):")
            print(pd.DataFrame({"count": vc, "percentage": pct}))

    def categorical_summary(self, max_unique: int = 20) -> None:
        """Print value counts only for low-cardinality / object columns."""
        cols = [c for c in self.df.columns
                if self.df[c].dtype == "object" or self.df[c].nunique() <= max_unique]
        for col in cols:
            print(f"\n== {col} | nunique={self.df[col].nunique()} "
                  f"| missing={self.df[col].isna().sum()}")
            print(self.df[col].value_counts(dropna=False).to_string())

    def numeric_summary(self) -> pd.DataFrame:
        """Descriptive stats for numeric columns."""
        return self.df.select_dtypes(include=[np.number]).describe().T

    def outlier_summary(self, subset: list = None, k: float = 1.5) -> pd.DataFrame:
        """IQR fences and outlier counts per numeric column."""
        subset = subset or self.df.select_dtypes(include=[np.number]).columns.tolist()
        rows = {}
        for col in subset:
            s = self.df[col].dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - k * iqr, q3 + k * iqr
            rows[col] = {
                "lower_fence": lo, "upper_fence": hi,
                "n_outliers": int(((s < lo) | (s > hi)).sum()),
                "min": s.min(), "max": s.max(),
                "missing": int(self.df[col].isna().sum()),
            }
        return pd.DataFrame(rows).T
