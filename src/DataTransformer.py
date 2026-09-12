import numpy as np
import pandas as pd


class DataTransformer:
    """Feature engineering + encoding for the student score dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # -- Engineered features --------------------------------------------------

    def add_sleep_duration(self, sleep_col: str = "sleep_time",
                           wake_col: str = "wake_time",
                           new_col: str = "sleep_duration") -> "DataTransformer":
        """Derive hours of sleep from the clock-time columns.

        Bedtimes straddle midnight (21:00 to 03:00), so the difference is taken
        modulo 24. The two raw clock columns carry no extra signal afterwards and
        are dropped by the Compiler.
        """
        if sleep_col not in self.df.columns or wake_col not in self.df.columns:
            return self

        sleep_h = self._to_hours(self.df[sleep_col])
        wake_h = self._to_hours(self.df[wake_col])
        self.df[new_col] = (wake_h - sleep_h) % 24
        print(f"[ok] Added {new_col} (mean={self.df[new_col].mean():.2f}h, "
              f"min={self.df[new_col].min():.1f}, max={self.df[new_col].max():.1f})")
        return self

    @staticmethod
    def _to_hours(series: pd.Series) -> pd.Series:
        """Convert H:MM / HH:MM strings to float hours."""
        parts = series.astype("object").str.strip().str.split(":", expand=True)
        return parts[0].astype(float) + parts[1].astype(float) / 60.0

    def add_class_composition(self, male_col: str = "n_male",
                              female_col: str = "n_female") -> "DataTransformer":
        """Build class_size and gender_ratio from the headcount columns.

        n_male and n_female are highly collinear with their sum, so the Compiler
        drops the raw counts once these two derived features exist.
        """
        if male_col not in self.df.columns or female_col not in self.df.columns:
            return self

        self.df["class_size"] = self.df[male_col] + self.df[female_col]
        # Guard against divide-by-zero in all-female classes.
        self.df["gender_ratio"] = np.where(
            self.df[female_col] > 0,
            self.df[male_col] / self.df[female_col].replace(0, np.nan),
            np.nan,
        )
        median_ratio = self.df["gender_ratio"].median()
        n_inf = int(self.df["gender_ratio"].isna().sum())
        self.df["gender_ratio"] = self.df["gender_ratio"].fillna(median_ratio)
        print(f"[ok] Added class_size (mean={self.df['class_size'].mean():.1f}) and "
              f"gender_ratio ({n_inf} undefined -> median {median_ratio:.2f})")
        return self

    def add_study_intensity(self, hours_col: str = "hours_per_week",
                            attendance_col: str = "attendance_rate") -> "DataTransformer":
        """Interaction term: study hours weighted by how often the student shows up."""
        if hours_col in self.df.columns and attendance_col in self.df.columns:
            self.df["study_intensity"] = self.df[hours_col] * self.df[attendance_col] / 100.0
            print(f"[ok] Added study_intensity "
                  f"(mean={self.df['study_intensity'].mean():.2f})")
        return self

    # -- Binning / encoding ---------------------------------------------------

    def apply_quantile_bins(self, col: str, new_label_col: str,
                            q: int = 10) -> "DataTransformer":
        """Bin a numeric column into q quantile buckets with readable labels."""
        if col not in self.df.columns:
            return self
        bins, edges = pd.qcut(self.df[col], q=q, retbins=True, labels=False, duplicates="drop")
        labels = [f"{edges[i]:.0f}-{edges[i + 1]:.0f}" for i in range(len(edges) - 1)]
        self.df[new_label_col] = bins.map(lambda x: labels[int(x)] if pd.notna(x) else np.nan)
        print(f"[ok] Binned {col} -> {new_label_col} ({len(labels)} quantile buckets)")
        return self

    def label_encode(self, col: str, mapping: dict, suffix: str = "_code") -> "DataTransformer":
        """Map a categorical column to integers using an explicit dict."""
        if col not in self.df.columns:
            return self
        target = col + suffix
        self.df[target] = self.df[col].astype("object").str.upper().map(mapping)
        missing = int(self.df[target].isna().sum())
        if missing:
            unmapped = self.df.loc[self.df[target].isna(), col].unique().tolist()
            print(f"[warn] {missing} unmapped value(s) in {col} -> NaN {unmapped}")
        else:
            print(f"[ok] Encoded {col} -> {target}")
        return self

    def binary_encode(self, cols: list, mapping: dict = None,
                      suffix: str = "_code") -> "DataTransformer":
        """Encode binary Y/N (or Yes/No) columns to 1/0."""
        mapping = mapping or {"Y": 1, "N": 0, "YES": 1, "NO": 0}
        for col in cols:
            if col not in self.df.columns:
                continue
            self.df[col + suffix] = self.df[col].astype("object").str.upper().map(mapping)
            n_bad = int(self.df[col + suffix].isna().sum())
            note = f" ({n_bad} unmapped)" if n_bad else ""
            print(f"[ok] Binary-encoded {col} -> {col}{suffix}{note}")
        return self

    def one_hot(self, cols: list, drop_first: bool = True) -> "DataTransformer":
        """One-hot encode nominal columns inside the DataFrame.

        Used for tree models fed raw frames; the sklearn pipeline in DataModeler
        does its own encoding for the linear baselines.
        """
        existing = [c for c in cols if c in self.df.columns]
        if not existing:
            return self
        before = self.df.shape[1]
        self.df = pd.get_dummies(self.df, columns=existing, drop_first=drop_first, dtype=int)
        print(f"[ok] One-hot encoded {existing}: {before} -> {self.df.shape[1]} columns")
        return self

    def get(self) -> pd.DataFrame:
        return self.df
