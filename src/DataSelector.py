import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


class DataSelector:
    """Multicollinearity-based feature selection.

    Engineered features such as class_size and study_intensity are by
    construction correlated with their parents, so a VIF sweep is run before
    modelling to keep the linear baselines stable and interpretable.
    """

    def __init__(self, vif_threshold: float = 6.0):
        self.vif_threshold = vif_threshold
        self.dropped = []

    def drop_columns(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        existing = [c for c in cols if c in df.columns]
        if existing:
            print(f"[fix] Dropping columns: {existing}")
        return df.drop(columns=existing)

    @staticmethod
    def _vif_series(numeric: pd.DataFrame) -> pd.Series:
        """VIF for each column, computed against an explicit intercept.

        statsmodels' variance_inflation_factor regresses each column on the
        others with no constant term. Without an intercept any column whose mean
        is large relative to its spread (attendance_rate, age) scores a huge VIF
        that reflects its offset, not real collinearity -- so a constant is added
        here and excluded from the results.
        """
        with_const = numeric.copy()
        with_const.insert(0, "_const", 1.0)
        values = with_const.to_numpy(dtype=float)
        vif = [variance_inflation_factor(values, i) for i in range(values.shape[1])]
        return pd.Series(vif, index=with_const.columns).drop("_const")

    def _numeric_frame(self, df: pd.DataFrame, exclude: list = None) -> pd.DataFrame:
        exclude = exclude or []
        numeric = df.select_dtypes(include=[np.number]).drop(
            columns=[c for c in exclude if c in df.columns], errors="ignore"
        ).dropna()
        return numeric.loc[:, numeric.nunique() > 1]

    def vif_table(self, df: pd.DataFrame, exclude: list = None) -> pd.DataFrame:
        """VIF for every numeric column (excluding the target and any exclusions)."""
        numeric = self._numeric_frame(df, exclude)
        if numeric.empty:
            return pd.DataFrame(columns=["VIF"])
        return (self._vif_series(numeric)
                .to_frame("VIF")
                .sort_values("VIF", ascending=False))

    def drop_high_vif(self, df: pd.DataFrame, exclude: list = None,
                      max_iter: int = 50) -> pd.DataFrame:
        """Iteratively drop the highest-VIF feature above the threshold.

        `exclude` columns (the target, and any categorical codes you want to
        keep regardless) are never considered for dropping.
        """
        exclude = [c for c in (exclude or []) if c in df.columns]
        numeric = self._numeric_frame(df, exclude)

        for _ in range(max_iter):
            if numeric.shape[1] <= 1:
                break
            vif = self._vif_series(numeric)
            max_vif = vif.max()
            if not np.isfinite(max_vif) or max_vif <= self.vif_threshold:
                break
            worst = vif.idxmax()
            print(f"[fix] Dropping {worst} (VIF={max_vif:.2f})")
            self.dropped.append(worst)
            numeric = numeric.drop(columns=[worst])

        kept = numeric.columns.tolist()
        print(f"[ok] VIF selection done (threshold={self.vif_threshold}). "
              f"Dropped: {self.dropped or 'none'}")
        print(f"[ok] Kept {len(kept)} numeric features: {kept}")

        keep_cols = [c for c in df.columns if c in kept or c in exclude
                     or df[c].dtype == "object"]
        return df[keep_cols]
