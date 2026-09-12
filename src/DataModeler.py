import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataModeler:
    """Train/val/test splitting and the shared preprocessing ColumnTransformer."""

    def __init__(self, numeric_cols: list = None, categorical_cols: list = None):
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []

    # -- Splitting ------------------------------------------------------------

    def split_data(self, df: pd.DataFrame, target_col: str,
                   val_size: float = 0.1, test_size: float = 0.2,
                   random_state: int = 42, stratify_bins: int = 10):
        """Split into train / validation / test.

        The target is continuous, so stratification is done on quantile bins of
        the score. That keeps the same spread of weak-to-strong students in all
        three splits, which matters here because the school cares specifically
        about the low-scoring tail.
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        strata = self._make_strata(y, stratify_bins)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strata
        )

        val_adjusted = val_size / (1 - test_size)
        strata_temp = self._make_strata(y_temp, stratify_bins)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_adjusted,
            random_state=random_state, stratify=strata_temp
        )

        print(f"[ok] Split -> train: {len(X_train):,} | val: {len(X_val):,} | "
              f"test: {len(X_test):,}")
        print(f"     target mean  train={y_train.mean():.2f} "
              f"val={y_val.mean():.2f} test={y_test.mean():.2f}")
        print(f"     target std   train={y_train.std():.2f} "
              f"val={y_val.std():.2f} test={y_test.std():.2f}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def _make_strata(y: pd.Series, bins: int):
        """Quantile bins of a continuous target, or None if it cannot be binned."""
        try:
            strata = pd.qcut(y, q=bins, labels=False, duplicates="drop")
        except ValueError:
            return None
        counts = pd.Series(strata).value_counts()
        return strata if counts.min() >= 2 else None

    # -- Preprocessing --------------------------------------------------------

    def build_preprocessor(self) -> ColumnTransformer:
        """StandardScaler for numerics, OneHotEncoder for categoricals.

        Scaling is inert for the tree models but required by Ridge/Linear, and
        keeping one shared transformer means every model sees identical inputs.
        """
        transformers = []
        if self.numeric_cols:
            transformers.append(("num", StandardScaler(), self.numeric_cols))
        if self.categorical_cols:
            transformers.append(
                ("cat",
                 OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
                 self.categorical_cols)
            )
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def infer_columns(self, df: pd.DataFrame, target_col: str) -> "DataModeler":
        """Populate numeric_cols / categorical_cols from the frame's dtypes."""
        features = df.drop(columns=[target_col], errors="ignore")
        self.numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = features.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        print(f"[ok] Numeric features ({len(self.numeric_cols)}): {self.numeric_cols}")
        print(f"[ok] Categorical features ({len(self.categorical_cols)}): "
              f"{self.categorical_cols}")
        return self
