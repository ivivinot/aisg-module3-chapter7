import os

import matplotlib
matplotlib.use("Agg")  # headless-safe: the Compiler writes PNGs, never opens a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


class XGBRegressorTrainer:
    """Train / evaluate / predict for a single regression estimator.

    Despite the name this wraps any sklearn-compatible regressor -- XGBoost is
    simply the default. The preprocessor is folded into a Pipeline so that
    scaling and encoding are fitted on the training fold only.
    """

    def __init__(self, preprocessor=None, params: dict = None, estimator=None,
                 name: str = "XGBoost"):
        self.preprocessor = preprocessor
        self.name = name
        self.params = params or {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.estimator = estimator if estimator is not None else XGBRegressor(**self.params)
        self.model = None

    def train(self, X_train, y_train):
        if self.preprocessor is not None:
            self.model = Pipeline([("prep", self.preprocessor), ("reg", self.estimator)])
        else:
            self.model = self.estimator
        self.model.fit(X_train, y_train)
        print(f"[ok] {self.name} trained on {len(X_train):,} rows.")
        return self

    def evaluate(self, X, y, label: str = "Validation", verbose: bool = True) -> dict:
        preds = self.model.predict(X)
        metrics = {
            "model": self.name,
            "split": label,
            "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
            "MAE": float(mean_absolute_error(y, preds)),
            "R2": float(r2_score(y, preds)),
        }
        if verbose:
            print(f"\n-- {self.name} | {label} ------------------------")
            print(f"RMSE: {metrics['RMSE']:.4f}  |  MAE: {metrics['MAE']:.4f}  "
                  f"|  R2: {metrics['R2']:.4f}")
        return metrics

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """K-fold RMSE on the training data -- guards against a lucky split."""
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=kf,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
        rmse = -scores
        print(f"[ok] {self.name} {cv}-fold CV RMSE: "
              f"{rmse.mean():.4f} (+/- {rmse.std():.4f})")
        return {"model": self.name, "cv_rmse_mean": float(rmse.mean()),
                "cv_rmse_std": float(rmse.std())}

    def predict(self, X) -> pd.Series:
        return pd.Series(self.model.predict(X), name="prediction", index=getattr(X, "index", None))

    # -- Diagnostics ----------------------------------------------------------

    def plot_residuals(self, X, y, out_path: str = "outputs/residuals.png") -> str:
        """Residuals vs fitted, plus a residual histogram."""
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        preds = self.model.predict(X)
        residuals = y - preds

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].scatter(preds, residuals, s=8, alpha=0.3, color="#3F7DE0",
                        edgecolors="none")
        axes[0].axhline(0, color="#E06B3F", linewidth=1.5)
        axes[0].set_xlabel("Predicted score")
        axes[0].set_ylabel("Residual (actual - predicted)")
        axes[0].set_title(f"{self.name}: residuals vs fitted", loc="left")

        axes[1].hist(residuals, bins=40, color="#3F7DE0", edgecolor="white")
        axes[1].axvline(0, color="#E06B3F", linewidth=1.5)
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Students")
        axes[1].set_title(f"Residual distribution (mean={np.mean(residuals):.2f})",
                          loc="left")

        for ax in axes:
            ax.grid(axis="y", color="#ededed")
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[ok] Residual plot saved to {out_path}")
        return out_path

    def feature_importance(self, top_n: int = 20,
                           out_path: str = "outputs/feature_importance.png") -> pd.DataFrame:
        """Importance ranking for tree models (empty frame for linear ones)."""
        model = self.model
        reg = model.named_steps["reg"] if isinstance(model, Pipeline) else model
        if not hasattr(reg, "feature_importances_"):
            print(f"[warn] {self.name} exposes no feature_importances_ - skipping.")
            return pd.DataFrame()

        names = self._feature_names()
        importances = reg.feature_importances_
        if names is None or len(names) != len(importances):
            names = [f"f{i}" for i in range(len(importances))]

        imp = (pd.DataFrame({"feature": names, "importance": importances})
               .sort_values("importance", ascending=False)
               .reset_index(drop=True))

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        top = imp.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(top))))
        ax.barh(top["feature"], top["importance"], color="#3F7DE0")
        ax.set_title(f"{self.name}: top {len(top)} features", loc="left")
        ax.grid(axis="x", color="#ededed")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[ok] Feature importance plot saved to {out_path}")
        return imp

    def _feature_names(self):
        if not isinstance(self.model, Pipeline):
            return None
        try:
            return list(self.model.named_steps["prep"].get_feature_names_out())
        except Exception:
            return None


class ModelComparison:
    """Train several regressors on identical splits and rank them.

    The brief asks for at least three candidate models: a regularised linear
    baseline, a bagged tree ensemble, and a boosted tree ensemble. Comparing
    across that spread shows whether the extra complexity actually buys accuracy.
    """

    def __init__(self, preprocessor_factory, random_state: int = 42):
        self.preprocessor_factory = preprocessor_factory
        self.random_state = random_state
        self.trainers = {}
        self.results = []

    def default_models(self) -> dict:
        rs = self.random_state
        return {
            "Ridge (linear baseline)": Ridge(alpha=1.0, random_state=rs),
            "Random Forest": RandomForestRegressor(
                n_estimators=300, max_depth=12, min_samples_leaf=4,
                random_state=rs, n_jobs=-1),
            "XGBoost": XGBRegressor(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=rs, n_jobs=-1),
        }

    def run(self, X_train, y_train, X_val, y_val, models: dict = None,
            cv: int = 5) -> pd.DataFrame:
        models = models or self.default_models()
        for name, estimator in models.items():
            print(f"\n=== {name} " + "=" * (52 - len(name)))
            trainer = XGBRegressorTrainer(
                preprocessor=self.preprocessor_factory(), estimator=estimator, name=name
            )
            trainer.train(X_train, y_train)
            train_metrics = trainer.evaluate(X_train, y_train, label="Train")
            val_metrics = trainer.evaluate(X_val, y_val, label="Validation")
            cv_metrics = trainer.cross_validate(X_train, y_train, cv=cv)

            self.trainers[name] = trainer
            self.results.append({
                "model": name,
                "train_RMSE": train_metrics["RMSE"],
                "val_RMSE": val_metrics["RMSE"],
                "val_MAE": val_metrics["MAE"],
                "val_R2": val_metrics["R2"],
                "cv_RMSE": cv_metrics["cv_rmse_mean"],
                "cv_RMSE_std": cv_metrics["cv_rmse_std"],
            })

        return self.leaderboard()

    def leaderboard(self) -> pd.DataFrame:
        return (pd.DataFrame(self.results)
                .sort_values("val_RMSE")
                .reset_index(drop=True))

    def best(self) -> tuple:
        """Lowest validation RMSE wins."""
        board = self.leaderboard()
        name = board.loc[0, "model"]
        return name, self.trainers[name]

    def plot_comparison(self, out_path: str = "outputs/model_comparison.png") -> str:
        board = self.leaderboard()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        y_pos = np.arange(len(board))

        axes[0].barh(y_pos - 0.2, board["train_RMSE"], height=0.4,
                     color="#B9CDF0", label="Train")
        axes[0].barh(y_pos + 0.2, board["val_RMSE"], height=0.4,
                     color="#3F7DE0", label="Validation")
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(board["model"])
        axes[0].invert_yaxis()
        axes[0].set_xlabel("RMSE (lower is better)")
        axes[0].set_title("Train vs validation RMSE", loc="left")
        axes[0].legend(frameon=False)

        axes[1].barh(y_pos, board["val_R2"], color="#3F7DE0")
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(board["model"])
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Validation R2 (higher is better)")
        axes[1].set_title("Explained variance", loc="left")

        for ax in axes:
            ax.grid(axis="x", color="#ededed")
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[ok] Model comparison plot saved to {out_path}")
        return out_path
