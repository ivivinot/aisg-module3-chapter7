"""End-to-end pipeline for predicting O-level mathematics final_test scores.

Run from the project root:

    python src/Compiler.py

Flat and procedural on purpose -- each numbered section maps to one module so the
run log can be read top to bottom.
"""

import os
import pickle
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DataCleaner import DataCleaner            # noqa: E402
from DataDownloader import DataDownloader      # noqa: E402
from DataModeler import DataModeler            # noqa: E402
from DataProfiler import DataProfiler          # noqa: E402
from DataSelector import DataSelector          # noqa: E402
from DataTransformer import DataTransformer    # noqa: E402
from XGBoostRegressor import ModelComparison   # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)


def header(step: str, title: str) -> None:
    print(f"\n{'=' * 78}\n  STEP {step} - {title}\n{'=' * 78}")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    os.chdir(PROJECT_ROOT)

    target = cfg["data"]["target"]
    cols = cfg["columns"]
    clean_cfg = cfg["cleaning"]
    out_folder = cfg["output"]["folder"]
    os.makedirs(out_folder, exist_ok=True)

    # 1 -- LOAD ---------------------------------------------------------------
    header("1", "LOAD")
    downloader = DataDownloader(
        filepath=cfg["data"]["filepath"], data_folder=cfg["data"]["data_folder"]
    )
    df = downloader.data_load()

    # 2 -- PROFILE ------------------------------------------------------------
    header("2", "PROFILE (raw)")
    profiler = DataProfiler(df)
    print(profiler.profile().to_string())
    print("\nDuplicates:", profiler.duplicate_summary(subset=clean_cfg["dedupe_key"]))
    print("\nNumeric summary:")
    print(profiler.numeric_summary().to_string())
    print("\nOutlier fences:")
    print(profiler.outlier_summary().to_string())
    profiler.categorical_summary()

    # 3 -- CLEAN --------------------------------------------------------------
    header("3", "CLEAN")
    cleaner = (
        DataCleaner(df)
        .clean_duplicate(subset=clean_cfg["dedupe_key"])
        .drop_missing_target(target)
        .standardise_categories(cols["categorical"])
        .fill_categorical("CCA", clean_cfg["cca_fill_value"])
        .normalise_yes_no(cols["yes_no"])
        .clean_negative_values(cols["non_negative"])
        .fix_age("age", valid=tuple(clean_cfg["valid_ages"]))
        .simple_impute(subset=clean_cfg["impute_median"], method="median")
    )
    df = cleaner.get()
    print(f"\n[ok] Clean frame: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 4 -- TRANSFORM ----------------------------------------------------------
    header("4", "TRANSFORM / FEATURE ENGINEERING")
    transformer = (
        DataTransformer(df)
        .add_sleep_duration()
        .add_class_composition()
        .add_study_intensity()
    )
    df = transformer.get()

    # Drop the raw columns now represented by engineered features.
    df = DataCleaner(df).drop_columns(
        cols["drop_identifiers"] + cols["drop_after_engineering"]
    ).get()
    print(f"[ok] Feature frame: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"     columns: {list(df.columns)}")

    # 5 -- FEATURE SELECTION --------------------------------------------------
    header("5", "FEATURE SELECTION (VIF)")
    selector = DataSelector(vif_threshold=cfg["selection"]["vif_threshold"])
    print("VIF before selection:")
    print(selector.vif_table(df, exclude=[target]).to_string())

    if cfg["selection"]["run_vif"]:
        df = selector.drop_high_vif(df, exclude=[target])
    else:
        print("[ok] VIF pruning disabled in config.")
    print(f"[ok] Modelling frame: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 6 -- SPLIT + PREPROCESS -------------------------------------------------
    header("6", "SPLIT + PREPROCESSING")
    modeler = DataModeler().infer_columns(df, target)
    split = cfg["split"]
    X_train, X_val, X_test, y_train, y_val, y_test = modeler.split_data(
        df, target,
        val_size=split["val_size"], test_size=split["test_size"],
        random_state=split["random_state"], stratify_bins=split["stratify_bins"],
    )

    # 7 -- TRAIN + COMPARE ----------------------------------------------------
    header("7", "TRAIN AND COMPARE MODELS")
    comparison = ModelComparison(
        preprocessor_factory=modeler.build_preprocessor,
        random_state=split["random_state"],
    )
    leaderboard = comparison.run(
        X_train, y_train, X_val, y_val, cv=cfg["training"]["cv_folds"]
    )

    # 8 -- EVALUATE -----------------------------------------------------------
    header("8", "EVALUATE")
    print("Validation leaderboard (sorted by RMSE):")
    print(leaderboard.round(4).to_string(index=False))
    leaderboard.round(6).to_csv(os.path.join(out_folder, "model_comparison.csv"),
                                index=False)
    comparison.plot_comparison(os.path.join(out_folder, "model_comparison.png"))

    best_name, best_trainer = comparison.best()
    print(f"\n[ok] Best model on validation RMSE: {best_name}")

    print("\nHeld-out test performance (touched once, after model selection):")
    test_metrics = best_trainer.evaluate(X_test, y_test, label="Test")

    best_trainer.plot_residuals(X_test, y_test,
                                os.path.join(out_folder, "residuals.png"))
    importance = best_trainer.feature_importance(
        out_path=os.path.join(out_folder, "feature_importance.png")
    )
    if not importance.empty:
        print("\nTop 15 features:")
        print(importance.head(15).to_string(index=False))
        importance.to_csv(os.path.join(out_folder, "feature_importance.csv"),
                          index=False)

    # 9 -- SAVE ---------------------------------------------------------------
    header("9", "SAVE")
    if cfg["training"]["save_model"]:
        model_path = cfg["training"]["model_path"]
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(best_trainer.model, f)
        print(f"[ok] Saved fitted pipeline ({best_name}) to {model_path}")

    predictions = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": best_trainer.predict(X_test).to_numpy(),
    })
    predictions["error"] = predictions["actual"] - predictions["predicted"]
    pred_path = os.path.join(out_folder, "test_predictions.csv")
    predictions.to_csv(pred_path, index=False)
    print(f"[ok] Saved test predictions to {pred_path}")

    print(f"\n{'=' * 78}")
    print(f"  PIPELINE COMPLETE - best: {best_name} | "
          f"test RMSE {test_metrics['RMSE']:.3f} | "
          f"MAE {test_metrics['MAE']:.3f} | R2 {test_metrics['R2']:.3f}")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
