# Predicting O-Level Mathematics Scores

## Project Overview

A school wants to identify weaker students **before** the O-level mathematics
examination so that remedial support can be directed where it matters. This project
takes 15,900 student records and builds a supervised **regression** pipeline that
estimates each student's `final_test` score (range 32–100) from demographic, class,
and study-habit features.

Three model families were evaluated on identical splits — a regularised linear
baseline (Ridge), a bagged tree ensemble (Random Forest), and a boosted tree ensemble
(XGBoost). **Random Forest** was selected on validation RMSE and achieves
**RMSE 5.49 / MAE 3.75 / R² 0.845** on the untouched held-out test set, against a
mean-only baseline of RMSE 13.98.

---

## File Structure

```
Module 3 Chapter 7 - Copy/
├── data/
│   └── regression_bonus_practice_data.csv   Raw dataset (15,900 × 18)
├── src/
│   ├── config.yaml              All tunable settings — paths, thresholds, split sizes
│   ├── DataDownloader.py        Load CSV / JSON / Excel / SQLite, or fetch from a URL
│   ├── DataProfiler.py          Read-only profiling: dtypes, missingness, IQR outliers
│   ├── DataCleaner.py           Dedupe, category standardisation, age repair, imputation
│   ├── DataTransformer.py       Feature engineering: sleep_duration, class_size, ratios
│   ├── DataSelector.py          VIF-based multicollinearity pruning
│   ├── DataModeler.py           Stratified train/val/test split + shared preprocessor
│   ├── XGBoostRegressor.py      Trainer, diagnostics, and the 3-model comparison harness
│   └── Compiler.py              Orchestrates the full run, step 1 → 9
├── outputs/                     Generated: metrics, plots, predictions, pickled model
├── eda.ipynb                    Original exploratory notebook (hand-written)
├── eda_pipeline.ipynb           Structured EDA that mirrors the pipeline's decisions
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
pip install -r requirements.txt
python src/Compiler.py
```

The run prints nine labelled sections and writes all artefacts to `outputs/`.

---

## Key Findings from EDA

- **Shape:** 15,900 rows × 18 columns; one row per student *record*.
- **900 duplicate students.** No whole-row duplicates exist, but 900 rows share a
  `student_id` with another row and are byte-identical apart from the occasional
  `bag_color`. Left in, the same student appears in both train and test. Removing them
  is the single most important correctness step in the pipeline.
- **Missing data is confined to three columns:** `CCA` (24%), `attendance_rate` (4.9%),
  `final_test` (3.1%).
- **`age` is corrupted:** 5 negative values (−4, −5) and 446 single-digit values (5, 6)
  that are clearly 15/16 with a dropped leading digit. The real cohort is 15–16 only.
- **Target is near-symmetric:** mean 67.2, median 68, std 14.0, skew −0.06. No target
  transform needed. The mean-only baseline is RMSE **13.98** / MAE **11.65**.
- **Top predictors:** `class_size` (r = −0.50), `number_of_siblings` (r = −0.36),
  `attendance_rate` (r = +0.34), `sleep_duration` (r = +0.33). Among categoricals,
  students with no recorded CCA score **+9.6** marks above the mean, visual learners
  **+4.4**, and students with tuition **+3.3**.
- **A counter-intuitive result:** `hours_per_week` correlates **negatively** with the
  score (r = −0.15). The plausible reading is reverse causation — struggling students
  study more hours *because* they are struggling. Reported as found, not "corrected".
- **`bag_color` is noise**, as expected: its six group means span 1.0 mark around an
  overall mean of 67.2. It is deliberately retained as a negative control — if it ever
  ranks highly in feature importance, something has gone wrong.

---

## Feature Engineering Decisions

| Feature | Raw State | Action Taken | Reason |
|---|---|---|---|
| `student_id` | 900 duplicated values | Deduplicate, then drop the column | Prevents the same student leaking across the train/test boundary; the ID itself has no signal |
| `index` | Row counter | Drop | Administrative, no predictive content |
| `final_test` | 495 missing (3.1%) | Drop those rows | Imputing a label fabricates the signal being learned; 3.1% is an acceptable loss |
| `CCA` | `Clubs` / `CLUBS` casing split; 24% missing | Upper-case, then fill missing with `NONE` | Merges spurious levels; "no CCA recorded" is information, so it becomes its own category rather than being mode-imputed |
| `tuition` | `Yes`/`Y` and `No`/`N` mixed | Collapse to first letter → `Y`/`N` | Four levels were really two |
| `direct_admission` | `Yes`/`No` | Normalise to `Y`/`N` | Consistency with `tuition` |
| `age` | −4, −5, 5, 6 among valid 15/16 | `abs()`, restore the dropped digit (`+10`), out-of-range → mode | Repairs 420 defective values in the cleaned frame; only one (the −4, which maps to 14) falls outside the cohort and needs mode imputation. Recovers the rows instead of discarding them |
| `attendance_rate` | 778 missing, long low tail | Median impute (95.0) | Median resists the low-attendance tail that drags the mean down |
| `sleep_time`, `wake_time` | Clock strings crossing midnight | `(wake − sleep) % 24` → `sleep_duration`; drop originals | Modulo handles 22:00 → 06:00 correctly; duration is the meaningful quantity |
| `n_male`, `n_female` | Two raw headcounts | Derive `class_size` and `gender_ratio`; drop originals | The sum is the real signal (r = −0.50); keeping all three is redundant |
| `hours_per_week` × `attendance_rate` | — | Derive `study_intensity` | Hours only count if the student attends |
| `study_intensity` | VIF ≈ 159 | Dropped by the VIF sweep | Collinear by construction with its parents; the interpretable parents are kept instead |
| All categoricals | Strings | One-hot (drop-first) inside the model Pipeline | Fitted per-fold, so no encoding leakage |
| All numerics | Varied scales | StandardScaler inside the model Pipeline | Required by Ridge; inert for the tree models |

### A note on the VIF implementation

`statsmodels.variance_inflation_factor` regresses each column on the others **without
an intercept**. Used naively, any column whose mean is large relative to its spread
(`attendance_rate` ≈ 93 ± 8, `age` ≈ 15.5 ± 0.5) scores a VIF in the hundreds that
reflects its offset, not genuine collinearity. That version of the sweep dropped five
of eight numeric features and cut test R² from **0.845 to 0.661**.
`DataSelector._vif_series` therefore inserts an explicit constant column before
computing VIF and excludes it from the results. With the correction, only the one
genuinely redundant feature (`study_intensity`) is pruned.

---

## Model Selection

All three models were trained on the same 10,175-row training split with an identical
preprocessing pipeline, tuned on a 1,454-row validation split, and scored once on a
2,908-row held-out test set. The split is **stratified on target deciles** so that the
weak-student tail the school cares about is represented in every split.

| Model | Train RMSE | Val RMSE | Val MAE | Val R² | 5-fold CV RMSE |
|---|---|---|---|---|---|
| **Random Forest** | 4.28 | **5.18** | **3.67** | **0.863** | 5.51 ± 0.18 |
| XGBoost | 4.81 | 5.35 | 3.99 | 0.854 | 5.63 ± 0.11 |
| Ridge (linear baseline) | 9.13 | 8.85 | 7.12 | 0.601 | 9.16 ± 0.12 |
| *Mean-only baseline* | *13.98* | — | *11.65* | *0.000* | — |

**Held-out test performance (Random Forest): RMSE 5.49 | MAE 3.75 | R² 0.845.**

**Why Random Forest.** It wins on validation RMSE, MAE, and R², and the margin over
XGBoost (5.18 vs 5.35) is roughly the size of the cross-validation standard deviation,
so the two boosted/bagged approaches are close to equivalent in practice. Random Forest
is preferred as the default because it reached that accuracy with no tuning beyond
sensible depth limits, and its train→validation gap (4.28 → 5.18) is modest.

**Why the linear baseline matters.** Ridge lands at R² 0.601 versus 0.863 for the tree
models. That ~0.26 gap is the clearest evidence that the relationships here are
genuinely non-linear and interactive — precisely what the negative `hours_per_week`
correlation hinted at. Without the baseline there would be no way to tell whether the
ensembles were earning their complexity.

**Practical reading.** A typical prediction lands within **~3.8 marks** (MAE) of the
true score. For flagging students at risk of falling below a pass threshold, that is
comfortably sharp enough to triage, though borderline cases should be reviewed by a
teacher rather than acted on automatically.

---

## Deployment Considerations

- **Interpretability.** Teachers will be asked to act on these predictions, so the
  ranking must be explainable. The pipeline already exports global feature importance;
  add per-student SHAP values before any classroom-facing rollout so a flag can be
  explained as "small class, high hours, low attendance" rather than as a bare number.
- **Retraining trigger.** Retrain each academic year once new `final_test` results land,
  and sooner if intake policy changes (`direct_admission` mix) or class sizes shift —
  `class_size` is the dominant feature, so structural changes there invalidate the model
  fastest.
- **Data-drift monitoring.** Track the distribution of `class_size`, `attendance_rate`,
  and `number_of_siblings` against the training baseline, plus the rate of missing `CCA`.
  Alert on prediction-distribution shift, since ground truth only arrives annually.
- **Data-quality gate at inference.** The `age` corruption and duplicated `student_id`s
  were upstream entry errors, not one-off accidents. Run `DataProfiler` as a validation
  gate on every new batch and reject or quarantine records that fail, rather than
  silently repairing them in production.
- **Integration format.** The fitted `ColumnTransformer` + estimator is pickled whole to
  `outputs/best_model.pkl`, so scoring is `pickle.load(...).predict(raw_df)` with no
  preprocessing to reimplement. Pin the scikit-learn and XGBoost versions alongside the
  artefact — pickles are not portable across major versions.
- **Fairness review before use.** `gender` and `mode_of_transport` show no meaningful
  effect and could reasonably be dropped. More importantly, a model used to allocate
  scarce teaching support should be audited for whether it systematically under- or
  over-flags any subgroup before it influences real decisions.
