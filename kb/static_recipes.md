# Static Recipes (task-agnostic best practices)

This file is injected into the SuperAgent's planning prompts. It contains
reusable, general knowledge for classification tasks. Do NOT hard-code any
task-specific solution here.

## General workflow
1. Always fit preprocessing on train only; transform val/test with the fitted objects.
2. Build a quick baseline FIRST (simple model, raw features) to establish a floor,
   then iterate. Measure everything against the validation split.
3. Prefer robust defaults over clever tricks when validation data is small.
4. Check for duplicate rows between splits (leakage) and for identical
   feature rows with conflicting labels.
5. Near the end, ensembling the 2-3 most diverse strong models (soft voting /
   rank averaging) often buys a small but real gain on top of the best single model.

## Tabular data (numeric + categorical features)
- Gradient boosted trees (LightGBM / XGBoost / CatBoost) are the strongest
  default for mixed-type tabular data at small/medium scale (<100k rows).
- Encoding: LightGBM/XGBoost work well with ordinal-encoded categoricals
  (+ `enable_categorical=True` for LightGBM pandas input). CatBoost handles
  categoricals natively. One-hot only for low-cardinality columns.
- Treat ordinal-looking categoricals as ordered when the semantics suggest it
  (e.g. income brackets, frequency buckets: never < less1 < 1~3 < 4~8 < gt8),
  or add both ordinal + one-hot views when unsure.
- Missing values: LightGBM/XGBoost handle NaN natively; for linear models
  impute (median for numeric, constant for categorical) + add missing indicators.
- Feature engineering that usually helps: per-group frequency counts,
  interactions of the 2-3 most important features, monotonic mappings for
  ordinal buckets, target/mean encoding ONLY with careful inner CV to avoid leakage.
- Class imbalance: check class ratios; use class_weight/scale_pos_weight if skewed.
- Calibrate decision: if metric is accuracy on a moderately imbalanced problem,
  try tuning the probability threshold on validation data (0.5 is not always optimal).

## Text data
- Bag-of-words baselines are very strong at small scale: TF-IDF (word 1-2 grams +
  char_wb 3-5 grams) + LogisticRegression / LinearSVC. Tune C and min_df on val.
- For pairwise text comparison tasks, build symmetric features: differences and
  products of the two sides' feature vectors, and consider "swap augmentation"
  (duplicate training pairs with A/B swapped and labels flipped) so the model
  is order-symmetric.
- HashingVectorizer if memory is a concern; TruncatedSVD (100-300 dims) when
  feeding dense models (LightGBM/MLP).
- Simple stylometric features (length, word counts, digits, punctuation,
  uppercase ratio, question marks) complement TF-IDF.
- Keep preprocessing deterministic; save the fitted vectorizer inside the model
  pipeline (joblib) so FINAL mode reproduces it exactly.

## Image data
- Freeze a pretrained backbone (torch, e.g. ResNet) -> embed -> train a small
  head (logistic / LightGBM) when the training set is small.
- Fine-tune only if the dataset is large enough and time budget allows.

## Runtime discipline
- A single solution.py run must stay well under the 600s sandbox timeout.
  Reserve time: fast models first; expensive CV (5-fold) only when cheap enough.
- Fix random_state everywhere; report deterministic results.

## Common failure modes (debug these first)
- Label column accidentally included in features; index column leaking.
- Object-dtype arrays from np.load(allow_pickle=True) needing astype(str)/float.
- Windows console encoding (use utf-8 when printing).
- Saving only the model but not the preprocessing -> FINAL mode mismatch.
- Memory blowups from dense one-hot on high-cardinality columns.
