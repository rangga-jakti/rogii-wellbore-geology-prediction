"""
ROGII Wellbore Geology Prediction - v3
Fix: ID = {well_id}_{row_index} bukan MD
"""

import pandas as pd
import numpy as np
import os, glob, warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold

# ============================================================
# CONFIG
# ============================================================
TRAIN_DIR       = "data/raw/train"
TEST_DIR        = "data/raw/test"
OUTPUT          = "submission.csv"
SEED            = 42
MAX_WELLS       = 300
SAMPLE_PER_WELL = 50
np.random.seed(SEED)

FEAT_COLS = [
    "MD", "MD_norm", "MD_diff", "MD_cumsum",
    "X", "Y", "Z", "Z_abs", "Z_diff",
    "X_diff", "Y_diff", "XY_dist",
    "GR_filled", "GR_rolling5", "GR_rolling20", "GR_diff", "GR_std5",
    "TVT_input_ffill", "TVT_input_bfill", "TVT_input_interp",
    "dist_to_known",
    "tw_GR_at_TVT", "tw_TVT_min", "tw_TVT_max", "tw_TVT_range",
]

# ============================================================
# HELPERS
# ============================================================
def get_well_id(filepath):
    return os.path.basename(filepath).replace("__horizontal_well.csv", "")

def load_well(hw_path, tw_dir):
    well_id = get_well_id(hw_path)
    usecols = ["MD","X","Y","Z","GR","TVT_input","TVT"]
    hw = pd.read_csv(hw_path, usecols=lambda c: c in usecols)
    for c in usecols:
        if c not in hw.columns:
            hw[c] = np.nan
    tw_path = os.path.join(tw_dir, f"{well_id}__typewell.csv")
    tw = pd.read_csv(tw_path) if os.path.exists(tw_path) else None
    return well_id, hw, tw

def engineer_features(hw, tw):
    df = hw.copy()
    md_min, md_max = df["MD"].min(), df["MD"].max()
    df["MD_norm"]   = (df["MD"] - md_min) / (md_max - md_min + 1e-9)
    df["MD_diff"]   = df["MD"].diff().fillna(1)
    df["MD_cumsum"] = df["MD"] - df["MD"].iloc[0]
    df["Z_abs"]     = df["Z"].abs()
    df["Z_diff"]    = df["Z"].diff().fillna(0)
    df["X_diff"]    = df["X"].diff().fillna(0)
    df["Y_diff"]    = df["Y"].diff().fillna(0)
    df["XY_dist"]   = np.sqrt(df["X_diff"]**2 + df["Y_diff"]**2)

    gr = df["GR"].interpolate(method="linear").bfill().ffill()
    df["GR_filled"]    = gr
    df["GR_rolling5"]  = gr.rolling(5,  min_periods=1).mean()
    df["GR_rolling20"] = gr.rolling(20, min_periods=1).mean()
    df["GR_diff"]      = gr.diff().fillna(0)
    df["GR_std5"]      = gr.rolling(5,  min_periods=1).std().fillna(0)

    df["TVT_input_ffill"]  = df["TVT_input"].ffill()
    df["TVT_input_bfill"]  = df["TVT_input"].bfill()
    df["TVT_input_interp"] = df["TVT_input"].interpolate(method="linear")

    known_idx = np.where(df["TVT_input"].notna())[0]
    if len(known_idx) > 0:
        all_idx = np.arange(len(df))
        dist = np.min(np.abs(all_idx[:, None] - known_idx[None, :]), axis=1)
        df["dist_to_known"] = dist.astype(np.float32)
    else:
        df["dist_to_known"] = 9999.0

    if tw is not None and len(tw) > 0:
        tw_clean = tw.dropna(subset=["TVT","GR"])
        if len(tw_clean) > 1:
            tw_interp = interp1d(tw_clean["TVT"].values, tw_clean["GR"].values,
                                  bounds_error=False, fill_value="extrapolate")
            anchor = df["TVT_input_interp"].fillna(df["Z_abs"])
            df["tw_GR_at_TVT"] = tw_interp(anchor.values)
        else:
            df["tw_GR_at_TVT"] = 0.0
        df["tw_TVT_min"]   = tw["TVT"].min()
        df["tw_TVT_max"]   = tw["TVT"].max()
        df["tw_TVT_range"] = tw["TVT"].max() - tw["TVT"].min()
    else:
        df["tw_GR_at_TVT"] = 0.0
        df["tw_TVT_min"]   = 0.0
        df["tw_TVT_max"]   = 0.0
        df["tw_TVT_range"] = 0.0

    return df

# ============================================================
# TRAIN
# ============================================================
print("="*60)
print("ROGII - v3 (ID Fix + Memory Optimized)")
print("="*60)

print(f"\n[1/5] Loading {MAX_WELLS} train wells ({SAMPLE_PER_WELL} rows/well)...")
hw_files = sorted(glob.glob(f"{TRAIN_DIR}/*__horizontal_well.csv"))
rng = np.random.default_rng(SEED)
hw_files = list(rng.choice(hw_files, min(MAX_WELLS, len(hw_files)), replace=False))

X_parts, y_parts = [], []
for i, hw_path in enumerate(hw_files):
    well_id, hw, tw = load_well(hw_path, TRAIN_DIR)
    df = engineer_features(hw, tw)
    null_mask = df["TVT_input"].isna()
    df_null = df[null_mask].copy()
    if len(df_null) == 0:
        continue
    if len(df_null) > SAMPLE_PER_WELL:
        df_null = df_null.sample(SAMPLE_PER_WELL, random_state=SEED)
    for col in FEAT_COLS:
        if col not in df_null.columns:
            df_null[col] = 0.0
    X_parts.append(df_null[FEAT_COLS].fillna(0).values.astype(np.float32))
    y_parts.append(df_null["TVT"].values.astype(np.float32))
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(hw_files)} wells...")

X_train = np.vstack(X_parts)
y_train = np.concatenate(y_parts)
del X_parts, y_parts
print(f"Train samples: {len(X_train)} | Features: {len(FEAT_COLS)}")

# ============================================================
# LIGHTGBM
# ============================================================
print("\n[2/5] Training LightGBM (5-fold CV)...")
lgb_params = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 127,
    "min_child_samples": 10, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.1,
    "reg_lambda": 0.1, "n_estimators": 2000,
    "random_state": SEED, "n_jobs": -1, "verbose": -1,
}

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
X_df = pd.DataFrame(X_train, columns=FEAT_COLS)
y_s  = pd.Series(y_train)
models, fold_scores = [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_df)):
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_df.iloc[tr_idx], y_s.iloc[tr_idx],
        eval_set=[(X_df.iloc[val_idx], y_s.iloc[val_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
    )
    preds = model.predict(X_df.iloc[val_idx])
    score = np.sqrt(np.mean((y_s.iloc[val_idx].values - preds)**2))
    fold_scores.append(score)
    models.append(model)
    print(f"  Fold {fold+1}/5 | RMSE: {score:.5f}")

cv_score = np.mean(fold_scores)
print(f"\nCV RMSE: {cv_score:.5f} +/- {np.std(fold_scores):.5f}")

# ============================================================
# TEST PREDICTIONS
# ============================================================
print("\n[3/5] Predicting test wells...")
test_hw_files = sorted(glob.glob(f"{TEST_DIR}/*__horizontal_well.csv"))

all_preds = []
for hw_path in test_hw_files:
    well_id, hw, tw = load_well(hw_path, TEST_DIR)
    df = engineer_features(hw, tw)

    for col in FEAT_COLS:
        if col not in df.columns:
            df[col] = 0.0

    null_mask = df["TVT_input"].isna()

    # Predict null rows
    if null_mask.sum() > 0:
        X_test = df.loc[null_mask, FEAT_COLS].fillna(0)
        preds = np.mean([m.predict(X_test) for m in models], axis=0)
        df.loc[null_mask, "tvt_pred"] = preds

    # Known rows pakai TVT_input langsung
    df.loc[~null_mask, "tvt_pred"] = df.loc[~null_mask, "TVT_input"]

    # Smooth
    df["tvt_pred"] = df["tvt_pred"].interpolate(method="linear").bfill().ffill()

    # ✅ ID = {well_id}_{row_index} — hanya null rows!
    null_df = df[null_mask].copy()
    null_df["id"] = well_id + "_" + null_df.index.astype(str)
    null_df["tvt"] = df.loc[null_mask, "tvt_pred"].values

    all_preds.append(null_df[["id", "tvt"]])
    print(f"  {well_id}: {null_mask.sum()} predicted rows")

# ============================================================
# SUBMISSION
# ============================================================
print("\n[4/5] Building submission...")
pred_df = pd.concat(all_preds, ignore_index=True)

sample_sub = pd.read_csv("data/raw/sample_submission.csv")
submission = sample_sub[["id"]].merge(pred_df, on="id", how="left")
submission["tvt"] = submission["tvt"].fillna(0)

matched = submission["tvt"].ne(0).sum()
print(f"Matched IDs: {matched}/{len(submission)}")
print(submission["tvt"].describe())

submission.to_csv(OUTPUT, index=False)
print(f"\n[5/5] Saved: {OUTPUT}")
print(f"CV RMSE: {cv_score:.5f}")
print("="*60)
print("DONE! Submit ke Kaggle!")
print("="*60)
