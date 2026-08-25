import pandas as pd
import numpy as np
import os, glob, warnings, gc
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.interpolate import interp1d, CubicSpline
from sklearn.model_selection import KFold
TRAIN_DIR = "/kaggle/input/competitions/rogii-wellbore-geology-prediction/train"
TEST_DIR  = "/kaggle/input/competitions/rogii-wellbore-geology-prediction/test"
OUTPUT    = "/kaggle/working/submission.csv"
SEED      = 42
N_FOLDS   = 5
np.random.seed(SEED)
FEAT_COLS = [
    "MD", "MD_norm", "MD_diff", "MD_pct",
    "X", "Y", "Z", "Z_abs", "Z_diff",
    "X_diff", "Y_diff", "XY_dist",
    "GR_filled", "GR_znorm",
    "GR_rolling5", "GR_rolling10", "GR_rolling20",
    "GR_diff", "GR_diff2",
    "GR_std5", "GR_std20",
    "GR_lag1", "GR_lag3", "GR_lag5",
    "GR_lead1","GR_lead3","GR_lead5",
    "TVT_input_ffill", "TVT_input_bfill", "TVT_input_interp",
    "TVT_input_cubic", "TVT_input_gradient",
    "dist_to_known", "dist_to_known_norm",
    "n_known_in_window",
    "tw_GR_at_TVT", "tw_TVT_min", "tw_TVT_max", "tw_TVT_range",
    "tw_TVT_pct",
    "well_gr_mean", "well_gr_std", "well_n_known", "well_known_ratio",
    "well_tvt_known_mean", "well_tvt_known_std",
]
def get_well_id(fp):
    return os.path.basename(fp).replace("__horizontal_well.csv", "")
def load_well(hw_path, tw_dir):
    well_id = get_well_id(hw_path)
    want = ["MD","X","Y","Z","GR","TVT_input","TVT"]
    hw = pd.read_csv(hw_path, usecols=lambda c: c in want)
    for c in want:
        if c not in hw.columns:
            hw[c] = np.nan
    tw_path = os.path.join(tw_dir, f"{well_id}__typewell.csv")
    tw = pd.read_csv(tw_path) if os.path.exists(tw_path) else None
    return well_id, hw, tw
def safe_spline(x_k, y_k, x_all):
    if len(x_k) < 2:
        return np.full(len(x_all), y_k[0] if len(y_k)==1 else 0.0)
    if len(x_k) < 4:
        f = interp1d(x_k, y_k, bounds_error=False, fill_value="extrapolate")
        return f(x_all)
    try:
        return CubicSpline(x_k, y_k, extrapolate=True)(x_all)
    except Exception:
        f = interp1d(x_k, y_k, bounds_error=False, fill_value="extrapolate")
        return f(x_all)
def engineer_features(hw, tw):
    df = hw.copy().reset_index(drop=True)
    n  = len(df)
    ai = np.arange(n, dtype=np.float32)
    md = df["MD"].values.astype(np.float64)
    md_min, md_max = md.min(), md.max()
    df["MD_norm"] = (md - md_min) / (md_max - md_min + 1e-9)
    df["MD_diff"] = np.concatenate([[0], np.diff(md)])
    df["MD_pct"]  = ai / (n - 1 + 1e-9)
    df["Z_abs"]  = df["Z"].abs()
    df["Z_diff"] = df["Z"].diff().fillna(0)
    df["X_diff"] = df["X"].diff().fillna(0)
    df["Y_diff"] = df["Y"].diff().fillna(0)
    df["XY_dist"]= np.sqrt(df["X_diff"]**2 + df["Y_diff"]**2)
    gr   = df["GR"].interpolate("linear").bfill().ffill().fillna(0).values
    gr_s = pd.Series(gr)
    gr_mean = gr.mean(); gr_std = gr.std() + 1e-9
    df["GR_filled"]    = gr
    df["GR_znorm"]     = (gr - gr_mean) / gr_std
    df["GR_rolling5"]  = gr_s.rolling(5,  min_periods=1).mean().values
    df["GR_rolling10"] = gr_s.rolling(10, min_periods=1).mean().values
    df["GR_rolling20"] = gr_s.rolling(20, min_periods=1).mean().values
    df["GR_diff"]      = gr_s.diff().fillna(0).values
    df["GR_diff2"]     = gr_s.diff().diff().fillna(0).values
    df["GR_std5"]      = gr_s.rolling(5,  min_periods=1).std().fillna(0).values
    df["GR_std20"]     = gr_s.rolling(20, min_periods=1).std().fillna(0).values
    for lag in [1, 3, 5]:
        df[f"GR_lag{lag}"]  = gr_s.shift(lag).bfill().values
        df[f"GR_lead{lag}"] = gr_s.shift(-lag).ffill().values
    ti = df["TVT_input"]
    df["TVT_input_ffill"]  = ti.ffill()
    df["TVT_input_bfill"]  = ti.bfill()
    df["TVT_input_interp"] = ti.interpolate("linear").bfill().ffill()
    known_idx = np.where(ti.notna())[0]
    if len(known_idx) >= 2:
        cubic = safe_spline(known_idx, ti.iloc[known_idx].values, ai)
    else:
        cubic = df["TVT_input_interp"].values.copy()
    df["TVT_input_cubic"]    = cubic
    df["TVT_input_gradient"] = np.gradient(cubic)
    if len(known_idx) > 0:
        dist = np.abs(ai[:, None] - known_idx[None, :]).min(axis=1)
        df["dist_to_known"]      = dist.astype(np.float32)
        df["dist_to_known_norm"] = dist / (n + 1e-9)
    else:
        df["dist_to_known"]      = float(n)
        df["dist_to_known_norm"] = 1.0
    window    = 20
    known_arr = np.zeros(n, dtype=np.float32)
    if len(known_idx) > 0:
        known_arr[known_idx] = 1.0
    df["n_known_in_window"] = np.convolve(known_arr, np.ones(2*window+1), mode="same")
    df["well_gr_mean"]        = gr_mean
    df["well_gr_std"]         = gr_std
    df["well_n_known"]        = len(known_idx)
    df["well_known_ratio"]    = len(known_idx) / n
    ktv = ti.dropna()
    df["well_tvt_known_mean"] = ktv.mean() if len(ktv) > 0 else 0.0
    df["well_tvt_known_std"]  = ktv.std()  if len(ktv) > 1 else 0.0
    if tw is not None and len(tw) > 0:
        tw_c = tw.dropna(subset=["TVT","GR"])
        if len(tw_c) >= 2:
            tw_f   = interp1d(tw_c["TVT"].values, tw_c["GR"].values,
                              bounds_error=False, fill_value="extrapolate")
            anchor = pd.Series(cubic).fillna(pd.Series(df["Z_abs"])).values
            df["tw_GR_at_TVT"] = tw_f(anchor)
        else:
            df["tw_GR_at_TVT"] = 0.0
        tw_min = tw["TVT"].min(); tw_max = tw["TVT"].max()
        tw_rng = tw_max - tw_min + 1e-9
        df["tw_TVT_min"]   = tw_min
        df["tw_TVT_max"]   = tw_max
        df["tw_TVT_range"] = tw_rng
        df["tw_TVT_pct"]   = (pd.Series(cubic).values - tw_min) / tw_rng
    else:
        for c in ["tw_GR_at_TVT","tw_TVT_min","tw_TVT_max","tw_TVT_range","tw_TVT_pct"]:
            df[c] = 0.0
    return df
hw_files = sorted(glob.glob(f"{TRAIN_DIR}/*__horizontal_well.csv"))
print(f"Loading {len(hw_files)} train wells...")
X_parts, y_parts = [], []
for i, hw_path in enumerate(hw_files):
    well_id, hw, tw = load_well(hw_path, TRAIN_DIR)
    df = engineer_features(hw, tw)
    mask = df["TVT_input"].isna() & df["TVT"].notna()
    df_null = df[mask]
    if len(df_null) == 0:
        continue
    X_parts.append(df_null[FEAT_COLS].fillna(0).values.astype(np.float32))
    y_parts.append(df_null["TVT"].values.astype(np.float32))
    if (i+1) % 100 == 0 or (i+1) == len(hw_files):
        print(f"  {i+1}/{len(hw_files)} wells | samples: {sum(len(x) for x in X_parts):,}")
X_train = np.vstack(X_parts)
y_train = np.concatenate(y_parts)
del X_parts, y_parts; gc.collect()
print(f"Train: {len(X_train):,} x {len(FEAT_COLS)} | y mean={y_train.mean():.2f} std={y_train.std():.2f}")
print("Training LightGBM 5-fold...")
params = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.03, "num_leaves": 255,
    "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8, "reg_alpha": 0.05, "reg_lambda": 0.1,
    "n_estimators": 5000, "random_state": SEED, "n_jobs": -1, "verbose": -1,
}
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
X_df = pd.DataFrame(X_train, columns=FEAT_COLS)
y_s  = pd.Series(y_train)
models, fold_scores = [], []
for fold, (tr, va) in enumerate(kf.split(X_df)):
    m = lgb.LGBMRegressor(**params)
    m.fit(X_df.iloc[tr], y_s.iloc[tr],
          eval_set=[(X_df.iloc[va], y_s.iloc[va])],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)])
    p = m.predict(X_df.iloc[va])
    s = np.sqrt(np.mean((y_s.iloc[va].values - p)**2))
    fold_scores.append(s); models.append(m)
    print(f"  Fold {fold+1}/{N_FOLDS} | RMSE: {s:.5f} | iter: {m.best_iteration_}")
cv = np.mean(fold_scores)
print(f"CV RMSE: {cv:.5f} +/- {np.std(fold_scores):.5f}")
print("Predicting test wells...")
all_preds = []
for hw_path in sorted(glob.glob(f"{TEST_DIR}/*__horizontal_well.csv")):
    well_id, hw, tw = load_well(hw_path, TEST_DIR)
    df = engineer_features(hw, tw)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    null_mask  = df["TVT_input"].isna()
    known_mask = ~null_mask
    if null_mask.sum() > 0:
        prd = np.mean([m.predict(df.loc[null_mask, FEAT_COLS].fillna(0)) for m in models], axis=0)
        df.loc[null_mask, "tvt_pred"] = prd
    df.loc[known_mask, "tvt_pred"] = df.loc[known_mask, "TVT_input"]
    nd = df[null_mask].copy()
    nd["id"]  = well_id + "_" + nd.index.astype(str)
    nd["tvt"] = nd["tvt_pred"]
    all_preds.append(nd[["id","tvt"]])
    print(f"  {well_id}: {null_mask.sum()} null | {known_mask.sum()} known")
pred_df = pd.concat(all_preds, ignore_index=True)
sample  = pd.read_csv("/kaggle/input/competitions/rogii-wellbore-geology-prediction/sample_submission.csv")
sub     = sample[["id"]].merge(pred_df, on="id", how="left")
sub["tvt"] = sub["tvt"].fillna(0)
sub.to_csv(OUTPUT, index=False)
print(f"Matched: {(sub['tvt']!=0).sum()}/{len(sub)}")
print(f"CV RMSE: {cv:.5f}")
print("DONE!")
