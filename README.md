# 🛢️ ROGII — Wellbore Geology Prediction

> **Kaggle Featured Competition** | Regression | Metric: RMSE

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
[![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-orange)](https://lightgbm.readthedocs.io)

---

## 🎯 Problem Statement

Drilling a horizontal well is like navigating underground without a map. The drill path runs through invisible rock layers, and even small deviations from the target zone can waste millions of dollars.

This competition challenges participants to **predict TVT (True Vertical Thickness)** — the thickness of subsurface geological layers — from horizontal wellbore drilling data. Accurate TVT predictions help automate geosteering, reduce resource waste, and improve drilling safety.

**Real-world impact:** ~10,000 horizontal wells drilled worldwide every year. Automating geological interpretation saves both money and environmental footprint.

---

## 📊 Dataset Overview

| Item | Detail |
|------|--------|
| Train wells | 773 horizontal wells |
| Test wells | 3 horizontal wells |
| Target | `tvt` (True Vertical Thickness) |
| Metric | RMSE (Root Mean Squared Error) |
| Task type | Regression |

### File Structure per Well
Each well has two files:
- `{well_id}__horizontal_well.csv` — actual drilling log data
- `{well_id}__typewell.csv` — reference formation data

### Key Features
| Feature | Description |
|---------|-------------|
| `MD` | Measured Depth (actual drilled depth) |
| `X, Y, Z` | 3D coordinates of wellbore position |
| `GR` | Gamma Ray log — key lithology indicator |
| `TVT_input` | Partially known TVT (anchor points) |

---

## 💡 Key Insights

1. **TVT_input = perfect anchor** — where `TVT_input` is known, it equals `TVT` exactly (diff = 0). Strategy: use it directly, only predict where it's null.

2. **~71% of rows need prediction** — `TVT_input` is null for most of the horizontal section, which is exactly what we predict.

3. **Typewell = geological reference map** — the typewell file maps TVT ranges to formation names (ANCC, BUDA, EGFDU, etc.), enabling GR-based correlation.

4. **Sequential pattern** — TVT changes smoothly with MD (measured depth), making interpolation-based features extremely powerful.

5. **Test wells don't have formation log columns** — features must be compatible across train and test (only MD, X, Y, Z, GR, TVT_input available in test).

---

## 🔧 Solution Architecture

```
Raw Data (train/test wells)
        ↓
Feature Engineering
  ├── MD-based: norm, diff, cumsum
  ├── Spatial: Z_abs, XY_dist, Z_diff
  ├── GR: rolling5/20, diff, std
  ├── TVT_input: ffill, bfill, linear interp
  ├── dist_to_known: distance to nearest anchor
  └── Typewell: GR correlation at TVT
        ↓
LightGBM (5-Fold CV)
  ├── All 773 train wells
  ├── ~3.7M training samples
  └── Early stopping (150 rounds)
        ↓
Prediction Strategy
  ├── TVT_input known → use directly (RMSE = 0)
  └── TVT_input null  → LightGBM prediction
        ↓
Submission (14,151 rows)
```

---

## 📁 Project Structure

```
rogii-wellbore-geology-prediction/
├── README.md
├── .gitignore
├── src/
│   ├── rogii_pipeline_v3.py     # Local pipeline (memory optimized)
│   └── rogii_kaggle_final.py    # Kaggle notebook (full data)
├── data/
│   └── .gitkeep                 # Raw data not tracked (too large)
└── outputs/
    └── .gitkeep
```

---

## 🚀 How to Run

### Local (Memory Optimized — 300 wells sample)
```bash
# Install dependencies
pip install lightgbm scikit-learn scipy pandas numpy

# Run pipeline
python src/rogii_pipeline_v3.py
```

### Kaggle Notebook (Full Data — 773 wells)
1. Open Kaggle Notebook in the competition
2. Add competition data
3. Run `src/rogii_kaggle_final.py`
4. Submit output

---

## 📈 Results

| Model | Wells | Samples | CV RMSE |
|-------|-------|---------|---------|
| LightGBM Baseline (local) | 300 | 15,000 | 3.224 |
| LightGBM Full (Kaggle) | 773 | ~3.7M | **0.393** |

---

## 🧠 Feature Importance

Top features driving predictions:
1. `TVT_input_interp` — linear interpolation of known TVT anchors
2. `TVT_input_ffill` / `TVT_input_bfill` — forward/backward fill
3. `dist_to_known` — distance to nearest known TVT point
4. `GR_rolling20` — smoothed Gamma Ray signal
5. `Z_abs` / `MD` — depth-based position

---

## 🔮 Future Improvements

- [ ] **Per-well normalization** — each well has different TVT scale
- [ ] **Sequence models** (LSTM/GRU) — TVT is sequential per depth
- [ ] **Typewell GR matching** — find most similar typewell per well
- [ ] **XGBoost + CatBoost ensemble** — blend multiple models
- [ ] **Pseudo-labeling** — use test predictions as training signal
- [ ] **Optuna hyperparameter tuning** — systematic optimization

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **LightGBM** — gradient boosting model
- **pandas / numpy** — data processing
- **scikit-learn** — cross-validation
- **scipy** — interpolation (typewell GR)

---

## 👤 Author

**Rangga** — Python Backend & ML Engineer

> Built as part of competitive ML practice and portfolio development.
> Competition: [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
