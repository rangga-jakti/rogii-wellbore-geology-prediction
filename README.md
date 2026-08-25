# 🛢️ ROGII — Wellbore Geology Prediction
**Kaggle Featured Competition | Regression | Metric: RMSE**

![Kaggle](https://img.shields.io/badge/Kaggle-Featured_Competition-20BEFF?logo=kaggle)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-orange)

---

## 🎯 Problem Statement

Drilling a horizontal well is like navigating underground without a map. The drill path runs through invisible rock layers, and even small deviations from the target zone can waste millions of dollars.

This competition challenges participants to predict **TVT (True Vertical Thickness)** — the thickness of subsurface geological layers — from horizontal wellbore drilling data. Accurate TVT predictions help automate geosteering, reduce resource waste, and improve drilling safety.

> **Real-world impact:** ~10,000 horizontal wells drilled worldwide every year. Automating geological interpretation saves both money and environmental footprint.

---

## 📊 Dataset Overview

| Item | Detail |
|---|---|
| Train wells | 773 horizontal wells |
| Test wells | 3 horizontal wells |
| Target | `tvt` (True Vertical Thickness) |
| Metric | RMSE (Root Mean Squared Error) |
| Task type | Regression |
| Training samples | ~3.78M rows |

### File Structure per Well
Each well has two files:
- `{well_id}__horizontal_well.csv` — actual drilling log data
- `{well_id}__typewell.csv` — reference formation data

### Key Features
| Feature | Description |
|---|---|
| `MD` | Measured Depth (actual drilled depth along wellbore) |
| `X, Y, Z` | 3D coordinates of wellbore position |
| `GR` | Gamma Ray log — key lithology indicator |
| `TVT_input` | Partially known TVT (anchor points from geosteering) |

---

## 💡 Key Insights

1. **TVT_input = perfect anchor** — where `TVT_input` is known, it equals `TVT` exactly (diff = 0). Strategy: use it directly, only predict where it's null.

2. **~71% of rows need prediction** — `TVT_input` is null for most of the horizontal section, which is exactly what we predict.

3. **Typewell = geological reference map** — the typewell file maps TVT ranges to formation names, enabling GR-based correlation between wells.

4. **Sequential pattern** — TVT changes smoothly with MD (measured depth), making interpolation-based features extremely powerful.

5. **Test wells don't have formation log columns** — features must be compatible across train and test (only MD, X, Y, Z, GR, TVT_input available in test).

---

## 🔧 Solution Architecture

```
Raw Data (773 train wells / 3 test wells)
        ↓
Feature Engineering (46 features)
  ├── MD-based       : norm, diff, pct position (0→1)
  ├── Spatial        : Z_abs, XY_dist, Z/X/Y diffs
  ├── GR signals     : rolling 5/10/20, diff, std, z-score normalization
  ├── GR lag/lead    : lag 1/3/5, lead 1/3/5 (sequence context)
  ├── TVT_input      : ffill, bfill, linear interp, cubic spline
  ├── TVT gradient   : rate of change from spline
  ├── dist_to_known  : distance to nearest anchor point (vectorized)
  ├── n_known_window : density of known points in ±20 row window
  ├── Typewell       : GR correlation, TVT range, position in range
  └── Well-level     : GR mean/std, known ratio, TVT known stats
        ↓
LightGBM (5-Fold KFold CV)
  ├── 3,783,989 training samples
  ├── 46 engineered features
  ├── lr=0.03, num_leaves=255, n_estimators=5000
  └── Early stopping (200 rounds)
        ↓
Prediction Strategy
  ├── TVT_input known → use directly (perfect signal)
  └── TVT_input null  → ensemble of 5 fold models
        ↓
Submission (14,151 rows, 100% matched)
```

---

## 📈 Results

| Version | Wells | Samples | Features | CV RMSE |
|---|---|---|---|---|
| v3 — LightGBM Baseline | 300 (sampled) | ~15K | 24 | 3.224 |
| v3 — LightGBM Full (Kaggle) | 773 | ~3.7M | 24 | 0.393 |
| **v4 — LightGBM Full + Better Features** | **773** | **3.78M** | **46** | **0.325** |

> **v3 → v4 improvement: -17% RMSE** via better feature engineering and full data utilization.

---

## 🏆 Feature Importance (v4)

Top features driving predictions:

| Rank | Feature | Description |
|---|---|---|
| 1 | `MD_norm` | Normalized measured depth position |
| 2 | `MD` | Raw measured depth |
| 3 | `dist_to_known` | Distance to nearest known TVT anchor |
| 4 | `dist_to_known_norm` | Normalized distance to anchor |
| 5 | `tw_GR_at_TVT` | Typewell GR value at predicted TVT |
| 6 | `TVT_input_gradient` | Rate of change of interpolated TVT |
| 7 | `TVT_input_cubic` | Cubic spline interpolated TVT |
| 8 | `Y` | Y spatial coordinate |
| 9 | `Z` | Z depth coordinate |
| 10 | `tw_TVT_pct` | Position within typewell TVT range |

---

## 📁 Project Structure

```
rogii-wellbore-geology-prediction/
├── README.md
├── .gitignore
├── src/
│   ├── rogii_pipeline_v3.py     # Baseline pipeline (300 wells, 24 features)
│   ├── rogii_pipeline_v4.py     # Improved local pipeline (773 wells, 46 features)
│   └── rogii_kaggle_v4.py       # Kaggle notebook version (full data)
├── data/
│   └── .gitkeep                 # Raw data not tracked (too large)
└── outputs/
    └── .gitkeep
```

---

## 🚀 How to Run

### Local (v4 — Full Pipeline)
```bash
# Install dependencies
pip install lightgbm scikit-learn scipy pandas numpy

# Run improved pipeline
python src/rogii_pipeline_v4.py
```

### Kaggle Notebook
1. Open notebook in the competition
2. Add competition data
3. Run `src/rogii_kaggle_v4.py`
4. Submit output file

---

## 🔮 Future Improvements

- [ ] **XGBoost + CatBoost ensemble** — blend multiple models for diversity
- [ ] **Optuna hyperparameter tuning** — systematic optimization
- [ ] **Per-well model** — train separate model per well cluster
- [ ] **Sequence models (LSTM/GRU)** — TVT is sequential per depth
- [ ] **Typewell GR matching** — find most similar typewell per well
- [ ] **Pseudo-labeling** — use test predictions as training signal

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **LightGBM** — gradient boosting model
- **pandas / numpy** — data processing
- **scikit-learn** — cross-validation (KFold)
- **scipy** — cubic spline interpolation, typewell GR correlation

---

## 👤 Author

**Rangga** — Python Backend & ML Engineer

Built as part of competitive ML practice and portfolio development.  
Competition: [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
