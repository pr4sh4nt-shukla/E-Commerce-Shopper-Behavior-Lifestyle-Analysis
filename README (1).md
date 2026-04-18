# 🛒 Customer Intelligence System

A full end-to-end machine learning pipeline for e-commerce customer analytics — from raw behavioral data to interpretable ML predictions. The project spans four phases: Exploratory Data Analysis, Preprocessing & Feature Engineering, Customer Segmentation, and GPU-accelerated Machine Learning.

---

## 📁 Project Structure

```
├── 01_EDA.ipynb                  # Phase 1: Exploratory Data Analysis
├── 02_Preprocessing.ipynb        # Phase 2: Preprocessing & Feature Engineering
├── 03_Segmentation.ipynb         # Phase 3: Customer Segmentation (KMeans)
├── 04_ML_Models_GPU.ipynb        # Phase 4: ML Models with GPU support
│
├── e_commerce_shopper_behaviour_and_lifestyle.csv   # Raw dataset (user-provided)
│
├── df_processed_raw.csv          # Output of Phase 2 (raw-scaled)
├── df_processed_std.csv          # Output of Phase 2 (StandardScaler)
├── df_processed_mm.csv           # Output of Phase 2 (MinMaxScaler)
└── df_segmented.csv              # Output of Phase 3 (with cluster labels)
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | E-commerce Shopper Behaviour & Lifestyle |
| Rows | ~1,000,000 |
| Columns | 45 |
| Key features | Demographics, spending behavior, digital engagement, loyalty signals, risk indicators |

---

## 🔬 Pipeline Overview

### Phase 1 — Exploratory Data Analysis (`01_EDA.ipynb`)

Understand the dataset before touching it.

- **Dataset overview:** shape, dtypes, statistical summary
- **Data quality:** missing value audit, duplicate detection
- **Univariate analysis:**
  - Demographics (age, gender, occupation)
  - Spending & purchase behavior (monthly spend, order value, impulse purchases)
  - Binary features (loyalty membership, premium subscription, weekend shopper)
  - Digital engagement metrics (session time, ad clicks, conversion & abandonment rates)
- **Bivariate analysis:**
  - Income vs. spending scatter plots
  - Loyalty members vs. non-members comparison
  - Age group spending and impulse-buying trends
  - Premium vs. free user behavioral comparison

---

### Phase 2 — Preprocessing & Feature Engineering (`02_Preprocessing.ipynb`)

Prepare a clean feature matrix ready for clustering and ML.

**Categorical Encoding**
- Binary Yes/No columns → Label Encoding (0/1)
- Gender → Label Encoding
- Low-cardinality nominals (occupation, country, etc.) → One-Hot Encoding

**Numeric Scaling** (two separate scaled copies are saved)

| Scaler | Use-case | Rationale |
|---|---|---|
| `StandardScaler` | Logistic Regression, Ridge, PCA | Zero mean, unit variance |
| `MinMaxScaler` | KMeans Clustering | Bounded [0,1], preserves distance metrics |

**Feature Engineering** (3 composite behavioral scores)

| Feature | Components | What It Captures |
|---|---|---|
| `engagement_score` | ad clicks, notification response, app usage | Digital channel responsiveness |
| `value_score` | monthly spend × loyalty × referral count | Customer lifetime value signal |
| `risk_score` | return rate, cart abandonment, checkout abandonments | Churn and loss risk |

**Outputs:** `df_processed_raw.csv`, `df_processed_std.csv`, `df_processed_mm.csv`

---

### Phase 3 — Customer Segmentation (`03_Segmentation.ipynb`)

Discover natural customer personas using unsupervised learning.

**Clustering Features** (curated subset of behavioral signals)
- Spending & value: `monthly_spend`, `average_order_value`, `weekly_purchases`
- Loyalty & retention: `loyalty_program_member`, `brand_loyalty_score`, `referral_count`
- Engagement & risk: `engagement_score`, `value_score`, `risk_score`, `purchase_conversion_rate`, `return_rate`, `cart_abandonment_rate`

**Optimal K Selection**
- Elbow method (inertia / WCSS)
- Silhouette score (cluster separation quality)
- Sampled at 50,000 rows for speed; final KMeans fitted on full dataset

**Visualization**
- 2D and 3D PCA scatter plots colored by persona
- Silhouette plots per cluster
- PCA loadings — which features drive PC1 & PC2
- Heatmap of cluster means (Z-scored)
- Radar charts and box plots by persona

**Persona Naming:** Clusters are automatically named based on ranking of spend, loyalty, engagement, risk, and conversion rate signals.

**Output:** `df_segmented.csv` (original data + `cluster` and `persona` columns)

---

### Phase 4 — Machine Learning Models (`04_ML_Models_GPU.ipynb`)

Train, evaluate, and explain four ML models — with automatic GPU acceleration when available.

| Model | Target | Task | Algorithm |
|---|---|---|---|
| Model 1 | `return_rate` (high/low) | Binary Classification | Random Forest |
| Model 2 | `purchase_conversion_rate` | Regression | XGBoost |
| Model 3 | `premium_subscription` | Binary Classification | Logistic Regression |
| Model 4 | `monthly_spend` | Regression | Ridge + XGBoost (compared) |

**GPU Support**
- Auto-detects CUDA availability at runtime
- Uses `cuML` (NVIDIA RAPIDS) for Random Forest and Logistic Regression on GPU
- XGBoost uses `device='cuda'` when available
- Falls back to CPU (`scikit-learn`) transparently

**Evaluation Metrics**
- Classifiers: Accuracy, ROC-AUC, Confusion Matrix, ROC curve
- Regressors: R², RMSE, Residual plots, Learning curves

**SHAP Explainability** (Phase 7 of the notebook)
- `TreeExplainer` applied to Random Forest (`return_rate`) and XGBoost (`conversion_rate`, `monthly_spend`)
- Summary beeswarm plots, feature importance bar charts, and dependence plots
- Provides per-sample, per-feature attributions for business interpretability

---

## ⚙️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost shap
```

For GPU acceleration (optional):
```bash
# Requires CUDA-capable GPU and NVIDIA RAPIDS
pip install cuml-cu11  # or cuml-cu12 depending on CUDA version
```

---

## 🚀 How to Run

Run the notebooks in order — each phase outputs data consumed by the next:

```
01_EDA.ipynb
    ↓
02_Preprocessing.ipynb  →  df_processed_raw/std/mm.csv
    ↓
03_Segmentation.ipynb   →  df_segmented.csv
    ↓
04_ML_Models_GPU.ipynb
```

Before running, update the `FILE_PATH` variable in each notebook to point to your local copy of the dataset:

```python
FILE_PATH = 'e_commerce_shopper_behaviour_and_lifestyle.csv'
```

---

## 📌 Key Design Decisions

- **Never mutate raw data** — all preprocessing works on `.copy()` of the original dataframe.
- **OHE columns are excluded from KMeans** — sparse binary dummies dilute distance metrics.
- **MinMax scaling for clustering, Standard scaling for linear models** — each scaler is chosen for the downstream task.
- **Engineered scores are computed on raw data first**, then scaled alongside other features.
- **SHAP on sampled data** — background set capped at 5,000 rows for tractable computation without sacrificing explanation quality.
