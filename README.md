# Week 2 Internship Task — Steel Industry Energy Consumption

**Deep Exploratory Analysis, Feature Engineering & Baseline Regression Modeling**

---

##  Project Overview

The project analyzes real energy consumption data from a steel manufacturing plant to understand
what drives energy usage and to build a baseline machine learning model that can predict `Usage_kWh`.

The work is split into two notebooks:

1. **`week2_eda.ipynb`** — Deep exploratory data analysis and feature engineering. Investigates data
   quality, extracts time-based features, engineers new columns, detects outliers, and identifies the
   strongest predictors of energy usage.
2. **`week2_baseline_models.ipynb`** — Uses the engineered dataset to train and compare four regression
   models (Linear Regression, Ridge Regression, Decision Tree, Random Forest), evaluate them properly
   with cross-validation, and select the best-performing baseline model.

Together these notebooks represent the full workflow of a real machine learning project: from raw data
to a validated baseline model.

---

## 📊 Dataset Information

| Detail | Value |
|---|---|
| **Name** | Steel Industry Energy Consumption Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip) |
| **Rows** | 35,040 |
| **Original Columns** | 11 |
| **Missing Values** | 0 |
| **Duplicate Rows** | 0 |
| **Time Span** | Full year of 15-minute interval readings |

**Original columns:** `date`, `Usage_kWh`, `Lagging_Current_Reactive.Power_kVarh`,
`Leading_Current_Reactive_Power_kVarh`, `CO2(tCO2)`, `Lagging_Current_Power_Factor`,
`Leading_Current_Power_Factor`, `NSM`, `WeekStatus`, `Day_of_week`, `Load_Type`

The dataset is stored in [`data/Steel_industry_data.csv`](data/Steel_industry_data.csv).


##  Environment Setup

**Option 1 — Google Colab (recommended, zero setup)**
1. Open either notebook in [Google Colab](https://colab.research.google.com/).
2. Run the first cell and upload `Steel_industry_data.csv` when prompted.
3. Run all cells top to bottom.

**Option 2 — Local Setup**
```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

**Key libraries used:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib` — full
pinned versions are listed in [`requirements.txt`](requirements.txt).

---

##  Feature Engineering Steps

| # | Feature | Description |
|---|---|---|
| 1 | `Hour` | Hour of day (0–23) extracted from `date` |
| 2 | `DayOfWeek_Num` | Day of week as a number (0=Monday … 6=Sunday) |
| 3 | `Month` | Month extracted from `date` |
| 4 | `Is_Weekend` | 1 if Saturday/Sunday, else 0 |
| 5 | `Power_Factor_Ratio` | `Leading_Current_Power_Factor` ÷ `Lagging_Current_Power_Factor` |
| 6 | `High_Load` | 1 if `Usage_kWh` is above the 75th percentile (**51.24 kWh**), else 0 *(used only for EDA — dropped before modeling to prevent data leakage)* |

##  EDA Findings (Quantitative Insights)

- **Data quality:** 0 missing values, 0 duplicate rows — clean dataset requiring no imputation.
- **Outliers:** IQR method flagged **328 outliers (0.94% of data)** in `Usage_kWh`, valid range
  `[-68.86, 123.29]` kWh. These correspond almost entirely to Maximum Load periods rather than sensor errors.
- **Top 3 features correlated with `Usage_kWh`** *(excluding the derived `High_Load` flag, which is
  trivially correlated by construction)*:
  1. `CO2(tCO2)` — r = **0.99**
  2. `Lagging_Current_Reactive.Power_kVarh` — r = **0.90**
  3. `Lagging_Current_Power_Factor` — r = **0.39**
- **Load Type impact:** Average usage scales sharply from Light → Medium → Maximum Load, confirming
  `Load_Type` is a major driver of consumption.
- **Hourly pattern:** Usage rises during working hours and drops overnight — a clear daily cycle tied
  to production shifts.
- **Hypothesis:** Energy spikes are primarily driven by the *scheduling of high-intensity production
  shifts* (captured by `Load_Type` and `Hour`) rather than random fluctuation.

  <img width="1185" height="585" alt="boxplot_outliers" src="https://github.com/user-attachments/assets/f81b0c8e-aab2-43a8-bea8-04f88343f0e9" />
 — Usage_kWh boxplot showing outliers

<img width="1494" height="1335" alt="image" src="https://github.com/user-attachments/assets/d23d4667-fce6-4443-b4e5-74d6b515f782" />
 — full correlation heatmap

<img width="1035" height="735" alt="image" src="https://github.com/user-attachments/assets/5c3b3f94-3db4-452e-a9c5-66145049e6f9" />
— grouped bar chart

<img width="1035" height="735" alt="image" src="https://github.com/user-attachments/assets/eb5963ba-90f1-4efa-9a02-fe518cb30e23" />
— hourly line chart

---

##  Model Training Process

1. Loaded the engineered dataset from Part 1 and dropped `date` and `High_Load` (leakage columns).
2. Encoded categorical columns (`WeekStatus`, `Day_of_week`, `Load_Type`) using **One-Hot Encoding**,
   chosen over Label Encoding because these categories are nominal (no natural order) — Label Encoding
   would falsely imply ranking, which particularly distorts linear models.
3. Split data **80% train / 20% test** with `random_state=42` for reproducibility.
4. Trained 4 models: **Linear Regression, Ridge Regression, Decision Tree Regressor, Random Forest Regressor**.
5. Evaluated each on the test set using **MAE, RMSE, and R²**, plus **5-fold cross-validation RMSE**
   to check consistency across data splits.

<img width="1335" height="313" alt="image" src="https://github.com/user-attachments/assets/52535f8e-8590-4936-98ee-d18ad20047f4" />
— printed metrics for all 4 models.

---

##  Results and Conclusions (Quantitative)

| Model | MAE | Test RMSE | R² | 5-Fold CV RMSE |
|---|---|---|---|---|
| **Random Forest**  | **0.35** | **1.05** | **0.999** | **1.02** |
| Decision Tree | 0.55 | 1.51 | 0.998 | 1.48 |
| Linear Regression | 2.63 | 4.15 | 0.985 | 4.51 |
| Ridge Regression | 4.36 | 6.27 | 0.965 | 6.23 |

**Best model: Random Forest Regressor**
- Lowest Test RMSE (1.05 kWh) and highest R² (0.999).
- Gap between CV RMSE (1.02) and Test RMSE (1.05) is very small (~0.03) → **no significant overfitting**,
  the model generalizes well to unseen data.
- Linear/Ridge Regression underfit because the true relationship between features and `Usage_kWh` is
  non-linear (tree-based models capture threshold effects like Load Type shifts that linear models miss).
- Decision Tree alone shows more variance across folds than Random Forest, which reduces variance through
  ensembling — hence Random Forest is more stable.

** Model carried forward: Random Forest Regressor**, saved as `Random_Forest_best_model.pkl`.

**Top predictive features (Random Forest importance):** `CO2(tCO2)`, `Lagging_Current_Reactive.Power_kVarh`,
and `NSM` (seconds since midnight) ranked highest — consistent with the EDA correlation findings.

<img width="1185" height="735" alt="image" src="https://github.com/user-attachments/assets/959f1bd9-d0ca-49a0-9bbf-5c11cd784bfa" />
 — RMSE comparison across all 4 models

 <img width="1034" height="1035" alt="image" src="https://github.com/user-attachments/assets/106ad5eb-f1db-4cb3-a73f-0ca733c645ba" />
— Predicted vs Actual scatter for Random Forest

<img width="1185" height="735" alt="image" src="https://github.com/user-attachments/assets/6b4bfae8-ee13-45e0-bad7-923e3f44f0df" />
— Top 10 feature importances

---

##  Key Takeaways

- Energy usage is highly predictable from electrical load measurements (CO2 output, reactive power) and
  time-of-day/load-type context — R² of 0.999 shows the baseline model already explains nearly all variance.
- Random Forest is the recommended baseline to carry forward into any future tuning or deployment work.
- Next steps could include hyperparameter tuning (GridSearchCV), testing gradient-boosted models
  (XGBoost/LightGBM), and deploying the saved model via a simple API or Streamlit app.

---

## Author

Muqadas — BS Business Data Analytics, COMSATS University Islamabad
Internship: ITSimplera Solutions
