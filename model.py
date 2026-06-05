"""
model.py  –  Sylhet Real Estate Price Prediction
=================================================
Pipeline
--------
1.  Load & clean dataset
2.  Exploratory visualisations (correlation heatmap, distributions)
3.  Feature engineering
4.  Proper three-way split: 60 % train | 20 % val | 20 % test
    - Hyper-parameter search on train+val (CV inside that portion)
    - Final hold-out evaluation on test only  → no data leakage
5.  Seven models: Linear Regression, Ridge, Decision Tree,
    Random Forest, Gradient Boosting, XGBoost, SVR  (all tuned)
6.  Residual analysis, learning curve, model comparison
7.  Save best model + artefacts
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
    learning_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# =========================================================
# GLOBAL STYLE
# =========================================================
THESIS_BLUE = "#1A3A5C"
ACCENT      = "#2E86AB"
ACCENT2     = "#E84855"
ACCENT3     = "#F4A261"
ACCENT4     = "#3BB273"
BG          = "#F8F9FA"
GRID_COLOR  = "#DDE3EA"
TEXT_COLOR  = "#1A3A5C"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#CCCCCC",
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlesize":    15,
    "axes.titleweight":  "bold",
    "axes.titlepad":     14,
    "axes.labelsize":    11,
    "axes.labelpad":     8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.8,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
    "figure.dpi":        130,
    "font.family":       "DejaVu Sans",
})

# =========================================================
# 1. LOAD DATASET
# =========================================================
df = pd.read_csv("sylhet_real_estate.csv")
df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())
print(f"Raw dataset size: {df.shape}")

# =========================================================
# 2. CLEAN DATA
# =========================================================

# --- Location ---
df["Location"] = (
    df["Location"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)
df["Location"] = df["Location"].replace({
    "ZindaBazar, Sylhet":          "Zindabazar, Sylhet",
    "Shahjalal Upashahar, Sylhet": "Shahjalal Uposhahar, Sylhet",
    "Mendibagh, Sylhete":          "Mendibagh, Sylhet",
    "Kazirbazar, Sylhete":         "Kazirbazar, Sylhet",
})

# --- Price ---
df["Selling price (BDT)"] = pd.to_numeric(
    df["Selling price (BDT)"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.extract(r"(\d+)")[0],
    errors="coerce",
)

# --- Floor Level → is_ground_floor binary flag ---
# "Whole house" and ground-level entries get flag=1, upper floors flag=0
df["Is_Ground_Or_Whole"] = df["Floor Level"].astype(str).str.lower().apply(
    lambda x: 1 if ("whole" in x or "ground" in x) else 0
)

# Drop rows with missing values in key columns
df = df.dropna(subset=["Selling price (BDT)", "Location", "Property Type",
                        "Size (sqft)", "Bedrooms", "Bathrooms"])
df = df.reset_index(drop=True)

# Crore column for readable axes
df["Price_Crore"] = df["Selling price (BDT)"] / 1e7

print(f"Dataset size after cleaning: {df.shape}")

# =========================================================
# 3. OUTLIER DETECTION  (domain-informed threshold)
# =========================================================
print("\n==============================")
print("OUTLIER DETECTION")
print("==============================")

Q1  = df["Selling price (BDT)"].quantile(0.25)
Q3  = df["Selling price (BDT)"].quantile(0.75)
IQR = Q3 - Q1
# Use 3× IQR (less aggressive) to preserve legitimate luxury properties
lower_bound = max(0, Q1 - 3.0 * IQR)
upper_bound = Q3 + 3.0 * IQR

outliers = df[
    (df["Selling price (BDT)"] < lower_bound) |
    (df["Selling price (BDT)"] > upper_bound)
]

print(f"IQR Lower Bound : BDT {lower_bound:,.0f}")
print(f"IQR Upper Bound : BDT {upper_bound:,.0f}")
print(f"Outliers Found  : {len(outliers)}")

if len(outliers) > 0:
    print("Outlier rows:")
    print(outliers[["Location", "Property Type", "Size (sqft)", "Selling price (BDT)"]])
    df = df[
        (df["Selling price (BDT)"] >= lower_bound) &
        (df["Selling price (BDT)"] <= upper_bound)
    ].reset_index(drop=True)
    print(f"Dataset size after removing outliers: {df.shape}")
else:
    print("No outliers detected — full dataset retained.")

# =========================================================
# 4. EXPLORATORY DATA ANALYSIS PLOTS
# =========================================================

# --- 4a. Price Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

axes[0].hist(df["Price_Crore"], bins=25, color=THESIS_BLUE,
             edgecolor="white", zorder=3)
axes[0].set_xlabel("Selling Price (BDT Crore)")
axes[0].set_ylabel("Count")
axes[0].set_title("Property Price Distribution")
axes[0].yaxis.grid(True, zorder=0)
axes[0].set_axisbelow(True)

axes[1].hist(np.log(df["Selling price (BDT)"]), bins=25,
             color=ACCENT, edgecolor="white", zorder=3)
axes[1].set_xlabel("log(Selling Price)")
axes[1].set_ylabel("Count")
axes[1].set_title("Log-Transformed Price Distribution")
axes[1].yaxis.grid(True, zorder=0)
axes[1].set_axisbelow(True)

fig.suptitle("Price Distributions (Raw vs Log-Transformed)",
             fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.02)
fig.tight_layout()
plt.show()

# --- 4b. Property Type Breakdown ---
pt_counts = df["Property Type"].value_counts()
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(BG)
colors_pt = [THESIS_BLUE, ACCENT, ACCENT3]
ax.bar(pt_counts.index, pt_counts.values, color=colors_pt,
       edgecolor="white", width=0.55)
for i, (idx, val) in enumerate(pt_counts.items()):
    ax.text(i, val + 1, str(val), ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=TEXT_COLOR)
ax.set_ylabel("Count")
ax.set_title("Property Type Distribution")
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
plt.show()

# --- 4c. Avg Price per sqft by Location ---
df["price_per_sqft"] = df["Selling price (BDT)"] / df["Size (sqft)"]
loc_avg = (df.groupby("Location")["price_per_sqft"]
           .mean()
           .sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
colors_loc = [THESIS_BLUE if i == 0 else ACCENT for i in range(len(loc_avg))]
ax.barh(loc_avg.index[::-1], loc_avg.values[::-1],
        color=colors_loc[::-1], edgecolor="white", height=0.65)
ax.set_xlabel("Avg Price per sqft (BDT)")
ax.set_title("Average Price per sqft by Location")
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=8)
fig.tight_layout()
plt.show()

# --- 4d. Correlation Heatmap ---
numeric_df = df.select_dtypes(include=np.number).drop(
    columns=["Price_Crore", "price_per_sqft"], errors="ignore"
)
correlation_matrix = numeric_df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor(BG)
sns.heatmap(
    correlation_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlBu_r", center=0, linewidths=0.5, linecolor="white",
    annot_kws={"size": 8.5}, cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    square=True, ax=ax,
)
ax.set_title("Feature Correlation Heatmap")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
plt.show()

# =========================================================
# 5. FEATURES & TARGET
# =========================================================
features = [
    "Size (sqft)",
    "Bedrooms",
    "Bathrooms",
    "Floor Number",
    "Is_Ground_Or_Whole",   # derived from Floor Level
    "Balcony",
    "Parking",
    "Lift",
    "CCTV",
    "Generator",
    "Location",
    "Property Type",
]

X = df[features].copy()
y_raw = df["Selling price (BDT)"]
y     = np.log(y_raw)          # log-transform target

# One-hot encode categoricals
X = pd.get_dummies(X, columns=["Location", "Property Type"])

print(f"\nEncoded feature count : {X.shape[1]}")
print("Property Type dummies :", [c for c in X.columns if "Property" in c])

# =========================================================
# 6. PROPER THREE-WAY SPLIT
#    60 % train  |  20 % validation  |  20 % test
#    • Hyper-parameter search (CV) runs on train+val
#    • Hold-out test set is NEVER touched during training
# =========================================================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42   # 0.25 × 0.80 = 0.20
)

print(f"\nSplit sizes → train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

# Scaling (fit only on train)
scaler         = StandardScaler()
X_train_sc     = scaler.fit_transform(X_train)
X_val_sc       = scaler.transform(X_val)
X_test_sc      = scaler.transform(X_test)
X_trainval_sc  = scaler.transform(X_trainval)

# =========================================================
# HELPERS
# =========================================================
results = {}

def evaluate(name, y_true_log, y_pred_log):
    """Metrics in original BDT space for interpretability."""
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : BDT {mae:,.0f}")
    print(f"  RMSE : BDT {rmse:,.0f}")
    return r2, mae, rmse


def run_cv(model, name, scaled=False):
    """CV on train+val portion only — test set stays unseen."""
    X_cv = X_trainval_sc if scaled else X_trainval
    cv = cross_val_score(model, X_cv, y_trainval, cv=5, scoring="r2")
    print(f"  CV R² : {np.round(cv, 4)}  mean={cv.mean():.4f} ±{cv.std():.4f}")
    return cv

# =========================================================
# 7. MODEL TRAINING & EVALUATION
# =========================================================

# ---------------------------------------------------------
# 7.1  LINEAR REGRESSION
# ---------------------------------------------------------
print("\n==============================")
print("1. LINEAR REGRESSION")
print("==============================")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
evaluate("Linear Regression", y_test, lr_preds)
run_cv(LinearRegression(), "Linear Regression")

# ---------------------------------------------------------
# 7.2  RIDGE REGRESSION  (cross-validated alpha)
# ---------------------------------------------------------
print("\n==============================")
print("2. RIDGE REGRESSION")
print("==============================")

ridge_cv_sel = RidgeCV(alphas=[0.01, 0.1, 1, 5, 10, 50, 100, 500], cv=5)
ridge_cv_sel.fit(X_train_sc, y_train)
best_alpha = ridge_cv_sel.alpha_
print(f"  Best alpha (RidgeCV): {best_alpha}")

ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train_sc, y_train)
ridge_preds = ridge_model.predict(X_test_sc)
evaluate("Ridge Regression", y_test, ridge_preds)

ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))])
run_cv(ridge_pipe, "Ridge Regression", scaled=False)

# ---------------------------------------------------------
# 7.3  DECISION TREE
# ---------------------------------------------------------
print("\n==============================")
print("3. DECISION TREE REGRESSOR")
print("==============================")

dt_model = DecisionTreeRegressor(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
evaluate("Decision Tree", y_test, dt_preds)
run_cv(DecisionTreeRegressor(random_state=42, max_depth=10), "Decision Tree")

# ---------------------------------------------------------
# 7.4  RANDOM FOREST  (RandomizedSearchCV on train+val)
# ---------------------------------------------------------
print("\n==============================")
print("4. RANDOM FOREST")
print("==============================")

rf_param_dist = {
    "n_estimators":      [100, 200, 300, 400],
    "max_depth":         [None, 8, 12, 16],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    rf_param_dist, n_iter=20, cv=5, scoring="r2",
    random_state=42, n_jobs=-1,
)
rf_search.fit(X_trainval, y_trainval)
print(f"  Best RF params : {rf_search.best_params_}")

rf_model = rf_search.best_estimator_
rf_preds = rf_model.predict(X_test)
evaluate("Random Forest", y_test, rf_preds)
run_cv(rf_model, "Random Forest")

# Feature Importance — Random Forest
fi_df  = (pd.DataFrame({"Feature": X.columns, "Importance": rf_model.feature_importances_})
          .sort_values("Importance", ascending=False))
top10  = fi_df.head(10).copy()
top10["Feature"] = (
    top10["Feature"]
    .str.replace("Location_", "", regex=False)
    .str.replace(", Sylhet", "", regex=False)
    .str.replace(" Sylhet", "", regex=False)
    .str.replace("Property Type_", "Type: ", regex=False)
)

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
colors_fi = [THESIS_BLUE] + [ACCENT] * (len(top10) - 1)
bars = ax.barh(top10["Feature"][::-1], top10["Importance"][::-1],
               color=colors_fi[::-1], edgecolor="white", height=0.65)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}", va="center", ha="left", fontsize=8.5, color=TEXT_COLOR)
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Top 10 Feature Importances — Random Forest")
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=9)
ax.legend(handles=[
    mpatches.Patch(color=THESIS_BLUE, label="Most important feature"),
    mpatches.Patch(color=ACCENT,      label="Other features"),
])
fig.tight_layout()
plt.show()

# Actual vs Predicted — Random Forest
y_test_cr   = np.exp(y_test.values) / 1e7
rf_preds_cr = np.exp(rf_preds) / 1e7

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(BG)
ax.scatter(y_test_cr, rf_preds_cr, c=ACCENT, edgecolors="white",
           s=65, linewidths=0.5, alpha=0.80, zorder=3)
lims = [min(y_test_cr.min(), rf_preds_cr.min()),
        max(y_test_cr.max(), rf_preds_cr.max())]
ax.plot(lims, lims, color=ACCENT2, linewidth=2, linestyle="--",
        label="Perfect prediction", zorder=4)
ax.text(0.05, 0.93, f"R² = {results['Random Forest']['R2']:.4f}",
        transform=ax.transAxes, fontsize=11, fontweight="bold", color=THESIS_BLUE,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID_COLOR))
ax.set_xlabel("Actual Price (BDT Crore)")
ax.set_ylabel("Predicted Price (BDT Crore)")
ax.set_title("Actual vs. Predicted House Prices\n(Random Forest — Hold-out Test Set)")
ax.legend()
ax.set_aspect("equal")
ax.xaxis.grid(True, zorder=0)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7.5  GRADIENT BOOSTING
# ---------------------------------------------------------
print("\n==============================")
print("5. GRADIENT BOOSTING REGRESSOR")
print("==============================")

gbm_param_dist = {
    "n_estimators":  [200, 300, 400],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth":     [3, 4, 5],
    "subsample":     [0.7, 0.8, 1.0],
}
gbm_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    gbm_param_dist, n_iter=20, cv=5, scoring="r2",
    random_state=42, n_jobs=-1,
)
gbm_search.fit(X_trainval, y_trainval)
print(f"  Best GBM params : {gbm_search.best_params_}")

gbm_model = gbm_search.best_estimator_
gbm_preds = gbm_model.predict(X_test)
evaluate("Gradient Boosting", y_test, gbm_preds)
run_cv(gbm_model, "Gradient Boosting")

# ---------------------------------------------------------
# 7.6  XGBOOST
# ---------------------------------------------------------
print("\n==============================")
print("6. XGBOOST REGRESSOR")
print("==============================")

xgb_param_dist = {
    "n_estimators":     [200, 300, 400],
    "learning_rate":    [0.01, 0.05, 0.1],
    "max_depth":        [3, 4, 5, 6],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}
xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    xgb_param_dist, n_iter=20, cv=5, scoring="r2",
    random_state=42, n_jobs=-1,
)
xgb_search.fit(X_trainval, y_trainval)
print(f"  Best XGBoost params : {xgb_search.best_params_}")

xgb_model = xgb_search.best_estimator_
xgb_preds = xgb_model.predict(X_test)
evaluate("XGBoost", y_test, xgb_preds)
run_cv(xgb_model, "XGBoost")

# Feature Importance — XGBoost
xgb_fi_df = (pd.DataFrame({"Feature": X.columns,
                            "Importance": xgb_model.feature_importances_})
             .sort_values("Importance", ascending=False))
xgb_top = xgb_fi_df.head(10).copy()
xgb_top["Feature"] = (
    xgb_top["Feature"]
    .str.replace("Location_", "", regex=False)
    .str.replace(", Sylhet", "", regex=False)
    .str.replace(" Sylhet", "", regex=False)
    .str.replace("Property Type_", "Type: ", regex=False)
)

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
colors_xgb = [ACCENT2] + [ACCENT3] * (len(xgb_top) - 1)
bars_xgb = ax.barh(xgb_top["Feature"][::-1], xgb_top["Importance"][::-1],
                   color=colors_xgb[::-1], edgecolor="white", height=0.65)
for bar in bars_xgb:
    w = bar.get_width()
    ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}", va="center", ha="left", fontsize=8.5, color=TEXT_COLOR)
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title("Top 10 Feature Importances — XGBoost")
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=9)
ax.legend(handles=[
    mpatches.Patch(color=ACCENT2, label="Most important feature"),
    mpatches.Patch(color=ACCENT3, label="Other features"),
])
fig.tight_layout()
plt.show()

# Actual vs Predicted — XGBoost
xgb_preds_cr = np.exp(xgb_preds) / 1e7

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(BG)
ax.scatter(y_test_cr, xgb_preds_cr, c=ACCENT2, edgecolors="white",
           s=65, linewidths=0.5, alpha=0.80, zorder=3)
lims = [min(y_test_cr.min(), xgb_preds_cr.min()),
        max(y_test_cr.max(), xgb_preds_cr.max())]
ax.plot(lims, lims, color=THESIS_BLUE, linewidth=2, linestyle="--",
        label="Perfect prediction", zorder=4)
ax.text(0.05, 0.93, f"R² = {results['XGBoost']['R2']:.4f}",
        transform=ax.transAxes, fontsize=11, fontweight="bold", color=THESIS_BLUE,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID_COLOR))
ax.set_xlabel("Actual Price (BDT Crore)")
ax.set_ylabel("Predicted Price (BDT Crore)")
ax.set_title("Actual vs. Predicted House Prices\n(XGBoost — Hold-out Test Set)")
ax.legend()
ax.set_aspect("equal")
ax.xaxis.grid(True, zorder=0)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7.7  SVR  (now properly tuned with RandomizedSearchCV)
# ---------------------------------------------------------
print("\n==============================")
print("7. SUPPORT VECTOR REGRESSOR")
print("==============================")

svr_param_dist = {
    "svr__C":       [0.1, 1, 10, 50, 100, 500],
    "svr__gamma":   ["scale", "auto", 0.01, 0.1],
    "svr__epsilon": [0.05, 0.1, 0.2, 0.5],
}
svr_pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
svr_search = RandomizedSearchCV(
    svr_pipe, svr_param_dist, n_iter=24, cv=5, scoring="r2",
    random_state=42, n_jobs=-1,
)
svr_search.fit(X_trainval, y_trainval)
print(f"  Best SVR params : {svr_search.best_params_}")

svr_model = svr_search.best_estimator_
svr_preds = svr_model.predict(X_test)
evaluate("SVR", y_test, svr_preds)
run_cv(svr_model, "SVR", scaled=False)

# =========================================================
# 8. RESIDUAL ANALYSIS — best model (Gradient Boosting)
# =========================================================
print("\n==============================")
print("RESIDUAL ANALYSIS (Gradient Boosting)")
print("==============================")

residuals   = np.exp(y_test.values) - np.exp(gbm_preds)
residuals_L = residuals / 1e5   # in Lakh

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

axes[0].scatter(np.exp(gbm_preds) / 1e7, residuals_L,
                alpha=0.65, color=ACCENT, edgecolors="white", s=55, zorder=3)
axes[0].axhline(0, color=ACCENT2, linewidth=1.8, linestyle="--", zorder=4)
axes[0].set_xlabel("Predicted Price (BDT Crore)")
axes[0].set_ylabel("Residual (BDT Lakh)")
axes[0].set_title("Residuals vs Predicted — Gradient Boosting")
axes[0].yaxis.grid(True, zorder=0)
axes[0].set_axisbelow(True)

axes[1].hist(residuals_L, bins=20, color=THESIS_BLUE, edgecolor="white", zorder=3)
axes[1].axvline(0, color=ACCENT2, linewidth=1.8, linestyle="--", zorder=4)
axes[1].set_xlabel("Residual (BDT Lakh)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution — Gradient Boosting")
axes[1].yaxis.grid(True, zorder=0)
axes[1].set_axisbelow(True)

fig.tight_layout()
plt.show()

# =========================================================
# 9. LEARNING CURVE — Gradient Boosting
# =========================================================
print("\n==============================")
print("LEARNING CURVE (Gradient Boosting)")
print("==============================")

train_sizes, train_scores, val_scores = learning_curve(
    gbm_model, X_trainval, y_trainval,
    cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="r2", n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(BG)
ax.plot(train_sizes, train_mean, color=THESIS_BLUE, linewidth=2,
        marker="o", markersize=5, label="Training R²")
ax.fill_between(train_sizes, train_mean - train_std,
                train_mean + train_std, alpha=0.12, color=THESIS_BLUE)
ax.plot(train_sizes, val_mean, color=ACCENT2, linewidth=2,
        marker="s", markersize=5, label="Validation R²")
ax.fill_between(train_sizes, val_mean - val_std,
                val_mean + val_std, alpha=0.12, color=ACCENT2)
ax.set_xlabel("Training Set Size")
ax.set_ylabel("R² Score")
ax.set_title("Learning Curve — Gradient Boosting (CV on train+val)")
ax.legend()
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
plt.show()

# =========================================================
# 10. MODEL COMPARISON
# =========================================================
print("\n")
print("=" * 65)
print("MODEL COMPARISON SUMMARY  (evaluated on hold-out test set)")
print("=" * 65)
print(f"{'Model':<25} {'R²':>8} {'MAE (BDT)':>15} {'RMSE (BDT)':>15}")
print("-" * 65)
for name, m in sorted(results.items(), key=lambda x: x[1]["R2"], reverse=True):
    print(f"{name:<25} {m['R2']:>8.4f} {m['MAE']:>15,.0f} {m['RMSE']:>15,.0f}")
print("=" * 65)

# Bar chart
names_sorted = sorted(results, key=lambda x: results[x]["R2"], reverse=True)
r2_sorted    = [results[n]["R2"] for n in names_sorted]

bar_colors = []
for r2 in r2_sorted:
    if r2 == max(r2_sorted):
        bar_colors.append(THESIS_BLUE)
    elif r2 >= 0.80:
        bar_colors.append(ACCENT)
    elif r2 >= 0.60:
        bar_colors.append(ACCENT3)
    else:
        bar_colors.append(ACCENT2)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
bars = ax.bar(names_sorted, r2_sorted, color=bar_colors, edgecolor="white", width=0.55)
for bar, score in zip(bars, r2_sorted):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005, f"{score:.4f}",
            ha="center", va="bottom", fontsize=9,
            color=TEXT_COLOR, fontweight="bold")
ax.set_ylim(0, 1.10)
ax.set_ylabel("R² Score")
ax.set_title("Model Comparison — R² Score on Hold-out Test Set")
ax.tick_params(axis="x", rotation=20, labelsize=9)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.legend(handles=[
    mpatches.Patch(color=THESIS_BLUE, label="Best model"),
    mpatches.Patch(color=ACCENT,      label="R² ≥ 0.80"),
    mpatches.Patch(color=ACCENT3,     label="R² ≥ 0.60"),
    mpatches.Patch(color=ACCENT2,     label="R² < 0.60"),
])
fig.tight_layout()
plt.show()

# =========================================================
# 11. SAMPLE PREDICTION — all 7 models
# =========================================================
sample_raw = pd.DataFrame([{
    "Size (sqft)":        2500,
    "Bedrooms":           3,
    "Bathrooms":          2,
    "Floor Number":       5,
    "Is_Ground_Or_Whole": 0,
    "Balcony":            1,
    "Parking":            1,
    "Lift":               1,
    "CCTV":               0,
    "Generator":          0,
    "Location":           "Akhalia, Sylhet",
    "Property Type":      "Apartment",
}])

sample_enc   = pd.get_dummies(sample_raw, columns=["Location", "Property Type"])
sample_enc   = sample_enc.reindex(columns=X.columns, fill_value=0)
sample_enc_sc = scaler.transform(sample_enc)

print("\nSample Prediction — 2500 sqft Apartment, 3BR, Floor 5, Akhalia:")
preds_sample = {
    "Linear Regression": int(np.exp(lr_model.predict(sample_enc)[0])),
    "Ridge Regression":  int(np.exp(ridge_model.predict(sample_enc_sc)[0])),
    "Decision Tree":     int(np.exp(dt_model.predict(sample_enc)[0])),
    "Random Forest":     int(np.exp(rf_model.predict(sample_enc)[0])),
    "Gradient Boosting": int(np.exp(gbm_model.predict(sample_enc)[0])),
    "XGBoost":           int(np.exp(xgb_model.predict(sample_enc)[0])),
    "SVR":               int(np.exp(svr_model.predict(sample_enc)[0])),
}
for model_name, price in preds_sample.items():
    print(f"  {model_name:<22} : BDT {price:,}")

# =========================================================
# 12. PREDICTION INTERVAL (±1 RMSE from best model)
# =========================================================
best_model_name = max(results, key=lambda x: results[x]["R2"])
best_rmse       = results[best_model_name]["RMSE"]
print(f"\nBest model : {best_model_name}  (R² = {results[best_model_name]['R2']:.4f})")
print(f"RMSE used for confidence interval : BDT {best_rmse:,.0f}")
gbm_sample_price = preds_sample["Gradient Boosting"]
print(f"\nGradient Boosting estimate for sample property:")
print(f"  Point estimate : BDT {gbm_sample_price:,}")
print(f"  ±1 RMSE range  : BDT {int(gbm_sample_price - best_rmse):,}  –  BDT {int(gbm_sample_price + best_rmse):,}")

# =========================================================
# 13. SAVE MODELS & ARTEFACTS
# =========================================================
joblib.dump({
    "model": gbm_model,
    "columns": list(X.columns),
    "rmse": best_rmse
}, "gbm_bundle.pkl")

print("Model bundle saved successfully as gbm_bundle.pkl")

# Save all model metrics to JSON for display in app.py
import json
import os

print("Saving metrics file...")

metrics_out = {
    name: {
        "R2": round(m["R2"], 4),
        "MAE": int(m["MAE"]),
        "RMSE": int(m["RMSE"])
    }
    for name, m in results.items()
}

save_path = os.path.join(os.getcwd(), "model_metrics.json")

with open(save_path, "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"Saved model_metrics.json at: {save_path}")
print("File exists:", os.path.exists(save_path))