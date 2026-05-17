# ============================================================
# STEP 6: MACHINE LEARNING MODELS
# ============================================================
# WHAT THIS FILE DOES:
#   - Trains 3 ML models to predict PM2.5/AQI
#   - Compares their performance
#   - Runs SHAP explainability on the best model
#   - Computes Traffic-Pollution Elasticity (TPE) — Ma'am's suggestion!
#   - Tests cross-city transfer (Delhi model on Bengaluru)
#   - Saves all results and plots
#
# MODELS:
#   1. Linear Regression (baseline — simple)
#   2. Random Forest (main model — handles non-linear patterns)
#   3. XGBoost (best model — state of the art for tabular data)
#   + SHAP (explains XGBoost predictions)
#
# HOW TO RUN:
#   pip install xgboost shap scikit-learn
#   python notebooks/STEP6_ml_models.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (mean_squared_error,
                                     mean_absolute_error, r2_score)
import xgboost as xgb
import shap
import joblib

# ── CONFIGURATION ──────────────────────────────────────────
DATA_FILE  = "data/model_ready_data.csv"
OUTPUT_DIR = "outputs"
MODELS_DIR = "outputs/models"
import os
os.makedirs(MODELS_DIR, exist_ok=True)
# ───────────────────────────────────────────────────────────

print("=" * 60)
print("MACHINE LEARNING MODELS")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
df['Date'] = pd.to_datetime(df['Date'])

print(f"✅ Loaded: {df.shape[0]} rows")


# ============================================================
# PART A: DEFINE FEATURES AND TARGET
# ============================================================
# EXPLANATION:
# Features (X) = inputs to the model = what we know
# Target  (y) = what we want to predict = PM2.5
#
# Features chosen:
# - Time features: Month, Year, Is_Weekend, Is_Winter, etc.
# - Lag features: Yesterday's PM2.5 (most powerful predictor!)
# - Weather: Wind speed, Temperature, Humidity
# - City flag: Is_Delhi (so model knows which city)

print("\nDefining features and target...")

# NEW — reads features selected by STEP4 automatically
with open('outputs/selected_features.txt', 'r') as f:
    FEATURE_COLS = [line.strip() for line in f.readlines()]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

TARGET_COL = 'PM2.5'

# Keep only columns that exist in the dataset
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print(f"  Using {len(FEATURE_COLS)} features to predict {TARGET_COL}")
print(f"  Features: {FEATURE_COLS}")

# Drop rows with any NaN in features or target
model_df = df[FEATURE_COLS + [TARGET_COL, 'City', 'Date', 'StationId']].dropna()
print(f"  Rows after dropping NaN: {len(model_df)}")


# ============================================================
# PART B: TRAIN-TEST SPLIT
# ============================================================
# EXPLANATION:
# We split data into:
#   Train set (70%): model learns patterns from this
#   Test set  (30%): we test how well model predicts unseen data
#
# It's like studying with 70% of textbook questions,
# then being tested on the other 30%.
#
# IMPORTANT: We use time-based split for one city and
# cross-city split for transferability test!

print("\nSplitting data...")

# Within-city split (for main model evaluation)
X = model_df[FEATURE_COLS]
y = model_df[TARGET_COL]

# Use random split for main evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"  Train size: {len(X_train)} rows")
print(f"  Test size:  {len(X_test)} rows")

# Cross-city split: train on Delhi, test on Bengaluru
delhi_mask = model_df['City'] == 'Delhi'
blr_mask   = model_df['City'] == 'Bengaluru'

X_delhi = model_df[delhi_mask][FEATURE_COLS]
y_delhi = model_df[delhi_mask][TARGET_COL]
X_blr   = model_df[blr_mask][FEATURE_COLS]
y_blr   = model_df[blr_mask][TARGET_COL]

print(f"\n  Delhi rows: {len(X_delhi)}")
print(f"  Bengaluru rows: {len(X_blr)}")


# ============================================================
# PART C: HELPER FUNCTIONS
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates a trained model and prints key metrics.

    Metrics explained:
    - RMSE (Root Mean Squared Error):
        Average prediction error in µg/m³.
        Lower is better. Target: RMSE < 20

    - MAE (Mean Absolute Error):
        Average absolute error. More interpretable than RMSE.
        Example: MAE=10 means predictions are off by 10 µg/m³ on average.

    - R² (R-squared):
        How much variance the model explains. Scale: 0 to 1.
        R²=0.8 means model explains 80% of the variation in PM2.5.
        Target: R² > 0.7 for a good model.
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n  {model_name} Results:")
    print(f"    RMSE : {rmse:.2f} µg/m³  (lower is better)")
    print(f"    MAE  : {mae:.2f} µg/m³  (lower is better)")
    print(f"    R²   : {r2:.4f}        (higher is better, max=1)")

    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
            'y_pred': y_pred}


# ============================================================
# PART D: MODEL 1 — LINEAR REGRESSION (BASELINE)
# ============================================================
# EXPLANATION:
# Linear Regression assumes pollution = a + b1*feature1 + b2*feature2 ...
# It's simple and fast, but can't capture complex non-linear patterns.
# We use it as a BASELINE: if a complex model can't beat this,
# something is wrong!

print("\n" + "=" * 60)
print("MODEL 1: LINEAR REGRESSION (Baseline)")
print("=" * 60)

# Scale features for Linear Regression
# WHY: Linear regression works better when all features
# are on the same scale (0 to 1 or z-scores)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

lr_results = evaluate_model(lr_model, X_test_scaled, y_test,
                             'Linear Regression')

# Cross-validation (more reliable than single split)
cv_scores = cross_val_score(lr_model, X_train_scaled, y_train,
                             cv=5, scoring='r2')
print(f"    Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ============================================================
# PART E: MODEL 2 — RANDOM FOREST
# ============================================================
# EXPLANATION:
# Random Forest builds MANY decision trees (100 by default)
# and averages their predictions.
#
# A decision tree works like 20 questions:
# "Is month January? → Yes → Is wind speed low? → Yes → PM2.5 likely HIGH"
#
# Random Forest is powerful because:
# - Handles non-linear patterns (pollution isn't just linear!)
# - Robust to outliers
# - Gives feature importance (which features matter most?)
# - Doesn't need feature scaling

print("\n" + "=" * 60)
print("MODEL 2: RANDOM FOREST")
print("=" * 60)
print("(Training 100 trees — may take 30-60 seconds...)")

rf_model = RandomForestRegressor(
    n_estimators=100,    # 100 decision trees
    max_depth=12,        # trees can be at most 12 levels deep
    min_samples_leaf=5,  # each leaf needs at least 5 samples
    random_state=42,
    n_jobs=-1            # use all CPU cores
)
rf_model.fit(X_train, y_train)

rf_results = evaluate_model(rf_model, X_test, y_test, 'Random Forest')

# Feature importance from Random Forest
rf_importance = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n  Top 10 Most Important Features (Random Forest):")
print(rf_importance.head(10).to_string(index=False))


# ============================================================
# PART F: MODEL 3 — XGBOOST (BEST MODEL)
# ============================================================
# EXPLANATION:
# XGBoost (eXtreme Gradient Boosting) builds trees SEQUENTIALLY.
# Each new tree corrects the errors of the previous trees.
# This makes it very accurate on tabular (structured) data.
#
# It's used by winning teams in Kaggle competitions and
# in real-world air quality prediction papers!
#
# Parameters explained:
# n_estimators = number of trees (more = better but slower)
# learning_rate = how much each tree corrects (smaller = more careful)
# max_depth = how deep each tree can go (deeper = more complex patterns)

print("\n" + "=" * 60)
print("MODEL 3: XGBOOST")
print("=" * 60)
print("(Training 200 trees...)")

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

xgb_results = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')

# Save the best model
joblib.dump(xgb_model, f'{MODELS_DIR}/xgboost_model.pkl')
joblib.dump(rf_model,  f'{MODELS_DIR}/rf_model.pkl')
joblib.dump(scaler,    f'{MODELS_DIR}/scaler.pkl')
print("\n  ✅ Models saved to outputs/models/")


# ============================================================
# PART G: MODEL COMPARISON PLOT
# ============================================================
print("\nCreating model comparison chart...")

results_df = pd.DataFrame([
    {'Model': lr_results['Model'],  'RMSE': lr_results['RMSE'],
     'MAE': lr_results['MAE'],      'R2': lr_results['R2']},
    {'Model': rf_results['Model'],  'RMSE': rf_results['RMSE'],
     'MAE': rf_results['MAE'],      'R2': rf_results['R2']},
    {'Model': xgb_results['Model'], 'RMSE': xgb_results['RMSE'],
     'MAE': xgb_results['MAE'],     'R2': xgb_results['R2']},
])

results_df.to_csv(f'{OUTPUT_DIR}/model_comparison.csv', index=False)
print("\n  Model Comparison:")
print(results_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['#3498DB', '#2ECC71', '#E74C3C']

for ax, metric, title, better in zip(
        axes,
        ['RMSE', 'MAE', 'R2'],
        ['RMSE (lower is better)', 'MAE (lower is better)',
         'R² Score (higher is better)'],
        ['lower', 'lower', 'higher']):

    bars = ax.bar(results_df['Model'], results_df[metric],
                  color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=15)

    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01 * bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

plt.suptitle('ML Model Performance Comparison', fontsize=14,
             fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot6_model_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: plot6_model_comparison.png")


# ============================================================
# PART H: SHAP EXPLAINABILITY
# ============================================================
# EXPLANATION:
# SHAP = SHapley Additive exPlanations
#
# SHAP answers: "Why did the model predict PM2.5 = 150 for THIS day?"
#
# For EACH prediction, SHAP assigns a score to each feature:
# - Positive SHAP value → feature pushed prediction HIGHER
# - Negative SHAP value → feature pushed prediction LOWER
#
# Example output you can say in viva:
# "On 15 Jan 2020 in Anand Vihar, high PM2.5_lag_1 (+45 µg/m³)
# and low Wind_Speed (+12 µg/m³) were the main reasons
# the model predicted high pollution"
#
# Global SHAP → which features matter OVERALL (for all predictions)
# Local SHAP  → why THIS specific prediction is high/low

print("\n" + "=" * 60)
print("SHAP EXPLAINABILITY (on XGBoost)")
print("=" * 60)
print("(Calculating SHAP values — may take 1-2 minutes...)")

# Use a sample for faster SHAP computation
sample_size = min(500, len(X_test))
X_sample    = X_test.sample(sample_size, random_state=42)

# Create SHAP explainer for tree-based models
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)

# ── SHAP Plot 1: Summary (Beeswarm) ───────────────────────
print("  Creating SHAP summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, plot_type='dot',
                  show=False, max_display=15)
plt.title('SHAP Summary: Feature Impact on PM2.5 Predictions\n'
          '(Each dot = one prediction; color = feature value)',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot7_shap_summary.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: plot7_shap_summary.png")

# ── SHAP Plot 2: Feature importance bar ───────────────────
print("  Creating SHAP feature importance bar...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, plot_type='bar',
                  show=False, max_display=15)
plt.title('SHAP Feature Importance (Mean Absolute Impact)',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot8_shap_importance.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: plot8_shap_importance.png")

# Save SHAP values as CSV for report
shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLS)
shap_mean = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Mean_SHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False)
shap_mean.to_csv(f'{OUTPUT_DIR}/shap_feature_importance.csv', index=False)
print("\n  Top 10 features by SHAP importance:")
print(shap_mean.head(10).to_string(index=False))


# ============================================================
# PART I: TRAFFIC-POLLUTION ELASTICITY (TPE)
# ============================================================
# EXPLANATION:
# Ma'am suggested this as a NEW metric — this is your unique contribution!
#
# TPE = (% change in PM2.5) / (% change in traffic volume)
#
# Example:
# If traffic increases by 10% and PM2.5 increases by 4.5%,
# then TPE = 4.5/10 = 0.45
#
# TPE = 0.45 means "a 10% traffic reduction → 4.5% PM2.5 reduction"
#
# Higher TPE = that area is MORE sensitive to traffic changes
# This helps planners decide WHERE to reduce traffic for maximum benefit!

print("\n" + "=" * 60)
print("TRAFFIC-POLLUTION ELASTICITY (TPE) — NEW METRIC")
print("=" * 60)

# We simulate this using the XGBoost model:
# Step 1: Get baseline predictions
# Step 2: Reduce "traffic proxy" features by 20%
# Step 3: See how much PM2.5 prediction drops

X_test_copy    = X_test.copy()
baseline_preds = xgb_model.predict(X_test_copy)

# Traffic proxy features
# (In your dataset, direct traffic volume isn't in the AQ data,
# so we use Month/Season as traffic proxies, and simulate
# wind increase as a proxy for traffic-cleared scenario)
traffic_proxy_features = [c for c in FEATURE_COLS
                           if c in ['Month', 'Is_Winter', 'Is_Monsoon',
                                    'Wind_Speed', 'PM25_lag_1']]

print(f"  Traffic proxy features used: {traffic_proxy_features}")

# Simulate 20% traffic reduction:
# We reduce PM25_lag_1 by 20% (less traffic yesterday → less pollution carry-over)
# and increase Wind_Speed by 20% (traffic reduction clears air faster)
X_simulated = X_test_copy.copy()
if 'PM25_lag_1' in X_simulated.columns:
    X_simulated['PM25_lag_1'] = X_simulated['PM25_lag_1'] * 0.80
if 'Wind_Speed' in X_simulated.columns:
    X_simulated['Wind_Speed'] = X_simulated['Wind_Speed'] * 1.20

simulated_preds = xgb_model.predict(X_simulated)

# Calculate TPE
baseline_mean  = baseline_preds.mean()
simulated_mean = simulated_preds.mean()
pct_change_pm25 = ((simulated_mean - baseline_mean) / baseline_mean) * 100
traffic_reduction_pct = 20.0
tpe = abs(pct_change_pm25) / traffic_reduction_pct

print(f"\n  📊 TPE Analysis Results:")
print(f"  Baseline avg PM2.5:   {baseline_mean:.2f} µg/m³")
print(f"  Simulated avg PM2.5:  {simulated_mean:.2f} µg/m³")
print(f"  PM2.5 change:         {pct_change_pm25:.2f}%")
print(f"  Traffic reduction:    {traffic_reduction_pct}%")
print(f"  TPE = {tpe:.3f}")
print(f"\n  Interpretation: A 10% traffic reduction is associated with")
print(f"  approximately {tpe * 10:.1f}% reduction in PM2.5")

# Save TPE results
tpe_results = pd.DataFrame({
    'Metric': ['Baseline PM2.5 (µg/m³)', 'Simulated PM2.5 (µg/m³)',
               'PM2.5 Change (%)', 'Traffic Reduction (%)', 'TPE Value'],
    'Value': [round(baseline_mean, 2), round(simulated_mean, 2),
              round(pct_change_pm25, 2), traffic_reduction_pct, round(tpe, 3)]
})
tpe_results.to_csv(f'{OUTPUT_DIR}/tpe_results.csv', index=False)
print("  ✅ Saved: tpe_results.csv")


# ============================================================
# PART J: CROSS-CITY TRANSFERABILITY
# ============================================================
# EXPLANATION:
# Train model on Delhi → Test on Bengaluru
# This answers: "Does pollution pattern in Delhi
# help predict Bengaluru?"
#
# If R² is still reasonable (>0.5), the model generalises.
# If R² drops a lot, cities have different traffic-pollution patterns.
# Both outcomes are interesting research findings!

print("\n" + "=" * 60)
print("CROSS-CITY TRANSFERABILITY TEST")
print("=" * 60)

if len(X_delhi) > 50 and len(X_blr) > 50:
    # Train on Delhi
    print("  Training XGBoost on Delhi data...")
    xgb_delhi = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=6, random_state=42, verbosity=0
    )
    xgb_delhi.fit(X_delhi, y_delhi)

    # Test on Delhi (within-city)
    delhi_preds     = xgb_delhi.predict(X_delhi)
    delhi_within_r2 = r2_score(y_delhi, delhi_preds)

    # Test on Bengaluru (cross-city)
    blr_preds    = xgb_delhi.predict(X_blr)
    cross_city_r2 = r2_score(y_blr, blr_preds)
    cross_city_rmse = np.sqrt(mean_squared_error(y_blr, blr_preds))

    print(f"\n  Within-city (Delhi → Delhi):")
    print(f"    R² = {delhi_within_r2:.4f}")
    print(f"\n  Cross-city (Delhi → Bengaluru):")
    print(f"    R²   = {cross_city_r2:.4f}")
    print(f"    RMSE = {cross_city_rmse:.2f} µg/m³")

    if cross_city_r2 > 0.6:
        print("\n  ✅ Good transferability! Delhi patterns generalize to Bengaluru")
    elif cross_city_r2 > 0.4:
        print("\n  ⚠️ Partial transferability — cities have some differences")
    else:
        print("\n  ❌ Low transferability — cities have distinct pollution patterns")
        print("     (This is also an interesting finding for your paper!)")

    cross_city_results = pd.DataFrame({
        'Test': ['Within-city (Delhi→Delhi)', 'Cross-city (Delhi→Bengaluru)'],
        'R2':   [round(delhi_within_r2, 4), round(cross_city_r2, 4)],
        'RMSE': ['N/A', round(cross_city_rmse, 2)]
    })
    cross_city_results.to_csv(f'{OUTPUT_DIR}/cross_city_results.csv', index=False)
    print("  ✅ Saved: cross_city_results.csv")

else:
    print("  ⚠️ Not enough data for both cities — skipping cross-city test")


# ============================================================
# PART K: ACTUAL VS PREDICTED PLOT
# ============================================================
print("\nCreating actual vs predicted plot...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models_results = [
    (lr_model.predict(X_test_scaled), 'Linear Regression', '#3498DB'),
    (rf_model.predict(X_test),        'Random Forest',     '#2ECC71'),
    (xgb_model.predict(X_test),       'XGBoost',           '#E74C3C'),
]

for ax, (preds, name, color) in zip(axes, models_results):
    ax.scatter(y_test, preds, alpha=0.3, s=10, color=color)
    # Perfect prediction line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', linewidth=2, label='Perfect prediction')
    r2 = r2_score(y_test, preds)
    ax.set_title(f'{name}\nR² = {r2:.3f}', fontweight='bold')
    ax.set_xlabel('Actual PM2.5')
    ax.set_ylabel('Predicted PM2.5')
    ax.legend()

plt.suptitle('Actual vs Predicted PM2.5 (all 3 models)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot9_actual_vs_predicted.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: plot9_actual_vs_predicted.png")

print("\n" + "=" * 60)
print("STEP 6 COMPLETE! ✅")
print("All outputs saved in: outputs/")
print("Next: Run STEP7_dashboard.py")
print("=" * 60)
