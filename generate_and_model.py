# Run from repository root. Reads generator CSVs (written in repo root),
# cleans and engineers features, writes consolidated CSV to root/datasets/all.csv,
# trains two sample regression models, and saves figures to root/figures/.

# Usage:
#     python3 generate_and_model.py

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED)

# --- Paths
ROOT = Path(".").resolve()
OUT_DIR = ROOT / "predicting-population-density-and-resource-demand-in-bangladesh" / "datasets"
FIG_DIR = ROOT / "predicting-population-density-and-resource-demand-in-bangladesh" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper utilities
def safe_read(csv_name):
    p = ROOT / csv_name
    if not p.exists():
        raise FileNotFoundError(f"Expected CSV not found: {p}")
    return pd.read_csv(p)

def enforce_types(df, spec):
    for col, typ in spec.items():
        if col not in df.columns:
            continue
        if typ == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").round().fillna(0).astype(int)
        elif typ == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        elif typ == "str":
            df[col] = df[col].astype(str)
    return df

# --- Load canonical tables produced by generator
tables = {}
csv_files = [
    "population_demographics.csv",
    "population_density.csv",
    "resource_demand.csv",
    "urban_infrastructure.csv",
    "migration_trends.csv",
    "economic_indicators.csv",
    "climate_impact.csv",
    "land_use_zoning.csv",
    "transportation.csv",
    "health_sanitation.csv",
    "education.csv",
    "energy_usage.csv",
    "housing.csv",
    "technology_connectivity.csv",
    "agriculture_food_security.csv",
]

for fn in csv_files:
    tables[fn.replace(".csv", "")] = safe_read(fn)

# --- Canonical typing and quick validation (subset of DATA_DICTIONARY)
# Apply minimal canonical types for key tables
tables["population_demographics"] = enforce_types(
    tables["population_demographics"],
    {
        "district": "str", "year": "int", "population": "int",
        "area_km2": "float", "density_per_km2": "float",
        "children_percent": "float", "working_age_percent": "float",
        "elderly_percent": "float", "urban_percent": "float",
        "urban_population": "int", "rural_population": "int",
    },
)

# Ensure no negative counts/areas
for df in tables.values():
    for c in df.select_dtypes(include=[np.number]).columns:
        if "percent" in c.lower():
            # clamp percent columns to 0-100
            df[c] = df[c].clip(lower=0.0, upper=100.0)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if c in ("area_km2",):
                df[c] = df[c].clip(lower=0.0)

# --- Feature engineering helpers
def add_per_capita_columns(merged):
    merged["water_per_person_lpd"] = (merged["water_mld"] * 1e6) / merged["population"]
    merged["elec_mwh_per_person"] = merged["electricity_mwh"] / merged["population"]
    merged["food_kg_per_person_per_year"] = (merged["food_demand_tons"] * 1000.0) / merged["population"]
    merged["housing_units_per_1000"] = merged["housing_units"] / (merged["population"] / 1000.0)
    return merged

def add_density_related(merged):
    merged["pop_per_hectare"] = merged["density_per_km2"] / 10.0
    merged["urban_to_rural_ratio"] = merged["urban_population"] / merged["rural_population"].replace(0, np.nan)
    merged["urban_to_rural_ratio"] = merged["urban_to_rural_ratio"].fillna(merged["urban_percent"] / (100 - merged["urban_percent"] + 1e-6))
    return merged

def add_climate_severity(merged):
    # normalized climate severity: combine flood risk, drought index and cyclone frequency
    merged["climate_severity"] = (
        (merged["flood_risk_score"].fillna(0) / 10.0) * 0.5 +
        (merged["drought_index"].fillna(0) / 10.0) * 0.3 +
        (merged["cyclone_events"].fillna(0) / (merged["cyclone_events"].max() + 1e-6)) * 0.2
    )
    return merged

def add_temporal_lags(merged, group="district", cols_to_lag=None, lag_years=[1,2]):
    if cols_to_lag is None:
        cols_to_lag = ["population", "density_per_km2", "water_mld", "electricity_mwh"]
    merged = merged.sort_values([group, "year"])
    for c in cols_to_lag:
        for l in lag_years:
            name = f"{c}_lag{l}"
            merged[name] = merged.groupby(group)[c].shift(l)
    # rolling mean (3-year) for numeric columns
    numeric_cols = ["population", "density_per_km2", "water_mld"]
    for c in numeric_cols:
        merged[f"{c}_rolling3"] = merged.groupby(group)[c].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    return merged

# --- Merge strategy: left-join population_demographics (base) with each table on district+year
base = tables["population_demographics"].copy()
merged = base
join_keys = ["district", "year"]

for name, df in tables.items():
    if name == "population_demographics":
        continue
    # prefer suffix to prevent collisions
    merged = merged.merge(df, on=join_keys, how="left", suffixes=("", f"_{name}"))

# There can be columns duplicated across merges (e.g., schools from urban_infrastructure and education)
# Resolve duplicates by keeping the column without suffix when present, else the suffixed one.
def coalesce_columns(df):
    cols = df.columns.tolist()
    new = df.copy()
    for c in cols:
        if c.endswith("_population_demographics"):
            base_name = c.replace("_population_demographics", "")
            if base_name in new.columns:
                new.drop(columns=[c], inplace=True)
            else:
                new.rename(columns={c: base_name}, inplace=True)
    return new

merged = coalesce_columns(merged)

# --- Derived features
merged = add_per_capita_columns(merged)
merged = add_density_related(merged)
merged = add_climate_severity(merged)
merged = add_temporal_lags(merged)

# Encode district as category and integer code for models
merged["district_cat"] = merged["district"].astype("category")
merged["district_code"] = merged["district_cat"].cat.codes

# Fill remaining missing numeric values with sensible defaults (median by district-year group)
num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    if merged[c].isna().any():
        merged[c] = merged.groupby("district")[c].transform(lambda x: x.fillna(x.median()))
        merged[c] = merged[c].fillna(0.0)

# Final sanity clamps per conventions
if "urban_percent" in merged.columns:
    merged["urban_percent"] = merged["urban_percent"].clip(0.0, 100.0)
if "children_percent" in merged.columns:
    merged["children_percent"] = merged["children_percent"].clip(0.0, 100.0)
if "avg_rent_bdt" in merged.columns:
    merged["avg_rent_bdt"] = merged["avg_rent_bdt"].clip(lower=0.0)

# --- Persist consolidated dataset
OUT_ALL = OUT_DIR / "all.csv"
merged.to_csv(OUT_ALL, index=False)
print(f"Wrote consolidated dataset to: {OUT_ALL}")

# --- Quick EDA plots (timeseries for each district: density and water demand)
sns.set(style="whitegrid", context="talk")
for var in ["density_per_km2", "water_mld"]:
    plt.figure(figsize=(10, 6))
    for d in merged["district"].unique():
        sub = merged[merged["district"] == d]
        plt.plot(sub["year"], sub[var], marker="o", label=d)
    plt.xlabel("Year")
    plt.ylabel(var)
    plt.title(f"Time series of {var} by district")
    plt.legend()
    plt.tight_layout()
    fn = FIG_DIR / f"time_series_{var}.png"
    plt.savefig(fn, dpi=150)
    plt.close()
    print(f"Saved figure: {fn}")

# --- Modeling helper function
def train_and_evaluate(target, features, df, group_time="year"):
    # Use last 3 years as holdout to respect temporal ordering
    years = sorted(df["year"].unique())
    if len(years) < 6:
        test_years = years[-2:]
    else:
        test_years = years[-3:]
    train_df = df[~df["year"].isin(test_years)].copy()
    test_df = df[df["year"].isin(test_years)].copy()

    X_train = train_df[features].copy()
    y_train = train_df[target].values
    X_test = test_df[features].copy()
    y_test = test_df[target].values

    # Simple preprocessing: numeric scaling, one-hot small cardinality
    numeric_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feats = [c for c in features if c not in numeric_feats]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_feats),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Save simple scatter plot observed vs predicted
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"{target} observed vs predicted (R2={r2:.3f} MAE={mae:.2f})")
    fname = FIG_DIR / f"obs_vs_pred_{target}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

    # Feature importance extraction (works because RF is last step)
    # We need to extract feature names after preprocessing
    pre = model.named_steps["pre"]
    rf = model.named_steps["rf"]
    # build feature names
    num_names = numeric_feats
    cat_names = []
    if categorical_feats:
        ohe = pre.named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(categorical_feats))
    feature_names = num_names + cat_names
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(30)

    # Save feature importance barplot
    plt.figure(figsize=(8, min(10, len(imp_df)*0.4 + 1)))
    sns.barplot(data=imp_df, x="importance", y="feature", palette="viridis")
    plt.title(f"Top features for predicting {target}")
    plt.tight_layout()
    fn2 = FIG_DIR / f"feature_importances_{target}.png"
    plt.savefig(fn2, dpi=150)
    plt.close()

    return {"r2": r2, "mae": mae, "imp_df": imp_df, "model": model, "y_test": y_test, "y_pred": y_pred, "test_index": test_df.index}

# --- Define features for two sample tasks
# Use a mixture of demographics, economy, climate, infrastructure and engineered features
base_features = [
    "year",
    "district_code",
    "population",
    "area_km2",
    "urban_percent",
    "children_percent",
    "working_age_percent",
    "elderly_percent",
    "avg_income_bdt",
    "employment_rate_percent",
    "poverty_rate_percent",
    "flood_risk_score",
    "drought_index",
    "cyclone_events",
    "green_space_km2",
    "road_km",
    "schools",
    "hospitals",
    "internet_penetration_percent",
    "mobile_penetration_percent",
    "residential_mwh",
    "commercial_mwh",
    "industrial_mwh",
    # engineered
    "water_per_person_lpd",
    "elec_mwh_per_person",
    "food_kg_per_person_per_year",
    "pop_per_hectare",
    "urban_to_rural_ratio",
    "climate_severity",
    "population_lag1",
    "population_lag2",
    "density_per_km2_lag1",
    "density_per_km2_lag2",
]

# Filter features that exist in merged (some columns might not exist depending on generator)
features = [f for f in base_features if f in merged.columns]

# --- Task 1: Predict density_per_km2
task1_target = "density_per_km2"
print("\nTraining model for:", task1_target)
task1_result = train_and_evaluate(task1_target, features, merged)
print(f"Density model: R2={task1_result['r2']:.3f}, MAE={task1_result['mae']:.3f}")

# --- Task 2: Predict water_mld
task2_target = "water_mld"
if task2_target in merged.columns:
    print("\nTraining model for:", task2_target)
    task2_result = train_and_evaluate(task2_target, features, merged)
    print(f"Water demand model: R2={task2_result['r2']:.3f}, MAE={task2_result['mae']:.3f}")
else:
    print("water_mld not present in merged dataset; skipping water demand model.")

# --- Save sample predictions into CSV for inspection (test rows with predictions)
def save_predictions(result, target, df):
    if result is None:
        return
    test_idx = result["test_index"]
    pred_df = df.loc[test_idx, ["district", "year"]].copy()
    pred_df[f"{target}_observed"] = result["y_test"]
    pred_df[f"{target}_predicted"] = result["y_pred"]
    outp = OUT_DIR / f"predictions_{target}.csv"
    pred_df.to_csv(outp, index=False)
    print(f"Saved predictions to: {outp}")

save_predictions(task1_result, task1_target, merged)
if task2_target in merged.columns:
    save_predictions(task2_result, task2_target, merged)

print("All done. Figures in:", FIG_DIR, "Consolidated CSV in:", OUT_ALL)
