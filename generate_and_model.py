"""
generate_and_model.py

Reads generator CSVs from ./datasets, cleans and engineers features,
writes consolidated CSV to ./datasets/all.csv, trains two sample regression models,
and saves figures to ./datasets/figures/.

Usage:
    python3 generate_and_model.py
"""
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

SEED = 42
np.random.seed(SEED)

# --- Paths: read/write within ./datasets
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "datasets"
FIG_DIR = REPO_ROOT / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# --- Utility helpers
def safe_read(csv_name):
    p = DATA_DIR / csv_name
    if p.exists():
        return pd.read_csv(p)
    raise FileNotFoundError(f"Expected CSV not found: {csv_name}\nChecked: {p}")


def enforce_types(df, spec):
    for col, typ in spec.items():
        if col not in df.columns:
            continue
        if typ == "int":
            df[col] = (
                pd.to_numeric(df[col], errors="coerce").round().fillna(0).astype(int)
            )
        elif typ == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        elif typ == "str":
            df[col] = df[col].astype(str)
    return df


# --- Load tables from datasets/
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
tables = {}
for fn in csv_files:
    try:
        tables[fn.replace(".csv", "")] = safe_read(fn)
        print(f"Loaded: {fn}")
    except FileNotFoundError:
        print(f"Warning: missing CSV (skipped): {fn}")

# --- Canonical typing for population_demographics (base)
if "population_demographics" not in tables:
    raise RuntimeError("population_demographics.csv must be present in datasets/")

tables["population_demographics"] = enforce_types(
    tables["population_demographics"],
    {
        "district": "str",
        "year": "int",
        "population": "int",
        "area_km2": "float",
        "density_per_km2": "float",
        "children_percent": "float",
        "working_age_percent": "float",
        "elderly_percent": "float",
        "urban_percent": "float",
        "urban_population": "int",
        "rural_population": "int",
    },
)

# Clamp percent columns and non-negative fields across all loaded tables
for name, df in list(tables.items()):
    for c in df.select_dtypes(include=[np.number]).columns:
        if "percent" in c.lower():
            df[c] = df[c].clip(lower=0.0, upper=100.0)
        if c in ("area_km2",):
            df[c] = df[c].clip(lower=0.0)
    tables[name] = df

# --- Preprocess migration_trends into per-district inflow/outflow aggregates
if "migration_trends" in tables:
    mig = tables["migration_trends"].copy()
    # ensure expected columns
    for col in ["from_district", "to_district", "year", "migrants"]:
        if col not in mig.columns:
            raise RuntimeError("migration_trends.csv missing expected column: " + col)
    mig["year"] = pd.to_numeric(mig["year"], errors="coerce").astype(int)
    mig["migrants"] = (
        pd.to_numeric(mig["migrants"], errors="coerce").fillna(0).astype(int)
    )
    outflow = (
        mig.groupby(["from_district", "year"], as_index=False)["migrants"]
        .sum()
        .rename(columns={"from_district": "district", "migrants": "migration_outflow"})
    )
    inflow = (
        mig.groupby(["to_district", "year"], as_index=False)["migrants"]
        .sum()
        .rename(columns={"to_district": "district", "migrants": "migration_inflow"})
    )
    mig_agg = inflow.merge(outflow, on=["district", "year"], how="outer").fillna(0)
    mig_agg["migration_inflow"] = mig_agg["migration_inflow"].astype(int)
    mig_agg["migration_outflow"] = mig_agg["migration_outflow"].astype(int)
    tables["migration_trends_agg"] = mig_agg
    tables.pop("migration_trends", None)
    print(
        "Aggregated migration_trends -> migration_trends_agg (migration_inflow/outflow)"
    )

# --- Merge strategy: left-join population_demographics (base) with each table on district+year
base = tables["population_demographics"].copy()
merged = base.copy()
join_keys = ["district", "year"]

for name, df in list(tables.items()):
    if name == "population_demographics":
        continue
    # Only merge tables that contain both join keys
    if not set(join_keys).issubset(df.columns):
        print(
            f"Skipping merge for '{name}': missing keys {join_keys}; cols: {list(df.columns)[:8]}"
        )
        continue
    dfc = df.copy()
    dfc["year"] = pd.to_numeric(dfc["year"], errors="coerce").astype(int)
    dfc["district"] = dfc["district"].astype(str)
    merged = merged.merge(dfc, on=join_keys, how="left", suffixes=("", f"_{name}"))
    print(f"Merged table: {name}")


# Resolve suffixed duplicates: if base (unsuffixed) exists keep it, else rename suffixed to base
def coalesce_merge_columns(df, table_names):
    out = df.copy()
    for t in table_names:
        sfx = f"_{t}"
        for col in list(out.columns):
            if col.endswith(sfx):
                base_col = col[: -len(sfx)]
                if base_col in out.columns:
                    out.drop(columns=[col], inplace=True)
                else:
                    out.rename(columns={col: base_col}, inplace=True)
    return out


merged = coalesce_merge_columns(merged, tables.keys())


# --- Feature engineering
def add_per_capita_columns(df):
    if {"water_mld", "population"}.issubset(df.columns):
        df["water_per_person_lpd"] = (df["water_mld"].fillna(0) * 1e6) / df[
            "population"
        ].replace(0, np.nan)
    if {"electricity_mwh", "population"}.issubset(df.columns):
        df["elec_mwh_per_person"] = df["electricity_mwh"].fillna(0) / df[
            "population"
        ].replace(0, np.nan)
    if {"food_demand_tons", "population"}.issubset(df.columns):
        df["food_kg_per_person_per_year"] = (
            df["food_demand_tons"].fillna(0) * 1000.0
        ) / df["population"].replace(0, np.nan)
    if {"housing_units", "population"}.issubset(df.columns):
        df["housing_units_per_1000"] = df["housing_units"].fillna(0) / (
            df["population"].replace(0, np.nan) / 1000.0
        )
    return df


def add_density_related(df):
    if "density_per_km2" in df.columns:
        df["pop_per_hectare"] = df["density_per_km2"].fillna(0) / 10.0
    if {"urban_population", "rural_population", "urban_percent"}.issubset(df.columns):
        df["urban_to_rural_ratio"] = df["urban_population"].replace(0, np.nan) / df[
            "rural_population"
        ].replace(0, np.nan)
        df["urban_to_rural_ratio"] = df["urban_to_rural_ratio"].fillna(
            df["urban_percent"] / (100.0 - df["urban_percent"] + 1e-6)
        )
    return df


def add_climate_severity(df):
    parts = []
    if "flood_risk_score" in df.columns:
        parts.append((df["flood_risk_score"].fillna(0) / 10.0) * 0.5)
    else:
        parts.append(0)
    if "drought_index" in df.columns:
        parts.append((df["drought_index"].fillna(0) / 10.0) * 0.3)
    else:
        parts.append(0)
    if "cyclone_events" in df.columns:
        max_c = df["cyclone_events"].max(skipna=True)
        denom = (max_c + 1e-6) if not np.isnan(max_c) else 1.0
        parts.append((df["cyclone_events"].fillna(0) / denom) * 0.2)
    else:
        parts.append(0)
    df["climate_severity"] = sum(parts)
    return df


def add_temporal_lags(df, group="district", cols_to_lag=None, lag_years=[1, 2]):
    if cols_to_lag is None:
        cols_to_lag = ["population", "density_per_km2", "water_mld", "electricity_mwh"]
    df = df.sort_values([group, "year"])
    for c in cols_to_lag:
        if c not in df.columns:
            continue
        for l in lag_years:
            df[f"{c}_lag{l}"] = df.groupby(group)[c].shift(l)
    numeric_cols = [
        c for c in ["population", "density_per_km2", "water_mld"] if c in df.columns
    ]
    for c in numeric_cols:
        df[f"{c}_rolling3"] = (
            df.groupby(group)[c]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return df


merged = add_per_capita_columns(merged)
merged = add_density_related(merged)
merged = add_climate_severity(merged)
merged = add_temporal_lags(merged)

# Encode district
merged["district"] = merged["district"].astype(str)
merged["district_cat"] = merged["district"].astype("category")
merged["district_code"] = merged["district_cat"].cat.codes

# Fill missing numeric values: district median then global median fallback
num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    if merged[c].isna().any():
        merged[c] = merged.groupby("district")[c].transform(
            lambda x: x.fillna(x.median())
        )
        merged[c] = merged[c].fillna(merged[c].median())

# Final sanity clamps
if "urban_percent" in merged.columns:
    merged["urban_percent"] = merged["urban_percent"].clip(0.0, 100.0)
if "children_percent" in merged.columns:
    merged["children_percent"] = merged["children_percent"].clip(0.0, 100.0)
if "avg_rent_bdt" in merged.columns:
    merged["avg_rent_bdt"] = merged["avg_rent_bdt"].clip(lower=0.0)

# --- Persist consolidated dataset to ./datasets/all.csv
OUT_ALL = DATA_DIR / "all.csv"
merged.to_csv(OUT_ALL, index=False)
print(f"Wrote consolidated dataset to: {OUT_ALL}")

# --- Quick EDA plots in ./datasets/figures
sns.set(style="whitegrid", context="talk")
for var in ["density_per_km2", "water_mld"]:
    if var not in merged.columns:
        continue
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


# --- Modeling: helper
def train_and_evaluate(target, features, df):
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

    numeric_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feats = [c for c in features if c not in numeric_feats]

    try:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_feats),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_feats,
                ),
            ],
            remainder="drop",
        )
    except TypeError:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_feats),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse=False),
                    categorical_feats,
                ),
            ],
            remainder="drop",
        )

    model = Pipeline(
        [
            ("pre", preprocessor),
            (
                "rf",
                RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # observed vs predicted plot
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linestyle="--",
    )
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"{target} observed vs predicted (R2={r2:.3f} MAE={mae:.2f})")
    fname = FIG_DIR / f"obs_vs_pred_{target}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

    # feature importances (RF)
    pre = model.named_steps["pre"]
    rf = model.named_steps["rf"]
    num_names = numeric_feats
    cat_names = []
    if categorical_feats:
        ohe = pre.named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(categorical_feats))
    feature_names = num_names + cat_names
    importances = rf.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(30)
    )

    plt.figure(figsize=(8, min(10, len(imp_df) * 0.4 + 1)))
    sns.barplot(data=imp_df, x="importance", y="feature", palette="viridis")
    plt.title(f"Top features for predicting {target}")
    plt.tight_layout()
    fn2 = FIG_DIR / f"feature_importances_{target}.png"
    plt.savefig(fn2, dpi=150)
    plt.close()

    return {
        "r2": r2,
        "mae": mae,
        "imp_df": imp_df,
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "test_index": test_df.index,
    }


# --- Define feature set (filter to available columns)
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
features = [f for f in base_features if f in merged.columns]

# --- Task 1: predict density_per_km2
if "density_per_km2" in merged.columns:
    print("Training density_per_km2 model...")
    res1 = train_and_evaluate("density_per_km2", features, merged)
    print(f"Density model: R2={res1['r2']:.3f} MAE={res1['mae']:.3f}")
    # save predictions
    preds = merged.loc[res1["test_index"], ["district", "year"]].copy()
    preds["density_per_km2_observed"] = res1["y_test"]
    preds["density_per_km2_predicted"] = res1["y_pred"]
    preds.to_csv(DATA_DIR / "predictions_density_per_km2.csv", index=False)
    print("Saved predictions_density_per_km2.csv")
else:
    print("density_per_km2 not available; skipping density model")

# --- Task 2: predict water_mld
if "water_mld" in merged.columns:
    print("Training water_mld model...")
    res2 = train_and_evaluate("water_mld", features, merged)
    print(f"Water model: R2={res2['r2']:.3f} MAE={res2['mae']:.3f}")
    preds2 = merged.loc[res2["test_index"], ["district", "year"]].copy()
    preds2["water_mld_observed"] = res2["y_test"]
    preds2["water_mld_predicted"] = res2["y_pred"]
    preds2.to_csv(DATA_DIR / "predictions_water_mld.csv", index=False)
    print("Saved predictions_water_mld.csv")
else:
    print("water_mld not available; skipping water model")

print("Done. Consolidated CSV:", OUT_ALL, "Figures in:", FIG_DIR)
