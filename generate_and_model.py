#!/usr/bin/env python3

"""
generate_and_model.py

Hardened pipeline:

- reads CSVs from ./datasets (also accepts ./generated_csvs_v3 if present)
- aggregates replica rows into district-year observations
- cleans, imputes, feature-engineers, trains time-split RandomForest models
- writes ./datasets/all.csv, predictions and metrics, and figures in ./figures
"""

from pathlib import Path
import json
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# Directories
REPO_ROOT = Path(__file__).resolve().parent
# Prefer datasets, but allow generator's alternate outdir
DATA_DIR = REPO_ROOT / "datasets"
ALT_DATA_DIR = REPO_ROOT / "generated_csvs_v3"
if not DATA_DIR.exists() and ALT_DATA_DIR.exists():
    DATA_DIR = ALT_DATA_DIR
FIG_DIR = REPO_ROOT / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Expected CSVs (generator produces these)
csv_files = [
    "population_demographics.csv",
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


def safe_read(path: Path):
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError(f"Expected CSV not found: {path}")


def enforce_types(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    df = df.copy()
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


# Load available CSVs
tables = {}
for fn in csv_files:
    p = DATA_DIR / fn
    try:
        tables[fn.replace(".csv", "")] = safe_read(p)
        print(f"Loaded: {fn}")
    except FileNotFoundError:
        print(f"Warning: missing CSV (skipped): {fn}")

if "population_demographics" not in tables:
    raise RuntimeError("population_demographics.csv must be present in datasets/")

# Canonical typing for base table
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

# Clamp percent-like columns and non-negative fields across loaded tables
for name, df in list(tables.items()):
    for c in df.select_dtypes(include=[np.number]).columns:
        if "percent" in c.lower():
            df[c] = df[c].clip(lower=0.0, upper=100.0)
        if c in ("area_km2",):
            df[c] = df[c].clip(lower=0.0)
    tables[name] = df

# Handle migration_trends aggregation if present
if "migration_trends" in tables:
    mig = tables["migration_trends"].copy()
    required_cols = ["from_district", "to_district", "year", "migrants"]
    if not set(required_cols).issubset(mig.columns):
        raise RuntimeError("migration_trends.csv missing expected columns")
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

# ---- Replica aggregation helper


def aggregate_replicas(df: pd.DataFrame, on=("district", "year")) -> pd.DataFrame:
    """
    If df contains 'replica_id' or multiple rows per district-year, aggregate to one row per district-year.
    Rules (default):
    - 'population', counts -> sum
    - numeric continuous (income, water_mld, electricity_mwh, etc) -> median
    - percent columns -> weighted or median (we use median)
    - text/district -> first
    """
    if "replica_id" not in df.columns and not df.duplicated(subset=list(on)).any():
        return df.copy()

    agg_map = {}
    for col, dtype in df.dtypes.items():
        col = str(col)
        if col in on:
            agg_map[col] = "first"
        elif col == "replica_id":
            agg_map[col] = "first"
        elif pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
            # prefer median for stability for numeric; sum for population specifically
            if col in (
                "population",
                "migration_inflow",
                "migration_outflow",
                "hospital_beds",
                "housing_units",
            ):
                agg_map[col] = "sum"
            else:
                agg_map[col] = "median"
        else:
            agg_map[col] = "first"

    grouped = df.groupby(list(on), as_index=False).agg(agg_map)
    # ensure consistent dtypes for some important columns
    if "year" in grouped.columns:
        grouped["year"] = pd.to_numeric(grouped["year"], errors="coerce").astype(int)
    return grouped


# Aggregate all tables that have replica_id
for name in list(tables.keys()):
    df = tables[name]
    if "replica_id" in df.columns:
        # Replace the table in-place with the aggregated version (keep canonical names)
        tables[name] = aggregate_replicas(df)
        print(f"Aggregated replica-level table -> {name} (in-place)")

# For migration_trends_agg we already aggregated; keep as is.

# ---- Merge strategy (population_demographics is authoritative base)
base = tables["population_demographics"].copy()
merged = base.copy()
join_keys = ["district", "year"]

# Merge other tables with deterministic precedence: existing base columns win
for name, df in list(tables.items()):
    if name == "population_demographics":
        continue
    # Only merge if join keys exist
    if not set(join_keys).issubset(df.columns):
        print(
            f"Skipping merge for '{name}': missing keys {join_keys}; cols: {list(df.columns)[:8]}"
        )
        continue
    dfc = df.copy()
    dfc["year"] = pd.to_numeric(dfc["year"], errors="coerce").astype(int)
    # some aggregated tables may use 'district' or may have maintained it
    if "district" in dfc.columns:
        dfc["district"] = dfc["district"].astype(str)
    merged = merged.merge(dfc, on=join_keys, how="left", suffixes=("", f"_{name}"))
    print(f"Merged table: {name}")


# Resolve suffixed columns deterministically:
# If base_col exists (unsuffixed), keep it; else rename suffixed -> base_col.
def coalesce_merge_columns(df: pd.DataFrame, table_names):
    out = df.copy()
    # Iterate over columns ending with _<table>
    for col in list(out.columns):
        if "_" not in col:
            continue
        # attempt to detect suffix matching a known table
        for t in table_names:
            sfx = f"_{t}"
            if col.endswith(sfx):
                base_col = col[: -len(sfx)]
                if base_col in out.columns:
                    out.drop(columns=[col], inplace=True)
                else:
                    out.rename(columns={col: base_col}, inplace=True)
                break
    return out


merged = coalesce_merge_columns(merged, tables.keys())

# ---- Feature engineering helpers (operate on aggregated district-year rows)


def add_per_capita_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    if {"housing_units_demand", "population"}.issubset(df.columns):
        df["housing_demand_per_1000"] = df["housing_units_demand"].fillna(0) / (
            df["population"].replace(0, np.nan) / 1000.0
        )
    return df


def add_density_related(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "density_per_km2" in df.columns:
        df["pop_per_hectare"] = df["density_per_km2"].fillna(0) / 10.0
    if {"urban_population", "rural_population", "urban_percent"}.issubset(df.columns):
        # avoid 0/0 by using percent fallback
        df["urban_to_rural_ratio"] = df["urban_population"].replace(0, np.nan) / df[
            "rural_population"
        ].replace(0, np.nan)
        df["urban_to_rural_ratio"] = df["urban_to_rural_ratio"].fillna(
            df["urban_percent"] / (100.0 - df["urban_percent"] + 1e-6)
        )
    return df


def add_climate_severity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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


def add_temporal_lags(
    df: pd.DataFrame, group="district", cols_to_lag=None, lag_years=(1, 2)
) -> pd.DataFrame:
    df = df.copy()
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


# Apply feature engineering (after aggregation)
merged = add_per_capita_columns(merged)
merged = add_density_related(merged)
merged = add_climate_severity(merged)
merged = add_temporal_lags(merged)

# ---- Encode district and prepare imputation
merged["district"] = merged["district"].astype(str)
merged["district_cat"] = merged["district"].astype("category")
merged["district_code"] = merged["district_cat"].cat.codes

# Imputation: district median, then global median. Also add _imputed indicator where imputation occurred.
num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    if merged[c].isna().any():
        # compute district medians
        med_by_district = merged.groupby("district")[c].median()
        merged[c + "_imputed"] = merged[c].isna().astype(int)
        # fill with district median where available
        merged[c] = merged.apply(
            lambda row: (
                med_by_district[row["district"]]
                if pd.isna(row[c]) and not pd.isna(med_by_district[row["district"]])
                else row[c]
            ),
            axis=1,
        )
        # global median fallback
        if merged[c].isna().any():
            glob_med = merged[c].median()
            merged[c] = merged[c].fillna(glob_med)

# Final sanity clamps
if "urban_percent" in merged.columns:
    merged["urban_percent"] = merged["urban_percent"].clip(0.0, 100.0)
if "children_percent" in merged.columns:
    merged["children_percent"] = merged["children_percent"].clip(0.0, 100.0)
if "avg_rent_bdt" in merged.columns:
    merged["avg_rent_bdt"] = merged["avg_rent_bdt"].clip(lower=0.0)

# ---- Safer proxy target creation
merged = merged.copy()
# Clothing proxy: base 8 kg/year, robust scaling of income (log1p) and bounded poverty effect
if "clothing_kg_per_person_per_year" not in merged.columns:
    if ("avg_income_bdt" in merged.columns) or (
        "poverty_rate_percent" in merged.columns
    ):
        if "avg_income_bdt" in merged.columns:
            income = merged["avg_income_bdt"].clip(lower=1.0)
            # use log-scale, then min-max to [0,1]
            income_log = np.log1p(income)
            min_ln, max_ln = income_log.min(), income_log.max()
            income_norm = (income_log - min_ln) / (max_ln - min_ln + 1e-9)
        else:
            income_norm = pd.Series(0.0, index=merged.index)
        poverty_norm = (
            ((100.0 - merged["poverty_rate_percent"]) / 100.0).clip(0.0, 1.0)
            if "poverty_rate_percent" in merged.columns
            else 1.0
        )
        # bounded multiplicative effect
        merged["clothing_kg_per_person_per_year"] = (
            8.0 * (1.0 + 0.6 * income_norm) * (0.8 + 0.4 * poverty_norm)
        )
        merged["clothing_kg_per_person_per_year"] = merged[
            "clothing_kg_per_person_per_year"
        ].clip(lower=4.0, upper=30.0)
    else:
        merged["clothing_kg_per_person_per_year"] = 8.0

# Shelter target: prefer housing_demand_per_1000, else derive sensible proxy with clipping
if "housing_demand_per_1000" in merged.columns:
    merged["shelter_units_needed_per_1000"] = merged["housing_demand_per_1000"]
elif "housing_units_per_1000" in merged.columns:
    # treat as capacity; assume 10% turnover demand, but clip
    merged["shelter_units_needed_per_1000"] = (
        merged["housing_units_per_1000"] * 0.10
    ).clip(lower=0.0)
elif ("housing_units_demand" in merged.columns) and ("population" in merged.columns):
    merged["shelter_units_needed_per_1000"] = (
        merged["housing_units_demand"] / (merged["population"] / 1000.0)
    ).replace([np.inf, -np.inf], np.nan)
    merged["shelter_units_needed_per_1000"] = merged[
        "shelter_units_needed_per_1000"
    ].fillna(0.0)
else:
    merged["shelter_units_needed_per_1000"] = np.nan

# Healthcare need index: avoid 1/x blowups, clip beds_per_1000, combine with sanitation robustly
if ("hospital_beds" in merged.columns) and ("population" in merged.columns):
    merged["beds_per_1000"] = (
        merged["hospital_beds"] / (merged["population"] / 1000.0)
    ).replace([np.inf, -np.inf], np.nan)
else:
    merged["beds_per_1000"] = np.nan

sanitation_norm = (
    (merged["sanitation_coverage_percent"] / 100.0).clip(0.0, 1.0)
    if "sanitation_coverage_percent" in merged.columns
    else pd.Series(0.5, index=merged.index)
)

# Use clipped beds and a bounded function for need: need ~ (beds_threshold - beds)_pos / (beds_threshold) + (1 - sanitation)
beds_threshold = 2.5  # configurable reasonable benchmark
beds = merged["beds_per_1000"].fillna(0.0)
beds_deficit = (beds_threshold - beds).clip(lower=0.0) / beds_threshold
merged["healthcare_need_index"] = 0.65 * beds_deficit + 0.35 * (1.0 - sanitation_norm)
merged["healthcare_need_index"] = merged["healthcare_need_index"].clip(
    lower=0.0, upper=10.0
)

# Persist consolidated dataset
OUT_ALL = DATA_DIR / "all.csv"
merged.to_csv(OUT_ALL, index=False)
print(f"Wrote consolidated dataset to: {OUT_ALL}")

# ---- Quick EDA plots
sns.set(style="whitegrid", context="talk")
for var in ["density_per_km2", "water_per_person_lpd", "food_kg_per_person_per_year"]:
    if var not in merged.columns:
        continue
    plt.figure(figsize=(10, 6))
    for d in merged["district"].unique():
        sub = merged[merged["district"] == d].sort_values("year")
        if sub.empty:
            continue
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


# ---- Modeling helper (time-based split, small-dataset safe)
def _onehot_compat(**kwargs):
    # compatibility wrapper for OneHotEncoder across sklearn versions
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        # older sklearn uses sparse instead of sparse_output
        if "sparse_output" in kwargs:
            k = dict(kwargs)
            kwargs.pop("sparse_output")
            kwargs["sparse"] = False
        return OneHotEncoder(**kwargs)


def train_and_evaluate(target: str, features: list, df: pd.DataFrame):
    years = sorted(df["year"].unique())
    if len(years) < 6:
        test_years = years[-2:]
    else:
        test_years = years[-3:]

    train_df = df[~df["year"].isin(test_years)].copy()
    test_df = df[df["year"].isin(test_years)].copy()

    # require non-null target rows
    train_df = train_df.dropna(subset=[target])
    test_df = test_df.dropna(subset=[target])

    if train_df.shape[0] < 2 or test_df.shape[0] < 1:
        raise RuntimeError(
            "Not enough rows after selecting target/time-split for training/testing"
        )

    X_train = train_df[features].copy()
    y_train = train_df[target].values
    X_test = test_df[features].copy()
    y_test = test_df[target].values

    numeric_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feats = [c for c in features if c not in numeric_feats]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_feats),
            (
                "cat",
                _onehot_compat(handle_unknown="ignore", sparse_output=False),
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

    # Impute missing features for modeling: use column median (already imputed earlier), but ensure no NaN
    X_train = X_train.fillna(X_train.median(axis=0))
    X_test = X_test.fillna(X_train.median(axis=0))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Observed vs predicted plot
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], color="red", linestyle="--")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"{target} observed vs predicted (R2={r2:.3f} MAE={mae:.2f})")
    fname = FIG_DIR / f"obs_vs_pred_{target}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

    # Feature importances (RF)
    pre = model.named_steps["pre"]
    rf = model.named_steps["rf"]
    num_names = numeric_feats
    cat_names = []
    if categorical_feats:
        ohe = pre.named_transformers_["cat"]
        try:
            cat_names = list(ohe.get_feature_names_out(categorical_feats))
        except Exception:
            # sklearn older versions: get_feature_names
            cat_names = list(ohe.get_feature_names(categorical_feats))
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
        "r2": float(r2),
        "mae": float(mae),
        "imp_df": imp_df,
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "test_index": test_df.index,
    }


# ---- Feature selection and modeling targets
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
targets = {
    "water_per_person_lpd": "water_per_person_lpd",
    "food_kg_per_person_per_year": "food_kg_per_person_per_year",
    "shelter_units_needed_per_1000": "shelter_units_needed_per_1000",
    "clothing_kg_per_person_per_year": "clothing_kg_per_person_per_year",
    "healthcare_need_index": "healthcare_need_index",
}

results_summary = {}
for key, target in targets.items():
    if target not in merged.columns:
        print(f"Skipping target {target}: column not present")
        continue
    # require non-null target rows
    df_target = merged[[*features, target, "district", "year"]].dropna(subset=[target])
    if df_target.shape[0] < 10:
        print(f"Skipping target {target}: not enough rows ({df_target.shape[0]})")
        continue
    print(f"Training model for target: {target} using {len(features)} features")
    try:
        res = train_and_evaluate(target, features, merged)
    except Exception as e:
        print(f"Failed training for {target}: {e}")
        continue
    results_summary[target] = {"r2": res["r2"], "mae": res["mae"]}

    # Save predictions for the test set
    preds = merged.loc[res["test_index"], ["district", "year"]].copy()
    preds[f"{target}_observed"] = res["y_test"]
    preds[f"{target}_predicted"] = res["y_pred"]
    preds_fn = DATA_DIR / f"predictions_{target}.csv"
    preds.to_csv(preds_fn, index=False)
    print(f"Saved {preds_fn.name} R2={res['r2']:.3f} MAE={res['mae']:.3f}")

# Persist metrics summary
metrics_fn = DATA_DIR / "model_metrics_summary.json"
with open(metrics_fn, "w") as fh:
    json.dump(results_summary, fh, indent=2)
print(f"Wrote model metrics summary to: {metrics_fn}")

print("Done. Consolidated CSV:", OUT_ALL, "Figures in:", FIG_DIR)
