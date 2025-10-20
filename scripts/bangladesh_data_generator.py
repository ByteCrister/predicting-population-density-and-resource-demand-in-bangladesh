"""
predicting-population-density-and-resource-demand-in-bangladesh/scripts/bangladesh_data_generator.py

Complete, modular, deterministic synthetic data generator that produces
multiple CSVs with configurable temporal granularity and replication.
Runs with Python 3.8+.

Features
- Yearly or monthly rows (freq "Y" or "M")
- Replication factor to increase rows per district-period
- Deterministic RNG seed
- Modular functions for each domain (all domains implemented)
- Tolerant density validation with logged warnings
- CLI via argparse to configure seed, freq, replicate, out-dir
- Stable numeric types and explicit casting before CSV export
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

# ------------------------------
# Default configuration (editable via CLI)
# ------------------------------
DEFAULT_DISTRICTS = ["Dhaka", "Sylhet", "Chattogram", "Rajshahi", "Khulna"]
DEFAULT_YEARS = list(range(2010, 2026))
DEFAULT_FREQ = "Y"  # "Y" for yearly, "M" for monthly
DEFAULT_REPLICATE = 3
DEFAULT_SEED = 42
DEFAULT_OUT_DIR = Path("datasets")
DENSITY_TOLERANCE = 1e-2  # tolerance in people/km2

# Base parameters (kept as floats or ints explicitly)
DIST_AREA: Dict[str, float] = {
    "Dhaka": 1460.0,
    "Sylhet": 3490.0,
    "Chattogram": 5283.0,
    "Rajshahi": 2400.0,
    "Khulna": 4395.0,
}
DIST_POP_2010: Dict[str, int] = {
    "Dhaka": 12_000_000,
    "Sylhet": 4_000_000,
    "Chattogram": 7_000_000,
    "Rajshahi": 5_000_000,
    "Khulna": 4_500_000,
}
DIST_POP_GROWTH: Dict[str, float] = {
    "Dhaka": 0.025,
    "Sylhet": 0.018,
    "Chattogram": 0.02,
    "Rajshahi": 0.015,
    "Khulna": 0.014,
}
DIST_URBAN_SHARE_2010: Dict[str, float] = {
    "Dhaka": 0.85,
    "Sylhet": 0.45,
    "Chattogram": 0.65,
    "Rajshahi": 0.55,
    "Khulna": 0.6,
}
DIST_URBAN_TREND: Dict[str, float] = {
    "Dhaka": 0.003,
    "Sylhet": 0.004,
    "Chattogram": 0.003,
    "Rajshahi": 0.0025,
    "Khulna": 0.0025,
}
DIST_AVG_INCOME_2010: Dict[str, float] = {
    "Dhaka": 200_000.0,
    "Sylhet": 120_000.0,
    "Chattogram": 160_000.0,
    "Rajshahi": 110_000.0,
    "Khulna": 115_000.0,
}
DIST_INCOME_GROWTH: Dict[str, float] = {
    "Dhaka": 0.05,
    "Sylhet": 0.035,
    "Chattogram": 0.045,
    "Rajshahi": 0.032,
    "Khulna": 0.034,
}
DIST_RAINFALL_BASE: Dict[str, float] = {
    "Dhaka": 2200.0,
    "Sylhet": 3500.0,
    "Chattogram": 2800.0,
    "Rajshahi": 1500.0,
    "Khulna": 1800.0,
}
DIST_TEMP_BASE: Dict[str, float] = {
    "Dhaka": 28.0,
    "Sylhet": 26.5,
    "Chattogram": 27.5,
    "Rajshahi": 29.0,
    "Khulna": 28.5,
}
DIST_FLOOD_RISK_BASE: Dict[str, float] = {
    "Dhaka": 7.0,
    "Sylhet": 6.0,
    "Chattogram": 5.0,
    "Rajshahi": 3.0,
    "Khulna": 4.0,
}
SECTOR_SHARES_2010: Dict[str, tuple] = {
    "Dhaka": (0.05, 0.35, 0.60),
    "Sylhet": (0.25, 0.25, 0.50),
    "Chattogram": (0.12, 0.38, 0.50),
    "Rajshahi": (0.30, 0.25, 0.45),
    "Khulna": (0.22, 0.33, 0.45),
}

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("bd_data_generator")


# ------------------------------
# Utilities
# ------------------------------
def clamp(x: Any, low: float, high: float) -> float:
    val = float(np.asarray(x).item())
    if val < low:
        return float(low)
    if val > high:
        return float(high)
    return float(val)


def year_idx(year: int, years: List[int]) -> int:
    return int(year - years[0])


def growth(base: float, rate: float, t: int) -> float:
    return float(base * ((1.0 + rate) ** t))


def drift(value: float, annual_change: float, t: int) -> float:
    return float(value + annual_change * t)


def pct_noise(rng: np.random.RandomState, scale: float = 0.02) -> float:
    return float(1.0 + rng.normal(0.0, scale))


def add_noise(rng: np.random.RandomState, x: float, scale: float = 0.02) -> float:
    return float(x + rng.normal(0.0, abs(scale * x)))


def validate_density_and_log(
    district: str, year: int, population: int, area_km2: float, density_reported: float
):
    expected = round(float(population) / float(area_km2), 2)
    if abs(expected - float(density_reported)) > DENSITY_TOLERANCE:
        logger.warning(
            "density mismatch %s %d: expected %.2f got %.2f",
            district,
            year,
            expected,
            float(density_reported),
        )


def ensure_out_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def make_period_index(years: List[int], freq: str = "Y") -> List[pd.Timestamp]:
    if freq == "M":
        start = pd.Timestamp(years[0], 1, 1)
        end = pd.Timestamp(years[-1], 12, 31)
        return list(pd.date_range(start, end, freq="MS"))
    return [pd.Timestamp(y, 1, 1) for y in years]


# ------------------------------
# Domain generators (return DataFrame)
# ------------------------------
def gen_population_demographics(
    rng: np.random.RandomState,
    districts: List[str],
    periods: Iterable[pd.Timestamp],
    replicate: int,
    years: List[int],
) -> pd.DataFrame:
    rows = []
    for period in periods:
        year = int(period.year)
        t = year_idx(year, years)
        for d in districts:
            for r in range(replicate):
                base_pop = DIST_POP_2010[d]
                pop_val = int(
                    round(add_noise(rng, growth(base_pop, DIST_POP_GROWTH[d], t), 0.03))
                )
                area = float(DIST_AREA[d])
                density = round(float(pop_val) / area, 2)
                urban_share = clamp(
                    drift(DIST_URBAN_SHARE_2010[d], DIST_URBAN_TREND[d], t), 0.2, 0.98
                )
                urban_pop = int(round(pop_val * urban_share))
                rural_pop = int(pop_val - urban_pop)
                children_pct = clamp(
                    0.28 - 0.03 * urban_share + rng.normal(0, 0.01), 0.18, 0.35
                )
                elderly_pct = clamp(
                    0.07 + 0.01 * (1 - urban_share) + rng.normal(0, 0.005), 0.05, 0.12
                )
                working_pct = clamp(1.0 - children_pct - elderly_pct, 0.55, 0.75)
                rows.append(
                    {
                        "district": d,
                        "period_start": period,
                        "year": year,
                        "population": int(pop_val),
                        "area_km2": float(area),
                        "density_per_km2": float(density),
                        "children_percent": round(children_pct * 100.0, 2),
                        "working_age_percent": round(working_pct * 100.0, 2),
                        "elderly_percent": round(elderly_pct * 100.0, 2),
                        "urban_percent": round(urban_share * 100.0, 2),
                        "urban_population": int(urban_pop),
                        "rural_population": int(rural_pop),
                        "replica_id": int(r),
                    }
                )
    df = pd.DataFrame(rows)
    # Validate densities (log mismatch)
    for _, row in df.iterrows():
        validate_density_and_log(
            row["district"],
            int(row["year"]),
            int(row["population"]),
            float(row["area_km2"]),
            float(row["density_per_km2"]),
        )
    df = df.astype(
        {
            "district": str,
            "year": int,
            "population": int,
            "area_km2": float,
            "density_per_km2": float,
            "children_percent": float,
            "working_age_percent": float,
            "elderly_percent": float,
            "urban_percent": float,
            "urban_population": int,
            "rural_population": int,
            "replica_id": int,
        }
    )
    return df


def gen_resource_demand(
    rng: np.random.RandomState, pop_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    grouped = pop_df.groupby(["district", "period_start", "year", "replica_id"])
    for (d, period, year, replica), g in grouped:
        pop = int(g["population"].iloc[0])
        income = float(
            add_noise(
                rng,
                growth(
                    DIST_AVG_INCOME_2010[d],
                    DIST_INCOME_GROWTH[d],
                    year - pop_df["year"].min(),
                ),
                0.05,
            )
        )
        water_mld = clamp((pop / 1000.0) * 0.15 * pct_noise(rng, 0.08), 50.0, 5000.0)
        urban_pct = float(g["urban_percent"].iloc[0]) / 100.0
        elec_mwh = clamp(
            (pop * (0.25 + 0.5 * urban_pct) * (income / 150000.0))
            * 0.05
            * pct_noise(rng, 0.1),
            1000.0,
            300000.0,
        )
        households = pop / 4.5
        housing_units_demand = int(
            round(clamp(households * 0.03 * pct_noise(rng, 0.1), 1000.0, 600000.0))
        )
        food_tons = int(
            round(clamp(pop * 0.55 * pct_noise(rng, 0.05), 50_000.0, 15_000_000.0))
        )
        rows.append(
            {
                "district": d,
                "period_start": period,
                "year": int(year),
                "replica_id": int(replica),
                "water_mld": round(float(water_mld), 2),
                "electricity_mwh": int(round(elec_mwh)),
                "housing_units_demand": int(housing_units_demand),
                "food_demand_tons": int(food_tons),
            }
        )
    return pd.DataFrame(rows)


def gen_urban_infrastructure(
    rng: np.random.RandomState, pop_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    base_roads = {
        "Dhaka": 2800.0,
        "Sylhet": 1500.0,
        "Chattogram": 2200.0,
        "Rajshahi": 1800.0,
        "Khulna": 1900.0,
    }
    grouped = pop_df.groupby(["district", "period_start", "year", "replica_id"])
    for (d, period, year, replica), g in grouped:
        t = year_idx(int(year), list(pop_df["year"].unique()))
        pop = int(g["population"].iloc[0])
        base_road = base_roads[d]
        road_km = int(
            round(
                clamp(
                    add_noise(
                        rng, base_road * ((1.0 + 0.015) ** t) + (pop / 1e5), 0.05
                    ),
                    500.0,
                    10000.0,
                )
            )
        )
        schools = int(round(clamp(add_noise(rng, pop / 4000.0, 0.15), 200.0, 8000.0)))
        hospitals = int(round(clamp(add_noise(rng, pop / 80000.0, 0.2), 10.0, 600.0)))
        green_km2 = round(
            clamp(
                add_noise(
                    rng,
                    float(DIST_AREA[d])
                    * (0.08 + 0.03 * (1 - (pop / (DIST_AREA[d] * 25000.0)))),
                    0.1,
                ),
                5.0,
                800.0,
            ),
            2,
        )
        rows.append(
            {
                "district": d,
                "period_start": period,
                "year": int(year),
                "replica_id": int(replica),
                "road_km": road_km,
                "schools": schools,
                "hospitals": hospitals,
                "green_space_km2": green_km2,
            }
        )
    return pd.DataFrame(rows)


def gen_migration_trends(
    rng: np.random.RandomState, pop_df: pd.DataFrame, districts: List[str]
) -> pd.DataFrame:
    rows = []
    periods = sorted(pop_df["period_start"].unique())
    for period in periods:
        t = year_idx(int(period.year), list(pop_df["year"].unique()))
        for d1 in districts:
            origin_candidates = pop_df[
                (pop_df["district"] == d1) & (pop_df["period_start"] == period)
            ]
            if origin_candidates.empty:
                continue
            origin_pop = int(round(origin_candidates["population"].mean()))
            for d2 in districts:
                if d1 == d2:
                    continue
                inc2 = add_noise(
                    rng,
                    growth(DIST_AVG_INCOME_2010[d2], DIST_INCOME_GROWTH[d2], t),
                    0.05,
                )
                inc1 = add_noise(
                    rng,
                    growth(DIST_AVG_INCOME_2010[d1], DIST_INCOME_GROWTH[d1], t),
                    0.05,
                )
                urb2 = clamp(
                    drift(DIST_URBAN_SHARE_2010[d2], DIST_URBAN_TREND[d2], t), 0.2, 0.98
                )
                urb1 = clamp(
                    drift(DIST_URBAN_SHARE_2010[d1], DIST_URBAN_TREND[d1], t), 0.2, 0.98
                )
                attractiveness = clamp(
                    (inc2 - inc1) / 50000.0
                    + (urb2 - urb1) * 0.8
                    + rng.normal(0.0, 0.1),
                    -1.5,
                    1.5,
                )
                base_flow = 500.0 + 3000.0 * max(0.0, attractiveness)
                migrants = int(
                    round(
                        clamp(
                            add_noise(rng, base_flow + origin_pop / 2e5, 0.3),
                            100.0,
                            80000.0,
                        )
                    )
                )
                rows.append(
                    {
                        "from_district": d1,
                        "to_district": d2,
                        "period_start": period,
                        "year": int(period.year),
                        "migrants": int(migrants),
                    }
                )
    return pd.DataFrame(rows)


def gen_economic_indicators(
    rng: np.random.RandomState,
    districts: List[str],
    periods: Iterable[pd.Timestamp],
    replicate: int,
    years: List[int],
) -> pd.DataFrame:
    rows = []
    for period in periods:
        year = int(period.year)
        t = year_idx(year, years)
        for d in districts:
            ag0, ind0, serv0 = SECTOR_SHARES_2010[d]
            for r in range(replicate):
                avg_income = int(
                    round(
                        clamp(
                            add_noise(
                                rng,
                                growth(
                                    DIST_AVG_INCOME_2010[d], DIST_INCOME_GROWTH[d], t
                                ),
                                0.05,
                            ),
                            60_000.0,
                            600_000.0,
                        )
                    )
                )
                employment_rate = clamp(
                    0.5 + 0.25 * (avg_income / 250_000.0) + rng.normal(0.0, 0.03),
                    0.45,
                    0.96,
                )
                poverty_rate = clamp(
                    0.35 - 0.2 * (avg_income / 250_000.0) + rng.normal(0.0, 0.02),
                    0.05,
                    0.45,
                )
                drift_services = serv0 + 0.01 * t + rng.normal(0.0, 0.01)
                drift_industry = ind0 + 0.003 * t + rng.normal(0.0, 0.008)
                drift_agri = ag0 - 0.013 * t + rng.normal(0.0, 0.008)
                total = drift_agri + drift_industry + drift_services
                agri_share = clamp(drift_agri / total, 0.05, 0.5)
                industry_share = clamp(drift_industry / total, 0.15, 0.5)
                services_share = clamp(1.0 - agri_share - industry_share, 0.3, 0.8)
                rows.append(
                    {
                        "district": d,
                        "period_start": period,
                        "year": year,
                        "replica_id": int(r),
                        "avg_income_bdt": int(avg_income),
                        "employment_rate_percent": round(employment_rate * 100.0, 2),
                        "poverty_rate_percent": round(poverty_rate * 100.0, 2),
                        "agri_share_percent": round(agri_share * 100.0, 2),
                        "industry_share_percent": round(industry_share * 100.0, 2),
                        "services_share_percent": round(services_share * 100.0, 2),
                    }
                )
    return pd.DataFrame(rows)


def gen_climate_impact(
    rng: np.random.RandomState,
    districts: List[str],
    periods: Iterable[pd.Timestamp],
    replicate: int,
    years: List[int],
) -> pd.DataFrame:
    rows = []
    for period in periods:
        year = int(period.year)
        t = year_idx(year, years)
        for d in districts:
            for r in range(replicate):
                avg_temp = round(
                    clamp(
                        add_noise(rng, DIST_TEMP_BASE[d] + 0.02 * t, 0.02), 22.0, 35.0
                    ),
                    2,
                )
                rainfall = int(
                    round(
                        clamp(
                            add_noise(
                                rng,
                                DIST_RAINFALL_BASE[d] + 10.0 * math.sin(t / 2.0),
                                0.08,
                            ),
                            800.0,
                            4500.0,
                        )
                    )
                )
                flood_risk = int(
                    round(
                        clamp(
                            add_noise(
                                rng,
                                DIST_FLOOD_RISK_BASE[d] + 0.1 * (rainfall / 1000.0),
                                0.2,
                            ),
                            1.0,
                            10.0,
                        )
                    )
                )
                lam = (
                    (rainfall - 1500.0) / 1200.0
                    if d in ["Chattogram", "Khulna", "Sylhet"]
                    else 0.5
                )
                cyclones = int(
                    round(
                        clamp(
                            rng.poisson(max(0.0, lam)) + rng.normal(0.0, 0.3), 0.0, 8.0
                        )
                    )
                )
                drought_index = round(
                    clamp(add_noise(rng, (2500.0 - rainfall) / 300.0, 0.2), 0.0, 10.0),
                    2,
                )
                rows.append(
                    {
                        "district": d,
                        "period_start": period,
                        "year": year,
                        "replica_id": int(r),
                        "avg_temp_c": avg_temp,
                        "rainfall_mm": int(rainfall),
                        "flood_risk_score": int(flood_risk),
                        "cyclone_events": int(cyclones),
                        "drought_index": float(drought_index),
                    }
                )
    return pd.DataFrame(rows)


def gen_land_use_zoning(
    rng: np.random.RandomState,
    districts: List[str],
    periods: Iterable[pd.Timestamp],
    replicate: int,
) -> pd.DataFrame:
    rows = []
    for period in periods:
        year = int(period.year)
        for d in districts:
            area = float(DIST_AREA[d])
            for r in range(replicate):
                # urban percent we do not have here; create a proxy using urban trend
                urban_pct = clamp(
                    drift(
                        DIST_URBAN_SHARE_2010[d],
                        DIST_URBAN_TREND[d],
                        year - DEFAULT_YEARS[0],
                    ),
                    0.2,
                    0.98,
                )
                residential = clamp(
                    add_noise(rng, area * (0.15 + 0.25 * urban_pct), 0.06),
                    50.0,
                    area * 0.7,
                )
                commercial = clamp(
                    add_noise(rng, area * (0.05 + 0.12 * urban_pct), 0.08),
                    10.0,
                    area * 0.3,
                )
                industrial = clamp(
                    add_noise(rng, area * (0.04 + 0.1 * urban_pct), 0.08),
                    5.0,
                    area * 0.25,
                )
                agricultural = clamp(
                    area - residential - commercial - industrial,
                    area * 0.1,
                    area * 0.85,
                )
                rows.append(
                    {
                        "district": d,
                        "period_start": period,
                        "year": year,
                        "replica_id": int(r),
                        "residential_area_km2": round(float(residential), 2),
                        "commercial_area_km2": round(float(commercial), 2),
                        "industrial_area_km2": round(float(industrial), 2),
                        "agricultural_area_km2": round(float(agricultural), 2),
                    }
                )
    return pd.DataFrame(rows)


def gen_transportation(
    rng: np.random.RandomState, pop_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    grouped = pop_df.groupby(["district", "period_start", "year", "replica_id"])
    for (d, period, year, replica), g in grouped:
        t = year_idx(int(year), list(pop_df["year"].unique()))
        urban_pop = int(g["urban_population"].iloc[0])
        bus_routes = int(
            round(clamp(add_noise(rng, urban_pop / 80000.0, 0.2), 20.0, 2000.0))
        )
        rail_base = 50 + 2 * t + {"Dhaka": 30, "Chattogram": 20}.get(d, 5)
        rail_km = int(round(clamp(add_noise(rng, rail_base, 0.1), 20.0, 800.0)))
        density = float(g["density_per_km2"].iloc[0])
        avg_commute = int(
            round(
                clamp(add_noise(rng, 25.0 + 0.01 * density - 0.1 * t, 0.1), 15.0, 90.0)
            )
        )
        vehicle_count = int(
            round(
                clamp(
                    add_noise(rng, urban_pop * (0.18 + 0.4), 0.15), 20_000.0, 900_000.0
                )
            )
        )
        rows.append(
            {
                "district": d,
                "period_start": period,
                "year": int(year),
                "replica_id": int(replica),
                "bus_routes": int(bus_routes),
                "rail_km": int(rail_km),
                "avg_commute_time_min": int(avg_commute),
                "vehicle_count": int(vehicle_count),
            }
        )
    return pd.DataFrame(rows)


def gen_health_sanitation(
    rng: np.random.RandomState, pop_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    grouped = pop_df.groupby(["district", "period_start", "year", "replica_id"])
    for (d, period, year, replica), g in grouped:
        t = year_idx(int(year), list(pop_df["year"].unique()))
        pop = int(g["population"].iloc[0])
        hospital_beds = int(
            round(clamp(add_noise(rng, pop / 700.0, 0.2), 300.0, 30000.0))
        )
        clinics = int(round(clamp(add_noise(rng, pop / 30000.0, 0.25), 30.0, 1500.0)))
        urban_pct = float(g["urban_percent"].iloc[0]) / 100.0
        sanitation_coverage = clamp(
            add_noise(rng, 0.55 + 0.01 * t + 0.1 * urban_pct, 0.05), 0.4, 0.99
        )
        waste_collection = clamp(
            add_noise(rng, 0.45 + 0.012 * t + 0.12 * urban_pct, 0.06), 0.3, 0.98
        )
        rows.append(
            {
                "district": d,
                "period_start": period,
                "year": int(year),
                "replica_id": int(replica),
                "hospital_beds": int(hospital_beds),
                "clinics": int(clinics),
                "sanitation_coverage_percent": round(sanitation_coverage * 100.0, 2),
                "waste_collection_percent": round(waste_collection * 100.0, 2),
            }
        )
    return pd.DataFrame(rows)


def gen_education(
    rng: np.random.RandomState, pop_df: pd.DataFrame, eco_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    grouped = pop_df.groupby(["district", "period_start", "year", "replica_id"])
    for (d, period, year, replica), g in grouped:
        t = year_idx(int(year), list(pop_df["year"].unique()))
        pop = int(g["population"].iloc[0])
        schools = int(round(clamp(add_noise(rng, pop / 3500.0, 0.15), 300.0, 9000.0)))
        income = int(
            eco_df[
                (eco_df["district"] == d)
                & (eco_df["year"] == int(year))
                & (eco_df["replica_id"] == int(replica))
            ]["avg_income_bdt"].iloc[0]
        )
        literacy = clamp(
            add_noise(rng, 0.55 + 0.12 * (income / 250_000.0) + 0.01 * t, 0.03),
            0.5,
            0.98,
        )
        student_teacher_ratio = round(
            clamp(
                add_noise(rng, 40.0 - 0.6 * t + rng.normal(0.0, 1.5), 0.05), 15.0, 55.0
            ),
            1,
        )
        rows.append(
            {
                "district": d,
                "period_start": period,
                "year": int(year),
                "replica_id": int(replica),
                "schools": int(schools),
                "literacy_rate_percent": round(literacy * 100.0, 2),
                "student_teacher_ratio": float(student_teacher_ratio),
            }
        )
    return pd.DataFrame(rows)


def gen_energy_usage(
    rng: np.random.RandomState, pop_df: pd.DataFrame, eco_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    grouped = pop_df.groupby(["district", "period_start", "year", "replica_id"])
    for (d, period, year, replica), g in grouped:
        pop = int(g["population"].iloc[0])
        urban_pct = float(g["urban_percent"].iloc[0]) / 100.0
        income = int(
            eco_df[
                (eco_df["district"] == d)
                & (eco_df["year"] == int(year))
                & (eco_df["replica_id"] == int(replica))
            ]["avg_income_bdt"].iloc[0]
        )
        residential_mwh = int(
            round(
                clamp(
                    add_noise(
                        rng, pop * (0.03 + 0.06 * urban_pct) * (income / 180000.0), 0.12
                    ),
                    500.0,
                    250000.0,
                )
            )
        )
        commercial_mwh = int(
            round(
                clamp(
                    add_noise(
                        rng, pop * (0.01 + 0.05 * urban_pct) * (income / 200000.0), 0.15
                    ),
                    1000.0,
                    300000.0,
                )
            )
        )
        industrial_mwh = int(
            round(
                clamp(
                    add_noise(
                        rng,
                        pop * (0.02 + 0.04 * (1 - urban_pct)) * (income / 220000.0),
                        0.18,
                    ),
                    2000.0,
                    400000.0,
                )
            )
        )
        renewable_percent = round(
            clamp(
                add_noise(
                    rng,
                    0.08
                    + 0.01 * year_idx(int(year), list(pop_df["year"].unique()))
                    + 0.05 * (1 - urban_pct),
                    0.05,
                ),
                0.05,
                0.55,
            )
            * 100.0,
            2,
        )
        rows.append(
            {
                "district": d,
                "period_start": period,
                "year": int(year),
                "replica_id": int(replica),
                "residential_mwh": int(residential_mwh),
                "commercial_mwh": int(commercial_mwh),
                "industrial_mwh": int(industrial_mwh),
                "renewable_percent": float(renewable_percent),
            }
        )
    return pd.DataFrame(rows)


def gen_housing(
    rng: np.random.RandomState, pop_df: pd.DataFrame, eco_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    grouped = pop_df.groupby(["district", "period_start", "year", "replica_id"])
    for (d, period, year, replica), g in grouped:
        pop = int(g["population"].iloc[0])
        urban_pct = float(g["urban_percent"].iloc[0]) / 100.0
        income = int(
            eco_df[
                (eco_df["district"] == d)
                & (eco_df["year"] == int(year))
                & (eco_df["replica_id"] == int(replica))
            ]["avg_income_bdt"].iloc[0]
        )
        housing_units = int(
            round(clamp(add_noise(rng, pop / 4.5, 0.12), 50_000.0, 1_500_000.0))
        )
        avg_rent_bdt = int(
            round(
                clamp(
                    add_noise(
                        rng,
                        3000.0 + 12000.0 * urban_pct + 0.2 * (income - 100000.0),
                        0.2,
                    ),
                    2000.0,
                    60000.0,
                )
            )
        )
        vacancy_rate = round(
            clamp(
                add_noise(rng, 0.09 - 0.04 * urban_pct + rng.normal(0.0, 0.01), 0.04),
                0.02,
                0.18,
            )
            * 100.0,
            2,
        )
        slum_population_percent = round(
            clamp(
                add_noise(
                    rng, 0.08 + 0.12 * urban_pct - 0.08 * (income / 250000.0), 0.04
                ),
                0.03,
                0.4,
            )
            * 100.0,
            2,
        )
        rows.append(
            {
                "district": d,
                "period_start": period,
                "year": int(year),
                "replica_id": int(replica),
                "housing_units": int(housing_units),
                "avg_rent_bdt": int(avg_rent_bdt),
                "vacancy_rate_percent": float(vacancy_rate),
                "slum_population_percent": float(slum_population_percent),
            }
        )
    return pd.DataFrame(rows)


def gen_technology_connectivity(
    rng: np.random.RandomState, pop_df: pd.DataFrame, eco_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    grouped = pop_df.groupby(["district", "period_start", "year", "replica_id"])
    for (d, period, year, replica), g in grouped:
        t = year_idx(int(year), list(pop_df["year"].unique()))
        income = int(
            eco_df[
                (eco_df["district"] == d)
                & (eco_df["year"] == int(year))
                & (eco_df["replica_id"] == int(replica))
            ]["avg_income_bdt"].iloc[0]
        )
        urban_pct = float(g["urban_percent"].iloc[0]) / 100.0
        internet = round(
            clamp(
                add_noise(
                    rng,
                    0.25 + 0.02 * t + 0.2 * urban_pct + 0.15 * (income / 250000.0),
                    0.05,
                ),
                0.2,
                0.98,
            )
            * 100.0,
            2,
        )
        mobile = round(
            clamp(add_noise(rng, 0.5 + 0.015 * t + 0.1 * urban_pct, 0.04), 0.4, 0.98)
            * 100.0,
            2,
        )
        rows.append(
            {
                "district": d,
                "period_start": period,
                "year": int(year),
                "replica_id": int(replica),
                "internet_penetration_percent": float(internet),
                "mobile_penetration_percent": float(mobile),
            }
        )
    return pd.DataFrame(rows)


def gen_agriculture_food_security(
    rng: np.random.RandomState,
    districts: List[str],
    periods: Iterable[pd.Timestamp],
    land_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    replicate: int,
) -> pd.DataFrame:
    rows = []
    for period in periods:
        year = int(period.year)
        for d in districts:
            for r in range(replicate):
                rainfall = int(
                    climate_df[
                        (climate_df["district"] == d)
                        & (climate_df["year"] == year)
                        & (climate_df["replica_id"] == r)
                    ]["rainfall_mm"].iloc[0]
                )
                drought = float(
                    climate_df[
                        (climate_df["district"] == d)
                        & (climate_df["year"] == year)
                        & (climate_df["replica_id"] == r)
                    ]["drought_index"].iloc[0]
                )
                agri_area = float(
                    land_df[
                        (land_df["district"] == d)
                        & (land_df["year"] == year)
                        & (land_df["replica_id"] == r)
                    ]["agricultural_area_km2"].iloc[0]
                )
                yield_ton_per_ha = round(
                    clamp(
                        add_noise(
                            rng,
                            2.5 + 0.001 * (rainfall - 1800.0) - 0.08 * drought,
                            0.15,
                        ),
                        1.5,
                        6.5,
                    ),
                    2,
                )
                fertilizer = int(
                    round(
                        clamp(
                            add_noise(
                                rng, 120.0 + 5.0 * (yield_ton_per_ha - 2.5), 0.12
                            ),
                            60.0,
                            300.0,
                        )
                    )
                )
                irrigated_area_km2 = int(
                    round(
                        clamp(
                            add_noise(rng, agri_area * (0.25 + 0.2), 0.1),
                            50.0,
                            max(60.0, agri_area * 0.7),
                        )
                    )
                )
                total_production_tons = int(
                    round(
                        clamp(
                            add_noise(rng, yield_ton_per_ha * (agri_area * 100.0), 0.2),
                            50_000.0,
                            30_000_000.0,
                        )
                    )
                )
                rows.append(
                    {
                        "district": d,
                        "period_start": period,
                        "year": int(year),
                        "replica_id": int(r),
                        "crop_yield_ton_per_hectare": float(yield_ton_per_ha),
                        "fertilizer_kg_per_hectare": int(fertilizer),
                        "irrigated_area_km2": int(irrigated_area_km2),
                        "total_production_tons": int(total_production_tons),
                    }
                )
    return pd.DataFrame(rows)


# ------------------------------
# Orchestration / main generate_all
# ------------------------------
def generate_all(
    districts: List[str],
    years: List[int],
    freq: str,
    replicate: int,
    seed: int,
    out_dir: Path,
):
    rng = np.random.RandomState(seed)
    ensure_out_dir(out_dir)
    periods = make_period_index(years, freq)
    logger.info(
        "Configuration: districts=%d years=%d periods=%d freq=%s replicate=%d seed=%d out_dir=%s",
        len(districts),
        len(years),
        len(periods),
        freq,
        replicate,
        seed,
        str(out_dir),
    )

    # 1 Population & Demographics
    pop_df = gen_population_demographics(rng, districts, periods, replicate, years)
    pop_csv = out_dir / "population_demographics.csv"
    pop_df.to_csv(pop_csv, index=False)
    logger.info("Wrote %s rows=%d", pop_csv.name, len(pop_df))

    # 2 Resource Demand
    res_df = gen_resource_demand(rng, pop_df)
    res_csv = out_dir / "resource_demand.csv"
    res_df.to_csv(res_csv, index=False)
    logger.info("Wrote %s rows=%d", res_csv.name, len(res_df))

    # 3 Urban Infrastructure
    infra_df = gen_urban_infrastructure(rng, pop_df)
    infra_csv = out_dir / "urban_infrastructure.csv"
    infra_df.to_csv(infra_csv, index=False)
    logger.info("Wrote %s rows=%d", infra_csv.name, len(infra_df))

    # 4 Migration Trends
    mig_df = gen_migration_trends(rng, pop_df, districts)
    mig_csv = out_dir / "migration_trends.csv"
    mig_df.to_csv(mig_csv, index=False)
    logger.info("Wrote %s rows=%d", mig_csv.name, len(mig_df))

    # 5 Economic Indicators
    eco_df = gen_economic_indicators(rng, districts, periods, replicate, years)
    eco_csv = out_dir / "economic_indicators.csv"
    eco_df.to_csv(eco_csv, index=False)
    logger.info("Wrote %s rows=%d", eco_csv.name, len(eco_df))

    # 6 Climate Impact
    climate_df = gen_climate_impact(rng, districts, periods, replicate, years)
    climate_csv = out_dir / "climate_impact.csv"
    climate_df.to_csv(climate_csv, index=False)
    logger.info("Wrote %s rows=%d", climate_csv.name, len(climate_df))

    # 7 Land Use Zoning
    land_df = gen_land_use_zoning(rng, districts, periods, replicate)
    land_csv = out_dir / "land_use_zoning.csv"
    land_df.to_csv(land_csv, index=False)
    logger.info("Wrote %s rows=%d", land_csv.name, len(land_df))

    # 8 Transportation
    trans_df = gen_transportation(rng, pop_df)
    trans_csv = out_dir / "transportation.csv"
    trans_df.to_csv(trans_csv, index=False)
    logger.info("Wrote %s rows=%d", trans_csv.name, len(trans_df))

    # 9 Health & Sanitation
    health_df = gen_health_sanitation(rng, pop_df)
    health_csv = out_dir / "health_sanitation.csv"
    health_df.to_csv(health_csv, index=False)
    logger.info("Wrote %s rows=%d", health_csv.name, len(health_df))

    # 10 Education
    edu_df = gen_education(rng, pop_df, eco_df)
    edu_csv = out_dir / "education.csv"
    edu_df.to_csv(edu_csv, index=False)
    logger.info("Wrote %s rows=%d", edu_csv.name, len(edu_df))

    # 11 Energy Usage
    energy_df = gen_energy_usage(rng, pop_df, eco_df)
    energy_csv = out_dir / "energy_usage.csv"
    energy_df.to_csv(energy_csv, index=False)
    logger.info("Wrote %s rows=%d", energy_csv.name, len(energy_df))

    # 12 Housing
    house_df = gen_housing(rng, pop_df, eco_df)
    house_csv = out_dir / "housing.csv"
    house_df.to_csv(house_csv, index=False)
    logger.info("Wrote %s rows=%d", house_csv.name, len(house_df))

    # 13 Technology & Connectivity
    tech_df = gen_technology_connectivity(rng, pop_df, eco_df)
    tech_csv = out_dir / "technology_connectivity.csv"
    tech_df.to_csv(tech_csv, index=False)
    logger.info("Wrote %s rows=%d", tech_csv.name, len(tech_df))

    # 14 Agriculture & Food Security
    agri_df = gen_agriculture_food_security(
        rng, districts, periods, land_df, climate_df, replicate
    )
    agri_csv = out_dir / "agriculture_food_security.csv"
    agri_df.to_csv(agri_csv, index=False)
    logger.info("Wrote %s rows=%d", agri_csv.name, len(agri_df))

    logger.info("Generation complete. CSVs written to %s", str(out_dir.resolve()))


# ------------------------------
# CLI
# ------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic Bangladesh district-level data generator"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for CSVs",
    )
    p.add_argument(
        "--freq",
        choices=("Y", "M"),
        default=DEFAULT_FREQ,
        help='Frequency: "Y" yearly rows or "M" monthly rows',
    )
    p.add_argument(
        "--replicate",
        type=int,
        default=DEFAULT_REPLICATE,
        help="Replication factor per district-period",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="RNG seed for reproducibility"
    )
    p.add_argument(
        "--districts",
        type=str,
        default=",".join(DEFAULT_DISTRICTS),
        help="Comma-separated list of districts",
    )
    p.add_argument(
        "--start-year", type=int, default=DEFAULT_YEARS[0], help="Start year"
    )
    p.add_argument("--end-year", type=int, default=DEFAULT_YEARS[-1], help="End year")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    districts = [d.strip() for d in args.districts.split(",") if d.strip()]
    years = list(range(args.start_year, args.end_year + 1))
    generate_all(
        districts=districts,
        years=years,
        freq=args.freq,
        replicate=args.replicate,
        seed=args.seed,
        out_dir=args.out_dir,
    )
