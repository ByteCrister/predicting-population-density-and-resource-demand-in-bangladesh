# predicting-population-density-and-resource-demand--in-bangladesh/scripts/bangladesh_data_generator.py
"""
Production-grade synthetic data generator for district-level
population, resources, infrastructure and climate indicators.

Fixes:
- deterministic randomness via RandomState
- stable numeric types and explicit conversions
- deterministic density calculation and tolerant validation
- improved helper utilities and defensive checks
"""

from typing import Any
import numpy as np
import pandas as pd

# ------------------------------
# Configuration
# ------------------------------
districts = ["Dhaka", "Sylhet", "Chattogram", "Rajshahi", "Khulna"]
years = list(range(2010, 2026))

RNG = np.random.RandomState(42)  # deterministic random state for reproducibility

DIST_AREA = {
    "Dhaka": 1460,
    "Sylhet": 3490,
    "Chattogram": 5283,
    "Rajshahi": 2400,
    "Khulna": 4395,
}

DIST_POP_2010 = {
    "Dhaka": 12_000_000,
    "Sylhet": 4_000_000,
    "Chattogram": 7_000_000,
    "Rajshahi": 5_000_000,
    "Khulna": 4_500_000,
}

DIST_POP_GROWTH = {
    "Dhaka": 0.025,
    "Sylhet": 0.018,
    "Chattogram": 0.02,
    "Rajshahi": 0.015,
    "Khulna": 0.014,
}

DIST_URBAN_SHARE_2010 = {
    "Dhaka": 0.85,
    "Sylhet": 0.45,
    "Chattogram": 0.65,
    "Rajshahi": 0.55,
    "Khulna": 0.6,
}

DIST_URBAN_TREND = {
    "Dhaka": 0.003,
    "Sylhet": 0.004,
    "Chattogram": 0.003,
    "Rajshahi": 0.0025,
    "Khulna": 0.0025,
}

DIST_AVG_INCOME_2010 = {
    "Dhaka": 200_000,
    "Sylhet": 120_000,
    "Chattogram": 160_000,
    "Rajshahi": 110_000,
    "Khulna": 115_000,
}

DIST_INCOME_GROWTH = {
    "Dhaka": 0.05,
    "Sylhet": 0.035,
    "Chattogram": 0.045,
    "Rajshahi": 0.032,
    "Khulna": 0.034,
}

DIST_RAINFALL_BASE = {
    "Dhaka": 2200,
    "Sylhet": 3500,
    "Chattogram": 2800,
    "Rajshahi": 1500,
    "Khulna": 1800,
}

DIST_TEMP_BASE = {
    "Dhaka": 28.0,
    "Sylhet": 26.5,
    "Chattogram": 27.5,
    "Rajshahi": 29.0,
    "Khulna": 28.5,
}

DIST_FLOOD_RISK_BASE = {
    "Dhaka": 7,
    "Sylhet": 6,
    "Chattogram": 5,
    "Rajshahi": 3,
    "Khulna": 4,
}

SECTOR_SHARES_2010 = {
    "Dhaka": (0.05, 0.35, 0.60),
    "Sylhet": (0.25, 0.25, 0.50),
    "Chattogram": (0.12, 0.38, 0.50),
    "Rajshahi": (0.30, 0.25, 0.45),
    "Khulna": (0.22, 0.33, 0.45),
}


# ------------------------------
# Helper utilities
# ------------------------------
def clamp(x: Any, low: float, high: float) -> float:
    """Return scalar float clamped to [low, high]. Accepts numpy scalars."""
    val = float(np.asarray(x).item())
    if val < low:
        return float(low)
    if val > high:
        return float(high)
    return float(val)


def year_idx(y: int) -> int:
    return int(y - years[0])


def growth(base: float, rate: float, t: int) -> float:
    return float(base * ((1.0 + rate) ** t))


def drift(value: float, annual_change: float, t: int) -> float:
    return float(value + annual_change * t)


def pct_noise(rng: np.random.RandomState, scale: float = 0.02) -> float:
    """Return multiplier ~1.0 +/- noise"""
    return float(1.0 + rng.normal(0.0, scale))


def add_noise(rng: np.random.RandomState, x: float, scale: float = 0.02) -> float:
    """Additive noise proportional to absolute value to avoid sign flips"""
    return float(x + rng.normal(0.0, abs(scale * x)))


# Tolerant density validator
DENSITY_TOLERANCE = 1e-2  # 0.01 people/km2 tolerance


def validate_density(district: str, year: int, population: int, area_km2: float, density_reported: float):
    expected = round(float(population) / float(area_km2), 2)
    # tolerant comparison
    if abs(expected - float(density_reported)) > DENSITY_TOLERANCE:
        raise ValueError(f"density mismatch {district} {year}: expected {expected} got {density_reported}")


# ------------------------------
# 1. Population & Demographics
# ------------------------------
pop_rows = []
for d in districts:
    for y in years:
        t = year_idx(y)
        base_pop = DIST_POP_2010[d]
        pop_val = int(round(add_noise(RNG, growth(base_pop, DIST_POP_GROWTH[d], t), 0.03)))
        area = float(DIST_AREA[d])
        density = round(float(pop_val) / area, 2)
        urban_share = clamp(drift(DIST_URBAN_SHARE_2010[d], DIST_URBAN_TREND[d], t), 0.2, 0.98)
        urban_pop = int(round(pop_val * urban_share))
        rural_pop = int(pop_val - urban_pop)

        children_pct = clamp(0.28 - 0.03 * urban_share + RNG.normal(0, 0.01), 0.18, 0.35)
        elderly_pct = clamp(0.07 + 0.01 * (1 - urban_share) + RNG.normal(0, 0.005), 0.05, 0.12)
        working_pct = clamp(1.0 - children_pct - elderly_pct, 0.55, 0.75)

        pop_rows.append(
            {
                "district": d,
                "year": int(y),
                "population": int(pop_val),
                "area_km2": float(area),
                "density_per_km2": float(density),
                "children_percent": round(children_pct * 100.0, 2),
                "working_age_percent": round(working_pct * 100.0, 2),
                "elderly_percent": round(elderly_pct * 100.0, 2),
                "urban_percent": round(urban_share * 100.0, 2),
                "urban_population": int(urban_pop),
                "rural_population": int(rural_pop),
            }
        )

pop_df = pd.DataFrame(pop_rows)
# validation pass to catch real inconsistencies early
for _, row in pop_df.iterrows():
    validate_density(row["district"], int(row["year"]), int(row["population"]), float(row["area_km2"]), float(row["density_per_km2"]))

pop_df = pop_df.astype(
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
    }
)
pop_df.to_csv("population_demographics.csv", index=False)

# ------------------------------
# 2. Resource Demand
# ------------------------------
res_rows = []
for d in districts:
    for y in years:
        pop = int(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "population"].iloc[0])
        income = float(add_noise(RNG, growth(DIST_AVG_INCOME_2010[d], DIST_INCOME_GROWTH[d], year_idx(y)), 0.05))
        water_mld = clamp((pop / 1000.0) * 0.15 * pct_noise(RNG, 0.08), 50.0, 5000.0)
        urban_pct = float(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "urban_percent"].iloc[0]) / 100.0
        elec_mwh = clamp((pop * (0.25 + 0.5 * urban_pct) * (income / 150000.0)) * 0.05 * pct_noise(RNG, 0.1), 1000.0, 300000.0)
        households = pop / 4.5
        housing_units_demand = int(round(clamp(households * 0.03 * pct_noise(RNG, 0.1), 1000.0, 600000.0)))
        food_tons = int(round(clamp(pop * 0.55 * pct_noise(RNG, 0.05), 50_000.0, 15_000_000.0)))

        res_rows.append(
            {
                "district": d,
                "year": int(y),
                "water_mld": round(float(water_mld), 2),
                "electricity_mwh": int(round(elec_mwh)),
                "housing_units_demand": int(housing_units_demand),
                "food_demand_tons": int(food_tons),
            }
        )

res_df = pd.DataFrame(res_rows)
res_df.to_csv("resource_demand.csv", index=False)

# ------------------------------
# 3. Urban Infrastructure
# ------------------------------
infra_rows = []
base_roads = {"Dhaka": 2800, "Sylhet": 1500, "Chattogram": 2200, "Rajshahi": 1800, "Khulna": 1900}
for d in districts:
    for y in years:
        t = year_idx(y)
        pop = int(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "population"].iloc[0])
        base_road = base_roads[d]
        road_km = int(round(clamp(add_noise(RNG, base_road * ((1.0 + 0.015) ** t) + (pop / 1e5), 0.05), 500.0, 10000.0)))
        schools = int(round(clamp(add_noise(RNG, pop / 4000.0, 0.15), 200.0, 8000.0)))
        hospitals = int(round(clamp(add_noise(RNG, pop / 80000.0, 0.2), 10.0, 600.0)))
        green_km2 = round(clamp(add_noise(RNG, float(DIST_AREA[d]) * (0.08 + 0.03 * (1 - (pop / (DIST_AREA[d] * 25000.0)))), 0.1), 5.0, 800.0), 2)
        infra_rows.append(
            {"district": d, "year": int(y), "road_km": road_km, "schools": schools, "hospitals": hospitals, "green_space_km2": green_km2}
        )

infra_df = pd.DataFrame(infra_rows)
infra_df.to_csv("urban_infrastructure.csv", index=False)

# ------------------------------
# 4. Migration Trends
# ------------------------------
mig_rows = []
for y in years:
    t = year_idx(y)
    for d1 in districts:
        origin_pop = int(pop_df.loc[(pop_df["district"] == d1) & (pop_df["year"] == y), "population"].iloc[0])
        for d2 in districts:
            if d1 == d2:
                continue
            inc2 = add_noise(RNG, growth(DIST_AVG_INCOME_2010[d2], DIST_INCOME_GROWTH[d2], t), 0.05)
            inc1 = add_noise(RNG, growth(DIST_AVG_INCOME_2010[d1], DIST_INCOME_GROWTH[d1], t), 0.05)
            urb2 = clamp(drift(DIST_URBAN_SHARE_2010[d2], DIST_URBAN_TREND[d2], t), 0.2, 0.98)
            urb1 = clamp(drift(DIST_URBAN_SHARE_2010[d1], DIST_URBAN_TREND[d1], t), 0.2, 0.98)
            attractiveness = clamp((inc2 - inc1) / 50000.0 + (urb2 - urb1) * 0.8 + RNG.normal(0.0, 0.1), -1.5, 1.5)
            base_flow = 500.0 + 3000.0 * max(0.0, attractiveness)
            migrants = int(round(clamp(add_noise(RNG, base_flow + origin_pop / 2e5, 0.3), 100.0, 80000.0)))
            mig_rows.append({"from_district": d1, "to_district": d2, "year": int(y), "migrants": migrants})

mig_df = pd.DataFrame(mig_rows)
mig_df.to_csv("migration_trends.csv", index=False)

# ------------------------------
# 5. Economic Indicators
# ------------------------------
eco_rows = []
for d in districts:
    ag0, ind0, serv0 = SECTOR_SHARES_2010[d]
    for y in years:
        t = year_idx(y)
        avg_income = int(round(clamp(add_noise(RNG, growth(DIST_AVG_INCOME_2010[d], DIST_INCOME_GROWTH[d], t), 0.05), 60_000.0, 600_000.0)))
        employment_rate = clamp(0.5 + 0.25 * (avg_income / 250_000.0) + RNG.normal(0.0, 0.03), 0.45, 0.96)
        poverty_rate = clamp(0.35 - 0.2 * (avg_income / 250_000.0) + RNG.normal(0.0, 0.02), 0.05, 0.45)
        drift_services = serv0 + 0.01 * t + RNG.normal(0.0, 0.01)
        drift_industry = ind0 + 0.003 * t + RNG.normal(0.0, 0.008)
        drift_agri = ag0 - 0.013 * t + RNG.normal(0.0, 0.008)
        total = drift_agri + drift_industry + drift_services
        agri_share = clamp(drift_agri / total, 0.05, 0.5)
        industry_share = clamp(drift_industry / total, 0.15, 0.5)
        services_share = clamp(1.0 - agri_share - industry_share, 0.3, 0.8)
        eco_rows.append(
            {
                "district": d,
                "year": int(y),
                "avg_income_bdt": int(avg_income),
                "employment_rate_percent": round(employment_rate * 100.0, 2),
                "poverty_rate_percent": round(poverty_rate * 100.0, 2),
                "agri_share_percent": round(agri_share * 100.0, 2),
                "industry_share_percent": round(industry_share * 100.0, 2),
                "services_share_percent": round(services_share * 100.0, 2),
            }
        )

eco_df = pd.DataFrame(eco_rows)
eco_df.to_csv("economic_indicators.csv", index=False)

# ------------------------------
# 6. Climate Impact
# ------------------------------
climate_rows = []
for d in districts:
    for y in years:
        t = year_idx(y)
        avg_temp = round(clamp(add_noise(RNG, DIST_TEMP_BASE[d] + 0.02 * t, 0.02), 22.0, 35.0), 2)
        rainfall = int(round(clamp(add_noise(RNG, DIST_RAINFALL_BASE[d] + 10.0 * np.sin(t / 2.0), 0.08), 800.0, 4500.0)))
        flood_risk = int(round(clamp(add_noise(RNG, DIST_FLOOD_RISK_BASE[d] + 0.1 * (rainfall / 1000.0), 0.2), 1.0, 10.0)))
        lam = (rainfall - 1500.0) / 1200.0 if d in ["Chattogram", "Khulna", "Sylhet"] else 0.5
        cyclones = int(round(clamp(RNG.poisson(max(0.0, lam)) + RNG.normal(0.0, 0.3), 0.0, 8.0)))
        drought_index = round(clamp(add_noise(RNG, (2500.0 - rainfall) / 300.0, 0.2), 0.0, 10.0), 2)
        climate_rows.append(
            {"district": d, "year": int(y), "avg_temp_c": avg_temp, "rainfall_mm": int(rainfall), "flood_risk_score": flood_risk, "cyclone_events": cyclones, "drought_index": drought_index}
        )

climate_df = pd.DataFrame(climate_rows)
climate_df.to_csv("climate_impact.csv", index=False)

# ------------------------------
# 7. Land Use Zoning
# ------------------------------
land_rows = []
for d in districts:
    area = float(DIST_AREA[d])
    for y in years:
        urban_pct = float(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "urban_percent"].iloc[0]) / 100.0
        residential = clamp(add_noise(RNG, area * (0.15 + 0.25 * urban_pct), 0.06), 50.0, area * 0.7)
        commercial = clamp(add_noise(RNG, area * (0.05 + 0.12 * urban_pct), 0.08), 10.0, area * 0.3)
        industrial = clamp(add_noise(RNG, area * (0.04 + 0.1 * urban_pct), 0.08), 5.0, area * 0.25)
        agricultural = clamp(area - residential - commercial - industrial, area * 0.1, area * 0.85)
        land_rows.append(
            {
                "district": d,
                "year": int(y),
                "residential_area_km2": round(float(residential), 2),
                "commercial_area_km2": round(float(commercial), 2),
                "industrial_area_km2": round(float(industrial), 2),
                "agricultural_area_km2": round(float(agricultural), 2),
            }
        )

land_df = pd.DataFrame(land_rows)
land_df.to_csv("land_use_zoning.csv", index=False)

# ------------------------------
# 8. Transportation
# ------------------------------
trans_rows = []
for d in districts:
    for y in years:
        t = year_idx(y)
        urban_pop = int(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "urban_population"].iloc[0])
        bus_routes = int(round(clamp(add_noise(RNG, urban_pop / 80000.0, 0.2), 20.0, 2000.0)))
        rail_base = 50 + 2 * t + {"Dhaka": 30, "Chattogram": 20}.get(d, 5)
        rail_km = int(round(clamp(add_noise(RNG, rail_base, 0.1), 20.0, 800.0)))
        density = float(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "density_per_km2"].iloc[0])
        avg_commute = int(round(clamp(add_noise(RNG, 25.0 + 0.01 * density - 0.1 * t, 0.1), 15.0, 90.0)))
        vehicle_count = int(round(clamp(add_noise(RNG, urban_pop * (0.18 + 0.4), 0.15), 20_000.0, 900_000.0)))
        trans_rows.append({"district": d, "year": int(y), "bus_routes": bus_routes, "rail_km": rail_km, "avg_commute_time_min": avg_commute, "vehicle_count": vehicle_count})

trans_df = pd.DataFrame(trans_rows)
trans_df.to_csv("transportation.csv", index=False)

# ------------------------------
# 9. Health & Sanitation
# ------------------------------
health_rows = []
for d in districts:
    for y in years:
        t = year_idx(y)
        pop = int(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "population"].iloc[0])
        hospital_beds = int(round(clamp(add_noise(RNG, pop / 700.0, 0.2), 300.0, 30000.0)))
        clinics = int(round(clamp(add_noise(RNG, pop / 30000.0, 0.25), 30.0, 1500.0)))
        urban_pct = float(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "urban_percent"].iloc[0]) / 100.0
        sanitation_coverage = clamp(add_noise(RNG, 0.55 + 0.01 * t + 0.1 * urban_pct, 0.05), 0.4, 0.99)
        waste_collection = clamp(add_noise(RNG, 0.45 + 0.012 * t + 0.12 * urban_pct, 0.06), 0.3, 0.98)
        health_rows.append(
            {
                "district": d,
                "year": int(y),
                "hospital_beds": hospital_beds,
                "clinics": clinics,
                "sanitation_coverage_percent": round(sanitation_coverage * 100.0, 2),
                "waste_collection_percent": round(waste_collection * 100.0, 2),
            }
        )

health_df = pd.DataFrame(health_rows)
health_df.to_csv("health_sanitation.csv", index=False)

# ------------------------------
# 10. Education
# ------------------------------
edu_rows = []
for d in districts:
    for y in years:
        t = year_idx(y)
        pop = int(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "population"].iloc[0])
        schools = int(round(clamp(add_noise(RNG, pop / 3500.0, 0.15), 300.0, 9000.0)))
        income = int(eco_df.loc[(eco_df["district"] == d) & (eco_df["year"] == y), "avg_income_bdt"].iloc[0])
        literacy = clamp(add_noise(RNG, 0.55 + 0.12 * (income / 250_000.0) + 0.01 * t, 0.03), 0.5, 0.98)
        student_teacher_ratio = round(clamp(add_noise(RNG, 40.0 - 0.6 * t + RNG.normal(0.0, 1.5), 0.05), 15.0, 55.0), 1)
        edu_rows.append({"district": d, "year": int(y), "schools": schools, "literacy_rate_percent": round(literacy * 100.0, 2), "student_teacher_ratio": student_teacher_ratio})

edu_df = pd.DataFrame(edu_rows)
edu_df.to_csv("education.csv", index=False)

# ------------------------------
# 11. Energy Usage
# ------------------------------
energy_rows = []
for d in districts:
    for y in years:
        pop = int(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "population"].iloc[0])
        urban_pct = float(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "urban_percent"].iloc[0]) / 100.0
        income = int(eco_df.loc[(eco_df["district"] == d) & (eco_df["year"] == y), "avg_income_bdt"].iloc[0])
        residential_mwh = int(round(clamp(add_noise(RNG, pop * (0.03 + 0.06 * urban_pct) * (income / 180000.0), 0.12), 500.0, 250000.0)))
        commercial_mwh = int(round(clamp(add_noise(RNG, pop * (0.01 + 0.05 * urban_pct) * (income / 200000.0), 0.15), 1000.0, 300000.0)))
        industrial_mwh = int(round(clamp(add_noise(RNG, pop * (0.02 + 0.04 * (1 - urban_pct)) * (income / 220000.0), 0.18), 2000.0, 400000.0)))
        renewable_percent = round(clamp(add_noise(RNG, 0.08 + 0.01 * year_idx(y) + 0.05 * (1 - urban_pct), 0.05), 0.05, 0.55) * 100.0, 2)
        energy_rows.append({"district": d, "year": int(y), "residential_mwh": residential_mwh, "commercial_mwh": commercial_mwh, "industrial_mwh": industrial_mwh, "renewable_percent": renewable_percent})

energy_df = pd.DataFrame(energy_rows)
energy_df.to_csv("energy_usage.csv", index=False)

# ------------------------------
# 12. Housing
# ------------------------------
house_rows = []
for d in districts:
    for y in years:
        pop = int(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "population"].iloc[0])
        urban_pct = float(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "urban_percent"].iloc[0]) / 100.0
        income = int(eco_df.loc[(eco_df["district"] == d) & (eco_df["year"] == y), "avg_income_bdt"].iloc[0])
        housing_units = int(round(clamp(add_noise(RNG, pop / 4.5, 0.12), 50_000.0, 1_500_000.0)))
        avg_rent_bdt = int(round(clamp(add_noise(RNG, 3000.0 + 12000.0 * urban_pct + 0.2 * (income - 100000.0), 0.2), 2000.0, 60000.0)))
        vacancy_rate = round(clamp(add_noise(RNG, 0.09 - 0.04 * urban_pct + RNG.normal(0.0, 0.01), 0.04), 0.02, 0.18) * 100.0, 2)
        slum_population_percent = round(clamp(add_noise(RNG, 0.08 + 0.12 * urban_pct - 0.08 * (income / 250000.0), 0.04), 0.03, 0.4) * 100.0, 2)
        house_rows.append({"district": d, "year": int(y), "housing_units": housing_units, "avg_rent_bdt": avg_rent_bdt, "vacancy_rate_percent": vacancy_rate, "slum_population_percent": slum_population_percent})

house_df = pd.DataFrame(house_rows)
house_df.to_csv("housing.csv", index=False)

# ------------------------------
# 13. Technology & Connectivity
# ------------------------------
tech_rows = []
for d in districts:
    for y in years:
        t = year_idx(y)
        income = int(eco_df.loc[(eco_df["district"] == d) & (eco_df["year"] == y), "avg_income_bdt"].iloc[0])
        urban_pct = float(pop_df.loc[(pop_df["district"] == d) & (pop_df["year"] == y), "urban_percent"].iloc[0]) / 100.0
        internet = round(clamp(add_noise(RNG, 0.25 + 0.02 * t + 0.2 * urban_pct + 0.15 * (income / 250000.0), 0.05), 0.2, 0.98) * 100.0, 2)
        mobile = round(clamp(add_noise(RNG, 0.5 + 0.015 * t + 0.1 * urban_pct, 0.04), 0.4, 0.98) * 100.0, 2)
        tech_rows.append({"district": d, "year": int(y), "internet_penetration_percent": internet, "mobile_penetration_percent": mobile})

tech_df = pd.DataFrame(tech_rows)
tech_df.to_csv("technology_connectivity.csv", index=False)

# ------------------------------
# 14. Agriculture & Food Security
# ------------------------------
agri_rows = []
for d in districts:
    for y in years:
        rainfall = int(climate_df.loc[(climate_df["district"] == d) & (climate_df["year"] == y), "rainfall_mm"].iloc[0])
        drought = float(climate_df.loc[(climate_df["district"] == d) & (climate_df["year"] == y), "drought_index"].iloc[0])
        agri_area = float(land_df.loc[(land_df["district"] == d) & (land_df["year"] == y), "agricultural_area_km2"].iloc[0])
        yield_ton_per_ha = round(clamp(add_noise(RNG, 2.5 + 0.001 * (rainfall - 1800.0) - 0.08 * drought, 0.15), 1.5, 6.5), 2)
        fertilizer = int(round(clamp(add_noise(RNG, 120.0 + 5.0 * (yield_ton_per_ha - 2.5), 0.12), 60.0, 300.0)))
        irrigated_area_km2 = int(round(clamp(add_noise(RNG, agri_area * (0.25 + 0.2), 0.1), 50.0, max(60.0, agri_area * 0.7))))
        total_production_tons = int(round(clamp(add_noise(RNG, yield_ton_per_ha * (agri_area * 100.0), 0.2), 50_000.0, 30_000_000.0)))
        agri_rows.append({"district": d, "year": int(y), "crop_yield_ton_per_hectare": yield_ton_per_ha, "fertilizer_kg_per_hectare": fertilizer, "irrigated_area_km2": irrigated_area_km2, "total_production_tons": total_production_tons})

agri_df = pd.DataFrame(agri_rows)
agri_df.to_csv("agriculture_food_security.csv", index=False)

# End
print("Generation complete. All CSVs written and basic validations passed.")
