# Bangladesh Urban Growth & Resource Demand — Data Dictionary

## Table of Contents

1. [Population & Demographics](#population--demographics)
2. [Population Density](#population-density)
3. [Resource Demand](#resource-demand)
4. [Urban Infrastructure](#urban-infrastructure)
5. [Land Use & Zoning](#land-use--zoning)
6. [Climate & Environmental Impact](#climate--environmental-impact)
7. [Economic Indicators](#economic-indicators)
8. [Migration](#migration)
9. [Health & Sanitation](#health--sanitation)
10. [Education](#education)
11. [Transportation](#transportation)
12. [Housing](#housing)
13. [Technology & Connectivity](#technology--connectivity)
14. [Agriculture & Food Security](#agriculture--food-security)
15. [District-Year Summary](#district-year-summary)
16. [Notes](#notes)

---

## Population & Demographics

### `population_demographics.csv`

| Column              | Unit            | Type      | Description                                     |
| ------------------- | --------------- | --------- | ----------------------------------------------- |
| district            | none            | string    | Administrative district name.                   |
| period_start        | date (ISO)      | timestamp | Period start (first day of interval).           |
| year                | yyyy            | integer   | Year of observation.                            |
| replica_id          | none            | integer   | Replica index for deterministic replication.    |
| population          | persons         | integer   | Total residents in the district in that period. |
| area_km2            | km²             | float     | Land area used for density calculations.        |
| density_per_km2     | persons per km² | float     | population / area_km2 (validated).              |
| children_percent    | percent         | float     | % of population aged 0–14.                      |
| working_age_percent | percent         | float     | % of population aged 15–64.                     |
| elderly_percent     | percent         | float     | % of population aged 65+.                       |
| urban_percent       | percent         | float     | % living in urban areas.                        |
| urban_population    | persons         | integer   | Count of urban residents.                       |
| rural_population    | persons         | integer   | Count of rural residents.                       |

---

## Population Density

### `population_density.csv` (merged/aggregated)

| Column          | Unit            | Type    | Description                                 |
| --------------- | --------------- | ------- | ------------------------------------------- |
| district        | none            | string  | District name.                              |
| year            | yyyy            | integer | Year of observation.                        |
| population      | persons         | integer | Total residents (aggregated from replicas). |
| area_km2        | km²             | float   | District land area.                         |
| density_per_km2 | persons per km² | float   | population / area_km2 (recomputed).         |

---

## Resource Demand

### `resource_demand.csv`

| Column               | Unit               | Type      | Description                     |
| -------------------- | ------------------ | --------- | ------------------------------- |
| district             | none               | string    | District name.                  |
| period_start         | date (ISO)         | timestamp | Period start.                   |
| year                 | yyyy               | integer   | Year of observation.            |
| replica_id           | none               | integer   | Replica index.                  |
| water_mld            | MLD                | float     | Daily water demand.             |
| electricity_mwh      | MWh                | integer   | Annual electricity demand.      |
| housing_units_demand | housing units/year | integer   | Net new housing units.          |
| food_demand_tons     | metric tons/year   | integer   | Annual food consumption demand. |

---

## Urban Infrastructure

### `urban_infrastructure.csv`

| Column          | Unit       | Type      | Description                      |
| --------------- | ---------- | --------- | -------------------------------- |
| district        | none       | string    | District name.                   |
| period_start    | date (ISO) | timestamp | Period start.                    |
| year            | yyyy       | integer   | Year of observation.             |
| replica_id      | none       | integer   | Replica index.                   |
| road_km         | km         | integer   | Total road network length.       |
| schools         | count      | integer   | Number of schools.               |
| hospitals       | count      | integer   | Number of hospitals.             |
| green_space_km2 | km²        | float     | Area of green/open public space. |

---

## Land Use & Zoning

### `land_use_zoning.csv`

| Column                | Unit       | Type      | Description                          |
| --------------------- | ---------- | --------- | ------------------------------------ |
| district              | none       | string    | District name.                       |
| period_start          | date (ISO) | timestamp | Period start.                        |
| year                  | yyyy       | integer   | Year of observation.                 |
| replica_id            | none       | integer   | Replica index.                       |
| residential_area_km2  | km²        | float     | Area zoned for residential purposes. |
| commercial_area_km2   | km²        | float     | Area zoned for commercial purposes.  |
| industrial_area_km2   | km²        | float     | Area zoned for industrial purposes.  |
| agricultural_area_km2 | km²        | float     | Area used for agriculture.           |

---

## Climate & Environmental Impact

### `climate_impact.csv`

| Column           | Unit        | Type      | Description                 |
| ---------------- | ----------- | --------- | --------------------------- |
| district         | none        | string    | District name.              |
| period_start     | date (ISO)  | timestamp | Period start.               |
| year             | yyyy        | integer   | Year of observation.        |
| replica_id       | none        | integer   | Replica index.              |
| avg_temp_c       | °C          | float     | Annual average temperature. |
| rainfall_mm      | mm          | integer   | Annual total rainfall.      |
| flood_risk_score | index 1–10  | integer   | Synthetic flood risk index. |
| cyclone_events   | events/year | integer   | Count of cyclone events.    |
| drought_index    | 0–10        | float     | Drought severity proxy.     |

### `environmental_stress.csv` (derived)

| Column              | Unit        | Type    | Description                                           |
| ------------------- | ----------- | ------- | ----------------------------------------------------- |
| district            | none        | string  | District name.                                        |
| year                | yyyy        | integer | Year of observation.                                  |
| air_quality_index   | index       | float   | Synthetic AQ proxy.                                   |
| heatwave_days       | days/year   | integer | Number of heatwave days.                              |
| flood_events        | events/year | integer | Count of flood events derived from climate variables. |
| green_cover_percent | percent     | float   | (green_space_km2 / area_km2) \* 100.                  |

---

## Economic Indicators

### `economic_indicators.csv`

| Column                  | Unit       | Type      | Description                            |
| ----------------------- | ---------- | --------- | -------------------------------------- |
| district                | none       | string    | District name.                         |
| period_start            | date (ISO) | timestamp | Period start.                          |
| year                    | yyyy       | integer   | Year of observation.                   |
| replica_id              | none       | integer   | Replica index.                         |
| avg_income_bdt          | BDT/year   | integer   | Avg annual per-capita income proxy.    |
| employment_rate_percent | percent    | float     | % of working-age population employed.  |
| poverty_rate_percent    | percent    | float     | % below synthetic poverty threshold.   |
| agri_share_percent      | percent    | float     | % of economic activity in agriculture. |
| industry_share_percent  | percent    | float     | % of economic activity in industry.    |
| services_share_percent  | percent    | float     | % of economic activity in services.    |

---

## Migration

### `migration_trends.csv`

| Column        | Unit         | Type      | Description                                           |
| ------------- | ------------ | --------- | ----------------------------------------------------- |
| from_district | none         | string    | Origin district name.                                 |
| to_district   | none         | string    | Destination district name.                            |
| period_start  | date (ISO)   | timestamp | Period start.                                         |
| year          | yyyy         | integer   | Year of migration flow.                               |
| migrants      | persons/year | integer   | Number of migrants moving from origin to destination. |

### `migration_trends_agg.csv`

| Column            | Unit           | Type    | Description                                |
| ----------------- | -------------- | ------- | ------------------------------------------ |
| district          | none           | string  | District name (aggregated inflow/outflow). |
| year              | yyyy           | integer | Year of observation.                       |
| migration_inflow  | persons/period | integer | Sum of incoming migrants.                  |
| migration_outflow | persons/period | integer | Sum of outgoing migrants.                  |
| net_migration     | persons/period | integer | migration_inflow - migration_outflow.      |

---

## Health & Sanitation

### `health_sanitation.csv`

| Column                      | Unit       | Type      | Description                     |
| --------------------------- | ---------- | --------- | ------------------------------- |
| district                    | none       | string    | District name.                  |
| period_start                | date (ISO) | timestamp | Period start.                   |
| year                        | yyyy       | integer   | Year of observation.            |
| replica_id                  | none       | integer   | Replica index.                  |
| hospital_beds               | beds       | integer   | Total hospital bed capacity.    |
| clinics                     | count      | integer   | Number of clinics.              |
| sanitation_coverage_percent | percent    | float     | % with basic sanitation access. |
| waste_collection_percent    | percent    | float     | % covered by waste collection.  |

---

## Education

### `education.csv`

| Column                | Unit             | Type      | Description                   |
| --------------------- | ---------------- | --------- | ----------------------------- |
| district              | none             | string    | District name.                |
| period_start          | date (ISO)       | timestamp | Period start.                 |
| year                  | yyyy             | integer   | Year of observation.          |
| replica_id            | none             | integer   | Replica index.                |
| schools               | count            | integer   | Number of schools.            |
| literacy_rate_percent | percent          | float     | % of literate population.     |
| student_teacher_ratio | students/teacher | float     | Average students per teacher. |

---

## Transportation

### `transportation.csv`

| Column               | Unit       | Type      | Description                   |
| -------------------- | ---------- | --------- | ----------------------------- |
| district             | none       | string    | District name.                |
| period_start         | date (ISO) | timestamp | Period start.                 |
| year                 | yyyy       | integer   | Year of observation.          |
| replica_id           | none       | integer   | Replica index.                |
| bus_routes           | count      | integer   | Number of bus routes.         |
| rail_km              | km         | integer   | Length of rail network.       |
| avg_commute_time_min | minutes    | integer   | Average one-way commute time. |
| vehicle_count        | vehicles   | integer   | Registered vehicles proxy.    |

---

## Housing

### `housing.csv`

| Column                  | Unit       | Type      | Description                        |
| ----------------------- | ---------- | --------- | ---------------------------------- |
| district                | none       | string    | District name.                     |
| period_start            | date (ISO) | timestamp | Period start.                      |
| year                    | yyyy       | integer   | Year of observation.               |
| replica_id              | none       | integer   | Replica index.                     |
| housing_units           | units      | integer   | Estimated housing stock.           |
| avg_rent_bdt            | BDT/month  | integer   | Typical monthly rent.              |
| vacancy_rate_percent    | percent    | float     | % of units vacant.                 |
| slum_population_percent | percent    | float     | % living in informal/slum housing. |

---

## Technology & Connectivity

### `technology_connectivity.csv`

| Column                       | Unit       | Type      | Description                           |
| ---------------------------- | ---------- | --------- | ------------------------------------- |
| district                     | none       | string    | District name.                        |
| period_start                 | date (ISO) | timestamp | Period start.                         |
| year                         | yyyy       | integer   | Year of observation.                  |
| replica_id                   | none       | integer   | Replica index.                        |
| internet_penetration_percent | percent    | float     | % of population with internet access. |
| mobile_penetration_percent   | percent    | float     | % of population with mobile access.   |

---

## Agriculture & Food Security

### `agriculture_food_security.csv`

| Column                     | Unit        | Type      | Description             |
| -------------------------- | ----------- | --------- | ----------------------- |
| district                   | none        | string    | District name.          |
| period_start               | date (ISO)  | timestamp | Period start.           |
| year                       | yyyy        | integer   | Year of observation.    |
| replica_id                 | none        | integer   | Replica index.          |
| crop_yield_ton_per_hectare | t/ha        | float     | Crop yield per hectare. |
| fertilizer_kg_per_hectare  | kg/ha       | integer   | Fertilizer rate.        |
| irrigated_area_km2         | km²         | integer   | Area under irrigation.  |
| total_production_tons      | metric tons | integer   | Gross crop production.  |

---

## District-Year Summary

### `district_year_summary.csv`

| Column                  | Unit        | Type    | Description                                      |
| ----------------------- | ----------- | ------- | ------------------------------------------------ |
| district                | none        | string  | District name.                                   |
| year                    | yyyy        | integer | Year of observation.                             |
| population_avg          | persons     | integer | Aggregated population.                           |
| density_per_km2         | persons/km² | float   | Aggregated density.                              |
| urban_percent           | percent     | float   | Aggregated urban percent.                        |
| water_mld               | MLD         | float   | Aggregated water demand.                         |
| electricity_mwh         | MWh         | integer | Aggregated electricity demand.                   |
| road_km                 | km          | integer | Aggregated road length.                          |
| schools                 | count       | integer | Aggregated schools.                              |
| hospital_beds           | beds        | integer | Aggregated hospital capacity.                    |
| avg_income_bdt          | BDT/year    | integer | Aggregated avg income.                           |
| employment_rate_percent | percent     | float   | Aggregated employment rate.                      |
| rainfall_mm             | mm          | integer | Aggregated rainfall.                             |
| flood_risk_score        | index       | integer | Aggregated flood risk.                           |
| migration_inflow        | persons     | integer | Aggregated inflow.                               |
| migration_outflow       | persons     | integer | Aggregated outflow.                              |
| net_migration           | persons     | integer | Aggregated net migration.                        |
| green_cover_percent     | percent     | float   | green_space_km2 / area_km2 \* 100.               |
| notes                   | none        | string  | Aggregation, replica collapse, imputation notes. |

---

## Notes

- Percent fields: 0–100 scale.
- `period_start` = canonical period column; `year` may aggregate pipelines.
- `replica_id` allows deterministic replication; downstream pipelines collapse replica rows.
- Merged/derived tables are pipeline outputs (not directly generated).
- Currency fields: BDT; `avg_income_bdt` = annual, `avg_rent_bdt` = monthly.
- Validate nulls, clamps, ranges before modeling.
