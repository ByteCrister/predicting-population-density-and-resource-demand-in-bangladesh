# Predicting Population Density and Resource Demand in Bangladesh — README

## Project

This repository generates synthetic datasets and example pipelines for predicting population density and resource demand in Bangladesh. Use it to prototype ML models for urban planning, resource allocation, and climate resilience.

## What this repo contains

- scripts/generate_data.py (generator script producing 13 CSVs)
- DATA_DICTIONARY.md — authoritative column definitions, units, and types
- README.md — this file

## Summary of datasets produced by the generator

| File                          | Primary units (summary)                                    | Short description                                                                |
| ----------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `population_demographics.csv` | persons; km²; percent                                      | Population totals, age shares, urban/rural split by district & year              |
| `population_density.csv`      | persons; km²                                               | Compact population + density view                                                |
| `resource_demand.csv`         | MLD (water); MWh (electricity); housing units; tons (food) | Daily water demand, annual electricity demand proxy, housing demand, food demand |
| `urban_infrastructure.csv`    | km (road); count (schools,hospitals); km² (green)          | Road network, facilities, green/open space                                       |
| `migration_trends.csv`        | persons/year                                               | Origin → destination migration flows                                             |
| `economic_indicators.csv`     | BDT/year; percent                                          | Avg income (BDT/yr), employment %, poverty %, sector shares %                    |
| `climate_impact.csv`          | °C; mm; index; counts                                      | Avg temp (°C), rainfall (mm), flood risk (1–10), cyclone events, drought index   |
| `land_use_zoning.csv`         | km²                                                        | Residential, commercial, industrial, agricultural area allocations               |
| `transportation.csv`          | count; km; minutes; vehicles                               | Bus routes, rail km, avg commute (min), vehicle counts                           |
| `health_sanitation.csv`       | beds; count; percent                                       | Hospital beds, clinics, sanitation %, waste collection %                         |
| `education.csv`               | count; percent; ratio                                      | Schools, literacy %, student-teacher ratio                                       |
| `energy_usage.csv`            | MWh; percent                                               | Residential/commercial/industrial MWh (annual), renewable %                      |
| `housing.csv`                 | housing units; BDT/month; percent                          | Housing stock, avg rent (BDT/month), vacancy %, slum %                           |
| `environmental_stress.csv`    | index; days; events; percent                               | AQ index, heatwave days, flood events, green cover %                             |

See DATA_DICTIONARY.md for column-level units, types and descriptions.

---

## Conventions (authoritative)

- Percent fields use 0–100 scale.
- Currency is BDT (Bangladeshi Taka): avg_income_bdt = annual; avg_rent_bdt = monthly.
- Area: km². Yield: t/ha. Rainfall: mm. Temperature: °C.
- Water: MLD = million liters per day. Energy: MWh (annual aggregate).
- Year fields use integer year (yyyy).
- All counts and areas are non-negative.
- DATA_DICTIONARY.md is the single source of truth for column-level units, types and short descriptions.

---

## Quick start (run locally)

1. Clone

```bash
git clone https://github.com/ByteCrister/predicting-population-density-and-resource-demand-in-bangladesh.git
cd predicting-population-density-and-resource-demand-in-bangladesh
```
