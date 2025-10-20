# Data Dictionary — Bangladesh Urban Growth & Resource Demand (CSV schema)

## population_demographics.csv

- **district** — Unit: none; Type: string; Description: Administrative district name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **population** — Unit: persons; Type: integer; Description: Total residents in the district in that year.
- **area_km2** — Unit: km²; Type: float; Description: Land area of the district used for density calculations.
- **density_per_km2** — Unit: persons per km²; Type: float; Description: population / area_km2.
- **children_percent** — Unit: percent; Type: float; Description: % of population aged 0–14 (synthetic estimate).
- **working_age_percent** — Unit: percent; Type: float; Description: % of population aged 15–64 (synthetic estimate).
- **elderly_percent** — Unit: percent; Type: float; Description: % of population aged 65+ (synthetic estimate).
- **urban_percent** — Unit: percent; Type: float; Description: % of population living in urban areas.
- **urban_population** — Unit: persons; Type: integer; Description: Count of residents classified as urban.
- **rural_population** — Unit: persons; Type: integer; Description: Count of residents classified as rural.

## population_density.csv

- **district** — Unit: none; Type: string; Description: District name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **population** — Unit: persons; Type: integer; Description: Total residents.
- **area_km2** — Unit: km²; Type: float; Description: District land area.
- **density_per_km2** — Unit: persons per km²; Type: float; Description: population / area_km2.

## resource_demand.csv

- **district** — Unit: none; Type: string; Description: District name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **water_mld** — Unit: million liters per day (MLD); Type: float; Description: Estimated daily water demand aggregated for the district.
- **electricity_mwh** — Unit: megawatt-hours (MWh) per year; Type: integer; Description: Annual electricity demand proxy for the district.
- **housing_units_demand** — Unit: housing units per year; Type: integer; Description: Net new housing units required (annual synthetic demand).
- **food_demand_tons** — Unit: metric tons per year; Type: integer; Description: Annual food consumption demand.

## urban_infrastructure.csv

- **district** — Unit: none; Type: string; Description: District name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **road_km** — Unit: kilometers; Type: integer; Description: Total road network length in the district.
- **schools** — Unit: count; Type: integer; Description: Number of schools.
- **hospitals** — Unit: count; Type: integer; Description: Number of hospitals.
- **green_space_km2** — Unit: km²; Type: float; Description: Area of green/open public space.

## land_use_zoning.csv

- **district** — Unit: none; Type: string; Description: District name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **residential_area_km2** — Unit: km²; Type: float; Description: Area zoned/used for residential purposes.
- **commercial_area_km2** — Unit: km²; Type: float; Description: Area zoned/used for commercial purposes.
- **industrial_area_km2** — Unit: km²; Type: float; Description: Area zoned/used for industrial purposes.
- **agricultural_area_km2** — Unit: km²; Type: float; Description: Area used for agriculture.

## climate_impact.csv

- **district** — Unit: none; Type: string; Description: District name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **avg_temp_c** — Unit: degrees Celsius (°C); Type: float; Description: Annual average temperature.
- **rainfall_mm** — Unit: millimeters (mm); Type: integer; Description: Annual total rainfall.
- **flood_risk_score** — Unit: index 1–10; Type: integer; Description: Synthetic flood risk index (higher = greater risk).
- **cyclone_events** — Unit: events per year; Type: integer; Description: Count of cyclone events in the year.
- **drought_index** — Unit: synthetic index (0–10); Type: float; Description: Drought severity proxy (higher = drier/stronger stress).

## environmental_stress.csv

- **district** — Unit: none; Type: string; Description: District name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **air_quality_index** — Unit: index (higher = worse); Type: integer or float; Description: Synthetic AQ proxy.
- **heatwave_days** — Unit: days per year; Type: integer; Description: Number of heatwave days.
- **flood_events** — Unit: events per year; Type: integer; Description: Count of flood events derived from climate variables.
- **green_cover_percent** — Unit: percent; Type: float; Description: (green_space_km2 / area_km2) \* 100.

## economic_indicators.csv

- **district** — Unit: none; Type: string; Description: District name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **avg_income_bdt** — Unit: Bangladeshi Taka (BDT) per year; Type: integer; Description: Average annual per-capita income proxy.
- **employment_rate_percent** — Unit: percent; Type: float; Description: % of working-age population employed (synthetic estimate).
- **poverty_rate_percent** — Unit: percent; Type: float; Description: % of population below synthetic poverty threshold.
- **agri_share_percent** — Unit: percent; Type: float; Description: % share of economic activity in agriculture.
- **industry_share_percent** — Unit: percent; Type: float; Description: % share of economic activity in industry.
- **services_share_percent** — Unit: percent; Type: float; Description: % share of economic activity in services.

## migration_trends.csv

- **from_district** — Unit: none; Type: string; Description: Origin district name.
- **to_district** — Unit: none; Type: string; Description: Destination district name.
- **year** — Unit: yyyy; Type: integer; Description: Year of migration flow.
- **migrants** — Unit: persons per year; Type: integer; Description: Number of migrants moving from origin to destination in that year (synthetic flow).

## health_sanitation.csv

- **district** — Unit: none; Type: string; Description: District name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **hospital_beds** — Unit: beds; Type: integer; Description: Total hospital bed capacity.
- **clinics** — Unit: count; Type: integer; Description: Number of clinics.
- **sanitation_coverage_percent** — Unit: percent; Type: float; Description: % of population with basic sanitation access.
- **waste_collection_percent** — Unit: percent; Type: float; Description: % of population/area covered by waste collection services.

## education.csv

- **district** — Unit: none; Type: string; Description: District name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **schools** — Unit: count; Type: integer; Description: Number of schools.
- **literacy_rate_percent** — Unit: percent; Type: float; Description: % of population classified as literate.
- **student_teacher_ratio** — Unit: students per teacher; Type: float; Description: Average number of students per teacher (proxy for class size).

## transportation.csv

- **district** — Unit: none; Type: string; Description: District name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **bus_routes** — Unit: count; Type: integer; Description: Number of bus routes serving urban areas.
- **rail_km** — Unit: kilometers; Type: integer; Description: Length of rail network in the district.
- **avg_commute_time_min** — Unit: minutes; Type: integer; Description: Average one-way commute time.
- **vehicle_count** — Unit: vehicles; Type: integer; Description: Registered vehicles proxy in the district.

## housing.csv

- **district** — Unit: none; Type: string; Description: District name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **housing_units** — Unit: housing units; Type: integer; Description: Estimated total housing stock.
- **avg_rent_bdt** — Unit: BDT per month; Type: integer; Description: Typical monthly rent (synthetic average).
- **vacancy_rate_percent** — Unit: percent; Type: float; Description: % of housing units vacant.
- **slum_population_percent** — Unit: percent; Type: float; Description: % of district population living in informal/slum housing.

## technology_connectivity.csv

- **district** — Unit: none; Type: string; Description: District name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **internet_penetration_percent** — Unit: percent; Type: float; Description: % of population with internet access.
- **mobile_penetration_percent** — Unit: percent; Type: float; Description: % of population with mobile access/subscription proxy.

## agriculture_food_security.csv

- **district** — Unit: none; Type: string; Description: District name.
- **year** — Unit: yyyy; Type: integer; Description: Year of observation.
- **crop_yield_ton_per_hectare** — Unit: metric tons per hectare (t/ha); Type: float; Description: Crop yield per hectare.
- **fertilizer_kg_per_hectare** — Unit: kg per hectare; Type: integer; Description: Fertilizer application rate.
- **irrigated_area_km2** — Unit: km²; Type: integer; Description: Area under irrigation within the district.
- **total_production_tons** — Unit: metric tons per year; Type: integer; Description: Gross crop production for the district.

# Notes

- Percent fields are expressed on a 0–100 scale.
- Currency fields use BDT; clarify whether values are monthly or annual in downstream schemas (avg_income_bdt is annual; avg_rent_bdt is monthly).
- Types are canonical expected types; downstream pipelines should validate nulls and ranges before modeling.
