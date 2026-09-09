# Unemployment Analysis with Python 

## Overview

This project **Unemployment Analysis with Python** for the EXPS Nexus Data Science Internship. It integrates two distinct datasets—national demographics/educational attainment and state-level regional metrics—to evaluate long-term unemployment patterns and the economic impact of the COVID-19 pandemic between 2010 and 2020.

## Datasets

* **`unemployment_data_us.csv`**: National-level data categorized by educational attainment (Primary School to Professional Degree) and demographic indicators (Race and Gender).
* **`unemployment_data_us_state.csv`**: Regional monthly state-by-state tracking metrics highlighting geographic unemployment variations.

## Prerequisites

Ensure Python 3.x is installed along with the required data science libraries. Install dependencies via terminal:

```bash
pip install pandas numpy matplotlib seaborn

```

## Project Structure

* `unemployment_analysis.py`: The core automation script handling data ingestion, cleaning, standardization, merging, and automated chart generation.
* `unemployment_data_us.csv`: Source file for national demographic and educational trends.
* `unemployment_data_us_state.csv`: Source file for state-level monthly metrics.
* `outputs_combined/`: Output directory containing generated visual artifacts and summary data tables.

## Running the Code

Execute the script from your project directory:

```bash
python unemployment_analysis.py

```

## Key Insights & Outputs

* **Education Protection:** Higher educational attainment (Professional and Associates Degrees) consistently sustained lower unemployment rates (~2-5%) relative to primary and high school levels during economic contractions.
* **COVID-19 Shock:** Quantifies the sharp nationwide unemployment spike during early 2020.
* **Regional Disparities:** Highlights state-specific baseline economic vulnerabilities to target localized policies.

All generated visual trend charts (`education_unemployment_trend.png`, `demographic_unemployment_trend.png`, `top_states_unemployment.png`) and summary tables (`top_states_summary.csv`) are automatically created and saved inside the `outputs_combined` folder upon script execution.
