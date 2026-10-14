# Air Quality Analytics Using Apache Spark

## Project goal
Analyze the UCI Air Quality time-series dataset with PySpark: clean sentinel missing values, parse dates/times, compute pollutant summaries and hourly/monthly patterns, and export tables/charts.

## Dataset
UCI Machine Learning Repository — Air Quality (Dataset ID 360), DOI: https://doi.org/10.24432/C59K5F
Official page: https://archive.ics.uci.edu/dataset/360/air+quality
The source describes 9,358 hourly records and 15 attributes, collected in an Italian city from March 2004 to February 2005. The source marks missing values with -200. Research use only; do not use commercially.

## Setup (Windows PowerShell)
1. Install Python 3.10+.
2. In this folder, run:
   `py -m venv .venv`
   ` .\.venv\Scripts\Activate.ps1`
   `python -m pip install --upgrade pip`
   `pip install pyspark pandas matplotlib openpyxl`
3. Download the dataset ZIP from the official UCI page, extract `AirQualityUCI.csv`, and place it in `data/AirQualityUCI.csv`.
   Alternative: run `python run_analysis.py --download` to attempt automatic download.
4. Run: `python run_analysis.py`
5. Review `outputs/` for cleaned data, analysis tables, charts, and a run summary.

## Important honesty note
This is a Spark-based analytics prototype running locally (`local[*]`), not a multi-machine cluster. The dataset is useful for demonstrating Spark DataFrame transformations and aggregation, but its ~9k rows are not itself a massive dataset. The pipeline can be scaled to larger air-quality archives or partitioned multi-year data in a cluster.

## Outputs
- `cleaned_air_quality.csv`
- `pollutant_summary.csv`
- `hourly_pattern.csv`
- `monthly_pattern.csv`
- `run_summary.txt`
- `pollutant_averages.png`
- `hourly_co_pattern.png`
- `monthly_co_pattern.png`
