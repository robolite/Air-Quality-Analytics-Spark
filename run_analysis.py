
import argparse
import io
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType


# ============================================================
# AIR QUALITY ANALYTICS USING APACHE SPARK
# Dataset: UCI Air Quality
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT = ROOT / "outputs"
DATA = DATA_DIR / "AirQualityUCI.csv"

URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"

PREFERRED_POLLUTANTS = [
    "CO_GT",
    "NMHC_GT",
    "C6H6_GT",
    "NOx_GT",
    "NO2_GT",
]

DISPLAY_NAMES = {
    "CO_GT": "CO (mg/m³)",
    "NMHC_GT": "NMHC (µg/m³)",
    "C6H6_GT": "Benzene (µg/m³)",
    "NOx_GT": "NOx (ppb)",
    "NO2_GT": "NO2 (µg/m³)",
}

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March",
    4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September",
    10: "October", 11: "November", 12: "December",
}


# ============================================================
# DOWNLOAD DATASET IF REQUESTED
# ============================================================

def fetch_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading the official UCI Air Quality dataset...")

    with urllib.request.urlopen(URL, timeout=45) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        candidates = [
            name for name in archive.namelist()
            if name.lower().endswith("airqualityuci.csv")
        ]

        if not candidates:
            raise RuntimeError(
                "AirQualityUCI.csv was not found in the downloaded archive."
            )

        DATA.write_bytes(archive.read(candidates[0]))

    print(f"Dataset saved to: {DATA}")


# ============================================================
# HELPERS
# ============================================================

def normalize_column_name(name):
    name = name.replace("\ufeff", "").strip()
    name = (
        name.replace(".", "_")
        .replace("(", "_")
        .replace(")", "")
        .replace(" ", "_")
        .replace("/", "_")
    )

    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


def quoted_col(name):
    """Safely reference Spark column names containing special characters."""
    return F.col("`" + name.replace("`", "``") + "`")


def save_line_chart(x, y, title, xlabel, ylabel, filepath):
    """Save a simple line chart if usable values are available."""
    if x not in y.columns or y.empty:
        return

    plt.figure(figsize=(10, 5))

    for column in y.columns:
        if column != x:
            plt.plot(
                y[x],
                y[column],
                marker="o",
                linewidth=2,
                label=DISPLAY_NAMES.get(column, column),
            )

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=160, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze the UCI Air Quality dataset with Apache Spark."
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset from UCI if it is missing.",
    )

    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    if not DATA.exists() and args.download:
        fetch_dataset()

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\n"
            "Place AirQualityUCI.csv in the data folder, "
            "or run: python run_analysis.py --download"
        )

    spark = (
        SparkSession.builder
        .appName("AirQualityAnalytics")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        # ----------------------------------------------------
        # 1. READ SOURCE CSV
        # ----------------------------------------------------

        print("\n[1/7] Reading source dataset...")

        raw = (
            spark.read
            .option("header", True)
            .option("sep", ";")
            .option("inferSchema", False)
            .option("encoding", "UTF-8")
            .csv(str(DATA))
        )

        # Remove unwanted unnamed columns and safely select names.
        selected_columns = [
            quoted_col(c).alias(c)
            for c in raw.columns
            if c
            and not c.startswith("_")
            and not c.strip().startswith("Unnamed")
        ]

        raw = raw.select(*selected_columns)
        input_count = raw.count()

        # Normalize headers.
        for old_name in list(raw.columns):
            new_name = normalize_column_name(old_name)

            if old_name != new_name and new_name not in raw.columns:
                raw = raw.withColumnRenamed(old_name, new_name)

        # Remove columns that are completely empty.
        for column in list(raw.columns):
            if column not in ("Date", "Time"):
                has_value = (
                    raw.where(
                        F.col(column).isNotNull()
                        & (F.trim(F.col(column).cast("string")) != "")
                    )
                    .limit(1)
                    .count()
                )

                if has_value == 0:
                    raw = raw.drop(column)

        if "Date" not in raw.columns or "Time" not in raw.columns:
            raise RuntimeError(
                "Date or Time column was not found. Columns detected: "
                + ", ".join(raw.columns)
            )

        # ----------------------------------------------------
        # 2. CLEAN NUMERIC VALUES
        # ----------------------------------------------------

        print("[2/7] Cleaning numeric columns and missing values...")

        df = raw
        numeric_columns = [
            c for c in raw.columns
            if c not in ("Date", "Time")
        ]

        for column in numeric_columns:
            text_value = F.regexp_replace(
                F.trim(F.col(column).cast("string")),
                ",",
                ".",
            )

            number = text_value.cast(DoubleType())

            df = df.withColumn(
                column,
                F.when(
                    (number == -200) | (text_value == ""),
                    F.lit(None).cast(DoubleType()),
                ).otherwise(number),
            )

        pollutants = [
            c for c in PREFERRED_POLLUTANTS
            if c in df.columns
        ]

        if not pollutants:
            raise RuntimeError(
                "No expected pollutant columns were found. "
                "Detected columns: " + ", ".join(df.columns)
            )

        # ----------------------------------------------------
        # 3. PARSE TIMESTAMPS SAFELY
        # ----------------------------------------------------

        print("[3/7] Parsing date and time columns...")

        date_text = F.trim(F.col("Date").cast("string"))
        time_text = F.trim(F.col("Time").cast("string"))

        timestamp_text = F.when(
            date_text.isNotNull()
            & (date_text != "")
            & time_text.isNotNull()
            & (time_text != ""),
            F.concat(
                date_text,
                F.lit(" "),
                F.regexp_replace(time_text, r"\.", ":"),
            ),
        ).otherwise(F.lit(None).cast("string"))

        # try_to_timestamp safely returns NULL for invalid timestamps.
        df = (
            df.withColumn(
                "timestamp",
                F.try_to_timestamp(
                    timestamp_text,
                    F.lit("dd/MM/yyyy HH:mm:ss"),
                ),
            )
            .withColumn("hour", F.hour("timestamp"))
            .withColumn("month", F.month("timestamp"))
        )

        # ----------------------------------------------------
        # 4. REMOVE DUPLICATES
        # ----------------------------------------------------

        print("[4/7] Removing duplicate rows...")

        df = df.dropDuplicates()

        # Cache because the cleaned data is reused by multiple actions.
        df = df.cache()
        cleaned_count = df.count()

        print(f"Input rows: {input_count:,}")
        print(f"Rows after duplicate removal: {cleaned_count:,}")

        # ----------------------------------------------------
        # 5. OVERALL POLLUTANT SUMMARY
        # ----------------------------------------------------

        print("[5/7] Calculating pollutant summary...")

        summary_expressions = []

        for column in pollutants:
            summary_expressions.extend([
                F.avg(F.col(column)).alias(column + "_mean"),
                F.count(F.col(column)).alias(
                    column + "_valid_count"
                ),
                F.sum(
                    F.col(column).isNull().cast("long")
                ).alias(column + "_missing_count"),
                F.min(F.col(column)).alias(column + "_min"),
                F.max(F.col(column)).alias(column + "_max"),
                F.stddev(F.col(column)).alias(
                    column + "_stddev"
                ),
            ])

        summary_df = df.agg(*summary_expressions)
        summary_pd = summary_df.toPandas()

        summary_pd.to_csv(
            OUT / "pollutant_summary.csv",
            index=False,
        )

        # ----------------------------------------------------
        # 6. HOURLY AND MONTHLY AGGREGATIONS
        # ----------------------------------------------------

        print("[6/7] Calculating hourly and monthly patterns...")

        # Means and counts are calculated separately for each pollutant.
        # Missing measurements are not treated as zero.
        def aggregation_expressions():
            expressions = []

            for column in pollutants:
                expressions.extend([
                    F.avg(F.col(column)).alias(
                        column + "_mean"
                    ),
                    F.count(F.col(column)).alias(
                        column + "_valid_count"
                    ),
                ])

            return expressions

        hourly_spark = (
            df.where(F.col("hour").isNotNull())
            .groupBy("hour")
            .agg(*aggregation_expressions())
            .orderBy("hour")
        )

        monthly_spark = (
            df.where(F.col("month").isNotNull())
            .groupBy("month")
            .agg(*aggregation_expressions())
            .orderBy("month")
        )

        hourly_pd = hourly_spark.toPandas()
        monthly_pd = monthly_spark.toPandas()

        hourly_pd.to_csv(
            OUT / "hourly_pattern.csv",
            index=False,
        )

        monthly_pd.to_csv(
            OUT / "monthly_pattern.csv",
            index=False,
        )

        # Friendly month labels for the dashboard and report.
        monthly_pd["month_name"] = monthly_pd["month"].map(
            MONTH_NAMES
        )

        monthly_pd.to_csv(
            OUT / "monthly_pattern.csv",
            index=False,
        )

        # Export the cleaned dataset.
        cleaned_pd = df.toPandas()

        cleaned_pd.to_csv(
            OUT / "cleaned_air_quality.csv",
            index=False,
        )

        # ----------------------------------------------------
        # 7. GENERATE CHARTS
        # ----------------------------------------------------

        print("[7/7] Creating charts and run summary...")

        # Overall average chart.
        means = summary_pd.iloc[0]
        labels = []
        values = []

        for column in pollutants:
            value = means.get(column + "_mean")

            if pd.notna(value):
                labels.append(
                    DISPLAY_NAMES.get(column, column)
                )
                values.append(float(value))

        if values:
            plt.figure(figsize=(10, 5))
            plt.bar(labels, values)
            plt.title("Average Measured Pollutant Concentrations")
            plt.ylabel(
                "Average concentration "
                "(pollutant units differ)"
            )
            plt.xticks(rotation=18, ha="right")
            plt.tight_layout()

            plt.savefig(
                OUT / "pollutant_averages.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close()

        # Separate CO charts keep the scale readable.
        if "CO_GT_mean" in hourly_pd.columns:
            plt.figure(figsize=(9, 5))
            plt.plot(
                hourly_pd["hour"],
                hourly_pd["CO_GT_mean"],
                marker="o",
            )
            plt.title("Average CO Concentration by Hour")
            plt.xlabel("Hour of day")
            plt.ylabel("CO (mg/m³)")
            plt.xticks(range(0, 24, 2))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(
                OUT / "hourly_co_pattern.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close()

        if "CO_GT_mean" in monthly_pd.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(
                monthly_pd["month"],
                monthly_pd["CO_GT_mean"],
                marker="o",
            )
            plt.title("Average CO Concentration by Month")
            plt.xlabel("Month")
            plt.ylabel("CO (mg/m³)")
            plt.xticks(
                range(1, 13),
                [MONTH_NAMES[m][:3] for m in range(1, 13)],
            )
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(
                OUT / "monthly_co_pattern.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close()

        # ----------------------------------------------------
        # RUN SUMMARY
        # ----------------------------------------------------

        lines = [
            "AIR QUALITY ANALYTICS — RUN SUMMARY",
            "=" * 45,
            f"Input rows read: {input_count}",
            f"Rows after duplicate removal: {cleaned_count}",
            f"Pollutants analyzed: {', '.join(pollutants)}",
            f"Spark master: {spark.sparkContext.master}",
            "Execution mode: local Spark on one machine",
            "",
            "OVERALL POLLUTANT STATISTICS",
        ]

        for column in pollutants:
            label = DISPLAY_NAMES.get(column, column)

            mean = means.get(column + "_mean")
            valid = means.get(column + "_valid_count")
            missing = means.get(column + "_missing_count")

            lines.append(f"\n{label}")

            lines.append(
                f"  Mean: {float(mean):.4f}"
                if pd.notna(mean)
                else "  Mean: unavailable"
            )

            lines.append(f"  Valid readings: {int(valid):,}")
            lines.append(f"  Missing readings: {int(missing):,}")

        lines.extend([
            "",
            "DATA QUALITY NOTES",
            "- The source value -200 was converted to NULL.",
            "- Means exclude missing pollutant readings.",
            "- Valid counts are reported for each pollutant.",
            "- Invalid timestamps are excluded from time aggregations.",
            "- Hourly and monthly summaries describe historical averages.",
            "- Observed associations do not establish causation.",
            "",
            "OUTPUT FILES",
            "- cleaned_air_quality.csv",
            "- pollutant_summary.csv",
            "- hourly_pattern.csv",
            "- monthly_pattern.csv",
            "- pollutant_averages.png",
            "- hourly_co_pattern.png",
            "- monthly_co_pattern.png",
        ])

        (OUT / "run_summary.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        print("\n" + "\n".join(lines))
        print(f"\nAll outputs saved to: {OUT}")

        df.unpersist()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
