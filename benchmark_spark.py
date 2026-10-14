"""
Local Spark scalability experiment for Air Quality Analytics.
IMPORTANT: scaled rows are synthetic replicas of the cleaned dataset.
This measures local Spark partitioning/processing only; it does NOT measure a cluster.
Run from the project root with the project's Python environment.
"""
from pathlib import Path
import csv
import statistics
import time
from datetime import datetime
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType

Project = 57;

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prefer the existing cleaned output. Fall back to common raw dataset locations.
CANDIDATES = [
    OUTPUT_DIR / "cleaned_air_quality.csv",
    PROJECT_ROOT / "data" / "AirQualityUCI.csv",
    PROJECT_ROOT / "data" / "air_quality.csv",
    PROJECT_ROOT / "AirQualityUCI.csv",
]
INPUT_FILE = next((p for p in CANDIDATES if p.exists()), None)
if INPUT_FILE is None:
    raise FileNotFoundError(
        "Could not find outputs/cleaned_air_quality.csv or a common raw CSV path. "
        "Edit CANDIDATES in benchmark_spark.py to point to your dataset."
    )

SCALE_FACTORS = [1, 10, 50]
PARTITION_COUNTS = [1, 2, 4]
TRIALS = 3

spark = (
    SparkSession.builder
    .appName("AirQualityLocalScalabilityBenchmark")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.adaptive.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

def normalize(name):
    return "".join(ch.lower() for ch in name if ch.isalnum())

print(f"Input file: {INPUT_FILE}")
print(f"Spark version: {spark.version}")
print(f"Master: {spark.sparkContext.master}")
print(f"Default parallelism: {spark.sparkContext.defaultParallelism}")

# Read the data and detect useful grouping columns.
base = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .option("sep", ";")
    .csv(str(INPUT_FILE))
)
# If a semicolon read produces only one column, retry comma-separated CSV.
if len(base.columns) == 1:
    base = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("sep", ",")
        .csv(str(INPUT_FILE))
    )

column_lookup = {normalize(c): c for c in base.columns}
hour_col = next((column_lookup[k] for k in ("hour", "hourofday") if k in column_lookup), None)
month_col = next((column_lookup[k] for k in ("month", "monthname") if k in column_lookup), None)

# If hour/month are not already available, derive them from a recognizable date column.
date_col = next((column_lookup[k] for k in ("datetime", "timestamp", "date", "datetimect") if k in column_lookup), None)
if date_col:
    if hour_col is None:
        base = base.withColumn("_benchmark_hour", F.hour(F.to_timestamp(F.col(date_col))))
        hour_col = "_benchmark_hour"
    if month_col is None:
        base = base.withColumn("_benchmark_month", F.month(F.to_timestamp(F.col(date_col))))
        month_col = "_benchmark_month"

numeric_cols = [f.name for f in base.schema.fields if isinstance(f.dataType, NumericType)]
# Exclude grouping fields from metric aggregation; keep only actual numeric measures.
metric_cols = [c for c in numeric_cols if c not in {hour_col, month_col}]
if not metric_cols:
    raise ValueError(
        f"No numeric measure columns detected. Columns found: {base.columns}. "
        "Check that the selected CSV is the cleaned dataset."
    )

# Convert common missing-value sentinel to null where numeric columns contain it.
for c in metric_cols:
    base = base.withColumn(c, F.when(F.col(c) == -200, F.lit(None)).otherwise(F.col(c)))

base_count = base.count()
print(f"Base rows: {base_count:,}")
print(f"Columns: {base.columns}")
print(f"Numeric measures benchmarked: {metric_cols}")
if not hour_col and not month_col:
    print("Note: no hour/month grouping column found; benchmark will aggregate numeric measures globally.")

# Build a readable grouping expression from whichever time dimensions exist.
group_cols = [c for c in (month_col, hour_col) if c]
if not group_cols:
    group_cols = []

results = []
plans = []

for scale in SCALE_FACTORS:
    # Synthetic scaling: repeat every original row using a replica id.
    # replica id is removed before aggregation; no claim is made that these are real observations.
    if scale == 1:
        scaled = base
    else:
        replicas = spark.range(scale).withColumnRenamed("id", "_replica_id")
        scaled = base.crossJoin(replicas).drop("_replica_id")

    expected_rows = base_count * scale
    print(f"\n--- Scale {scale}x: expected {expected_rows:,} rows (synthetic replicas when >1) ---")

    for partitions in PARTITION_COUNTS:
        prepared = scaled.repartition(partitions)
        # Save one physical plan for architecture evidence.
        if scale == 1 and partitions == PARTITION_COUNTS[0]:
            plan_path = OUTPUT_DIR / "spark_physical_plan.txt"
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write("Air Quality Spark benchmark physical plan\n")
                f.write(f"Input: {INPUT_FILE}\nScale factor: {scale}\nRequested partitions: {partitions}\n\n")
                f.write("Spark physical execution plan (executedPlan):\n")
                plan_query = prepared.groupBy(*group_cols).agg(
                    *[F.avg(F.col(c)).alias(f"avg_{normalize(c)}") for c in metric_cols]
                )
                f.write(plan_query._jdf.queryExecution().executedPlan().toString())
                f.write("\n\nSpark formatted explain is also printed to the terminal below.\n")
                plan_query.explain(mode="formatted")
            plans.append(str(plan_path))

        # Warm-up once for this scale/partition combination, then time repeated actions.
        query = prepared.groupBy(*group_cols).agg(
            *[F.avg(F.col(c)).alias(f"avg_{normalize(c)}") for c in metric_cols]
        )
        query.collect()
        timings = []
        for trial in range(1, TRIALS + 1):
            start = time.perf_counter()
            query.collect()  # result is small; forces the aggregation to execute
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

        results.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "input_file": INPUT_FILE.name,
            "scale_factor": scale,
            "dataset_type": "original" if scale == 1 else "synthetically_replicated",
            "input_rows": expected_rows,
            "requested_partitions": partitions,
            "actual_partition_count": prepared.rdd.getNumPartitions(),
            "trial_count": TRIALS,
            "median_seconds": round(statistics.median(timings), 6),
            "min_seconds": round(min(timings), 6),
            "max_seconds": round(max(timings), 6),
            "spark_master": spark.sparkContext.master,
            "spark_version": spark.version,
            "default_parallelism": spark.sparkContext.defaultParallelism,
            "group_columns": ",".join(group_cols) if group_cols else "(global)",
            "measure_columns": ",".join(metric_cols),
        })
        print(
            f"partitions={partitions}, actual={prepared.rdd.getNumPartitions()}, "
            f"median={statistics.median(timings):.4f}s "
            f"(trials: {', '.join(f'{x:.4f}' for x in timings)})"
        )

csv_path = OUTPUT_DIR / "spark_scalability_benchmark.csv"
fieldnames = list(results[0].keys())
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

readme_path = OUTPUT_DIR / "benchmark_notes.txt"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("LOCAL SPARK BENCHMARK — INTERPRETATION NOTES\n")
    f.write("============================================\n")
    f.write(f"Input: {INPUT_FILE}\nBase rows: {base_count}\n")
    f.write(f"Scale factors: {SCALE_FACTORS}\nPartitions requested: {PARTITION_COUNTS}\nTrials per setting: {TRIALS}\n")
    f.write(f"Spark master: {spark.sparkContext.master}\nSpark version: {spark.version}\n")
    f.write(f"Physical plan file: {plans[0] if plans else 'not captured'}\n\n")
    f.write("Scale factors above 1 repeat existing rows to create controlled synthetic workloads.\n")
    f.write("They are not new real-world air-quality observations and must be labelled synthetic.\n")
    f.write("All runs use local[*] on one laptop. Results do not establish multi-node cluster performance.\n")
    f.write("Timings can vary with JVM warm-up, OS activity, caching, memory, and disk state.\n")
    f.write("Use medians, report all configurations, and do not claim Spark is faster unless measurements support it.\n")

print(f"\nSaved benchmark results: {csv_path}")
print(f"Saved physical plan: {OUTPUT_DIR / 'spark_physical_plan.txt'}")
print(f"Saved interpretation notes: {readme_path}")
spark.stop()
