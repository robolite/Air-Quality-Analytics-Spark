
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Air Quality Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING — PRESERVING THE EXISTING DESIGN
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1450px;
        }

        [data-testid="stAppViewContainer"] {
            background: #f6f8fb;
        }

        [data-testid="stSidebar"] {
            background: #eef3f7;
        }

        h1, h2, h3 {
            letter-spacing: -0.025em;
        }

        .hero {
            padding: 1.5rem 1.7rem;
            border-radius: 18px;
            background: linear-gradient(
                120deg, #163b4d 0%, #226b69 100%
            );
            color: white;
            margin-bottom: 1.4rem;
        }

        .hero h1 {
            color: white;
            margin: 0 0 .45rem 0;
            font-size: 2.1rem;
        }

        .hero p {
            color: #e0f2f1;
            margin: 0;
            font-size: 1rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e4eaf0;
            padding: 1rem 1.1rem;
            border-radius: 14px;
            box-shadow: 0 2px 8px rgba(20,40,60,.035);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        .note {
            color: #526477;
            font-size: .9rem;
        }

        .section-note {
            color: #526477;
            font-size: .95rem;
            margin-bottom: .8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE LOCATIONS
# ============================================================

HERE = Path(__file__).resolve().parent

OUTPUT_DIRS = [
    HERE / "outputs",
    HERE / "output",
    HERE / "data" / "outputs",
    HERE,
    HERE.parent / "outputs",
]

EXPECTED = {
    "cleaned": "cleaned_air_quality.csv",
    "summary": "pollutant_summary.csv",
    "hourly": "hourly_pattern.csv",
    "monthly": "monthly_pattern.csv",
}

DISPLAY_NAMES = {
    "CO_GT": "Carbon Monoxide (CO)",
    "NMHC_GT": "Non-Methane Hydrocarbons (NMHC)",
    "C6H6_GT": "Benzene",
    "NOx_GT": "Nitrogen Oxides (NOx)",
    "NO2_GT": "Nitrogen Dioxide (NO₂)",
}

UNITS = {
    "CO_GT": "mg/m³",
    "NMHC_GT": "µg/m³",
    "C6H6_GT": "µg/m³",
    "NOx_GT": "ppb",
    "NO2_GT": "µg/m³",
}

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def locate(filename):
    for folder in OUTPUT_DIRS:
        candidate = folder / filename
        if candidate.exists():
            return candidate
    return None


# The file's modification time is part of the cache key.
# After run_analysis.py generates new CSVs, new data can be loaded.
@st.cache_data(show_spinner=False)
def read_csv_cached(path_string, modified_time):
    return pd.read_csv(path_string, low_memory=False)


def get_data():
    paths = {
        key: locate(filename)
        for key, filename in EXPECTED.items()
    }

    data = {}
    errors = []

    for key, path in paths.items():
        if path is None:
            continue

        try:
            data[key] = read_csv_cached(
                str(path),
                path.stat().st_mtime,
            )
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    return paths, data, errors


def pollutant_from_column(column):
    """Map a summary column such as CO_GT_mean to CO_GT."""
    for pollutant in DISPLAY_NAMES:
        if column.startswith(pollutant + "_"):
            return pollutant
    return None


def pollutant_label(pollutant):
    name = DISPLAY_NAMES.get(pollutant, pollutant)
    unit = UNITS.get(pollutant, "")
    return f"{name} ({unit})" if unit else name


def mean_columns(frame):
    """Return pollutant means only, excluding count columns."""
    if frame is None:
        return []

    return [
        column
        for column in frame.columns
        if column.endswith("_mean")
        and pollutant_from_column(column) is not None
    ]


def format_number(value, decimals=2):
    try:
        if pd.isna(value):
            return "Unavailable"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def get_pollutant_column(frame, pollutant, suffix="_mean"):
    column = pollutant + suffix
    return column if frame is not None and column in frame.columns else None


# ============================================================
# LOAD OUTPUTS
# ============================================================

paths, data, errors = get_data()

cleaned = data.get("cleaned")
summary = data.get("summary")
hourly = data.get("hourly")
monthly = data.get("monthly")

missing = [
    EXPECTED[key]
    for key, path in paths.items()
    if path is None
]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌿 Air Quality Analytics</h1>
        <p>
            Explore pollutant concentrations, daily variations,
            and monthly patterns using Apache Spark.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Dashboard")
    st.caption("Apache Spark · Environmental data exploration")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Pollutant Explorer",
            "Time Patterns",
            "Dataset",
        ],
        index=0,
    )

    st.divider()

    st.caption(
        "The dashboard reads CSV outputs generated by your "
        "Spark analysis."
    )

    if st.button("🔄 Refresh cached data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("Historical dataset · Not real-time monitoring")


# ============================================================
# WARNINGS
# ============================================================

if missing:
    st.warning(
        "Some output files were not found: "
        + ", ".join(missing)
        + ". Run run_analysis.py first."
    )

for error in errors:
    st.error(error)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("At a glance")

    if cleaned is not None:
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Records available",
            f"{len(cleaned):,}",
        )

        c2.metric(
            "Dataset columns",
            f"{len(cleaned.columns):,}",
        )

        c3.metric(
            "Missing output files",
            str(len(missing)),
        )
    else:
        st.info(
            "The cleaned dataset is unavailable. "
            "Run the Spark analysis first."
        )

    if summary is not None and not summary.empty:
        st.subheader("Average measured concentrations")

        available_means = [
            column for column in summary.columns
            if column.endswith("_mean")
            and pollutant_from_column(column) is not None
        ]

        if available_means:
            chart_rows = []

            for column in available_means:
                pollutant = pollutant_from_column(column)
                value = pd.to_numeric(
                    summary[column],
                    errors="coerce",
                ).iloc[0]

                if pd.notna(value):
                    chart_rows.append({
                        "Pollutant": pollutant_label(pollutant),
                        "Average": float(value),
                    })

            if chart_rows:
                chart_df = pd.DataFrame(chart_rows).set_index(
                    "Pollutant"
                )

                st.bar_chart(chart_df, height=320)

                st.caption(
                    "Concentration units differ between pollutants. "
                    "Compare values within the same pollutant, not "
                    "directly across different units."
                )

        with st.expander("View complete pollutant summary"):
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )

    st.markdown(
        '<p class="note">This is descriptive analysis of historical '
        'measurements, not a real-time monitoring or health-alert '
        'service.</p>',
        unsafe_allow_html=True,
    )


# ============================================================
# POLLUTANT EXPLORER
# ============================================================

elif page == "Pollutant Explorer":

    st.subheader("Explore individual pollutants")

    if cleaned is None:
        st.info("The cleaned dataset was not found.")
    else:
        available_pollutants = [
            column for column in DISPLAY_NAMES
            if column in cleaned.columns
        ]

        if not available_pollutants:
            st.info(
                "No recognized pollutant columns were found "
                "in the cleaned dataset."
            )
        else:
            selected = st.selectbox(
                "Select pollutant",
                available_pollutants,
                format_func=pollutant_label,
            )

            series = pd.to_numeric(
                cleaned[selected],
                errors="coerce",
            ).dropna()

            a, b, c = st.columns(3)

            a.metric(
                "Valid readings",
                f"{len(series):,}",
            )

            b.metric(
                "Mean",
                format_number(series.mean()),
            )

            c.metric(
                "Median",
                format_number(series.median()),
            )

            st.caption(
                f"Measurements in {UNITS.get(selected, 'recorded units')}."
            )

            # Use a sample for a responsive browser chart.
            chart_values = series.iloc[
                ::max(1, len(series) // 1500)
            ].reset_index(drop=True)

            chart_values.name = pollutant_label(selected)

            st.line_chart(
                chart_values,
                height=350,
            )

            with st.expander("View first 100 readings"):
                st.dataframe(
                    cleaned[[selected]].head(100),
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "⬇️ Download cleaned dataset",
                data=cleaned.to_csv(index=False).encode("utf-8"),
                file_name="cleaned_air_quality.csv",
                mime="text/csv",
            )


# ============================================================
# TIME PATTERNS — FIXED MONTHLY AND HOURLY CHARTS
# ============================================================

elif page == "Time Patterns":

    st.subheader("Pollutant patterns over time")

    st.markdown(
        '<p class="section-note">Select one pollutant at a time '
        'for a clearer view of its historical pattern.</p>',
        unsafe_allow_html=True,
    )

    hourly_tab, monthly_tab = st.tabs(
        ["🕒 Hourly Patterns", "🗓️ Monthly Patterns"]
    )

    # --------------------------------------------------------
    # HOURLY
    # --------------------------------------------------------

    with hourly_tab:

        if hourly is None or hourly.empty:
            st.info(
                "Hourly data is unavailable. "
                "Run run_analysis.py to generate it."
            )

        elif "hour" not in hourly.columns:
            st.error(
                "The hourly CSV does not contain an 'hour' column. "
                "Please regenerate the Spark outputs."
            )

        else:
            columns = mean_columns(hourly)

            if not columns:
                st.warning(
                    "No pollutant mean columns were found "
                    "in hourly_pattern.csv."
                )

            else:
                chosen_column = st.selectbox(
                    "Choose pollutant",
                    columns,
                    format_func=lambda c: pollutant_label(
                        pollutant_from_column(c)
                    ),
                    key="hourly_pollutant",
                )

                pollutant = pollutant_from_column(chosen_column)

                chart = hourly[["hour", chosen_column]].copy()

                chart["hour"] = pd.to_numeric(
                    chart["hour"],
                    errors="coerce",
                )

                chart[chosen_column] = pd.to_numeric(
                    chart[chosen_column],
                    errors="coerce",
                )

                chart = (
                    chart.dropna(subset=["hour"])
                    .sort_values("hour")
                    .set_index("hour")
                )

                chart.columns = [pollutant_label(pollutant)]

                st.line_chart(
                    chart,
                    height=360,
                )

                st.caption(
                    "Hourly averages summarize measurements collected "
                    "at the same hour across the dataset. They do not "
                    "represent every individual day."
                )

                count_col = pollutant + "_valid_count"

                if count_col in hourly.columns:
                    with st.expander(
                        "Inspect valid readings by hour"
                    ):
                        counts = hourly[
                            ["hour", count_col]
                        ].copy()

                        counts[count_col] = pd.to_numeric(
                            counts[count_col],
                            errors="coerce",
                        )

                        st.dataframe(
                            counts.sort_values("hour"),
                            use_container_width=True,
                            hide_index=True,
                        )

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    with monthly_tab:

        if monthly is None or monthly.empty:
            st.info(
                "Monthly data is unavailable. "
                "Run run_analysis.py to generate it."
            )

        elif "month" not in monthly.columns:
            st.error(
                "The monthly CSV does not contain a 'month' column. "
                "Please regenerate the Spark outputs."
            )

        else:
            columns = mean_columns(monthly)

            if not columns:
                st.warning(
                    "No pollutant mean columns were found "
                    "in monthly_pattern.csv."
                )

            else:
                chosen_column = st.selectbox(
                    "Choose pollutant",
                    columns,
                    format_func=lambda c: pollutant_label(
                        pollutant_from_column(c)
                    ),
                    key="monthly_pollutant",
                )

                pollutant = pollutant_from_column(chosen_column)

                chart = monthly[
                    ["month", chosen_column]
                ].copy()

                chart["month"] = pd.to_numeric(
                    chart["month"],
                    errors="coerce",
                )

                chart[chosen_column] = pd.to_numeric(
                    chart[chosen_column],
                    errors="coerce",
                )

                chart = (
                    chart.dropna(subset=["month"])
                    .sort_values("month")
                )

                # Month labels are independent of the pollutant values.
                chart["Month"] = chart["month"].map(
                    lambda value: MONTH_NAMES.get(
                        int(value),
                        str(value),
                    )
                    if pd.notna(value)
                    else "Unknown"
                )

                # Preserve the calendar order and keep missing values
                # as gaps; never convert missing readings into zero.
                chart = chart.drop(columns=["month"])
                chart = chart.set_index("Month")

                chart.columns = [pollutant_label(pollutant)]

                st.line_chart(
                    chart,
                    height=360,
                )

                valid_values = pd.to_numeric(
                    monthly[chosen_column],
                    errors="coerce",
                )

                valid_months = int(valid_values.notna().sum())
                missing_months = int(valid_values.isna().sum())

                a, b = st.columns(2)

                a.metric(
                    "Months with averages",
                    f"{valid_months} / 12",
                )

                b.metric(
                    "Months without averages",
                    f"{missing_months} / 12",
                )

                if missing_months:
                    st.warning(
                        f"{missing_months} month(s) have no average "
                        "for this pollutant. These are missing values, "
                        "not zero concentrations."
                    )

                st.caption(
                    "Monthly averages summarize available measurements "
                    "within each month. Differences may reflect data "
                    "availability as well as changes in concentrations."
                )

                count_col = pollutant + "_valid_count"

                if count_col in monthly.columns:
                    with st.expander(
                        "Inspect valid readings by month"
                    ):
                        counts = monthly[
                            ["month", count_col]
                        ].copy()

                        counts["month"] = pd.to_numeric(
                            counts["month"],
                            errors="coerce",
                        )

                        counts[count_col] = pd.to_numeric(
                            counts[count_col],
                            errors="coerce",
                        )

                        counts["Month"] = counts["month"].map(
                            lambda value: MONTH_NAMES.get(
                                int(value),
                                str(value),
                            )
                            if pd.notna(value)
                            else "Unknown"
                        )

                        st.dataframe(
                            counts[
                                ["Month", count_col]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.subheader("Dataset explorer")

    if cleaned is None:
        st.info("The cleaned dataset was not found.")
    else:
        st.caption(
            f"{len(cleaned):,} rows · "
            f"{len(cleaned.columns):,} columns"
        )

        st.dataframe(
            cleaned.head(500),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "The preview shows up to 500 rows for responsiveness. "
            "Download the complete cleaned dataset below."
        )

        st.download_button(
            "⬇️ Download full cleaned dataset",
            data=cleaned.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_air_quality.csv",
            mime="text/csv",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Air Quality Analytics · Apache Spark · Historical dataset analysis"
)
