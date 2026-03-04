import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load CSV dataset into pandas DataFrame.
    """
    return pd.read_csv(file_path)


# -----------------------------------------
# COLUMN TYPE DETECTION
# -----------------------------------------
def detect_column_type(series: pd.Series) -> str:

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Try datetime detection
    try:
        sample = series.dropna().astype(str).head(10)
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().sum() / len(sample) > 0.7:
            return "datetime"
    except Exception:
        pass

    unique_ratio = series.nunique() / len(series)

    if unique_ratio < 10:
        return "categorical"

    return "text"


# -----------------------------------------
# OUTLIER DETECTION (IQR METHOD)
# -----------------------------------------
def detect_outliers(series: pd.Series) -> int:

    if len(series) == 0:
        return 0

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = series[(series < lower) | (series > upper)]

    return int(len(outliers))


# -----------------------------------------
# INVALID DATE DETECTION
# -----------------------------------------
def detect_invalid_dates(series: pd.Series) -> int:

    parsed = pd.to_datetime(series, errors="coerce")

    invalid = parsed.isna() & series.notna()

    return int(invalid.sum())


# -----------------------------------------
# DATASET HEALTH SCORE
# -----------------------------------------
def calculate_health_score(df: pd.DataFrame, duplicate_rows: int, columns_info: list):

    score = 100

    # Duplicate penalty
    if duplicate_rows > 0:
        score -= 10

    for col in columns_info:

        if col["null_percentage"] > 50:
            score -= 15

        if col["null_percentage"] > 20:
            score -= 5

        if "outliers_detected" in col["issues"]:
            score -= 5

        if "invalid_dates" in col["issues"]:
            score -= 5

        if "single_value_column" in col["issues"]:
            score -= 5

        if "high_skew" in col["issues"]:
            score -= 5

    return max(score, 0)


# -----------------------------------------
# MAIN PROFILER
# -----------------------------------------
def generate_profile(df: pd.DataFrame) -> dict:

    duplicate_rows = int(df.duplicated().sum())

    profile = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_rows": duplicate_rows,
        "dataset_issues": [],
        "columns": []
    }

    for col in df.columns:

        col_data = df[col]

        semantic_type = detect_column_type(col_data)

        column_info = {
            "name": col,
            "dtype": str(col_data.dtype),
            "semantic_type": semantic_type,
            "null_count": int(col_data.isnull().sum()),
            "null_percentage": round(
                (col_data.isnull().sum() / len(df)) * 100, 2
            ),
            "unique_count": int(col_data.nunique()),
            "sample_values": col_data.dropna().unique()[:5].tolist(),
            "issues": []
        }

        # -----------------------------
        # NUMERIC STATISTICS
        # -----------------------------
        if semantic_type == "numeric":

            column_info.update({
                "min": float(col_data.min()) if not col_data.isnull().all() else None,
                "max": float(col_data.max()) if not col_data.isnull().all() else None,
                "mean": float(col_data.mean()) if not col_data.isnull().all() else None,
                "median": float(col_data.median()) if not col_data.isnull().all() else None
            })

            # Outlier detection
            outlier_count = detect_outliers(col_data.dropna())

            if outlier_count > 0:
                column_info["issues"].append("outliers_detected")
                column_info["outlier_count"] = outlier_count

            # Skew detection
            try:
                skew = col_data.skew()
                if abs(skew) > 2:
                    column_info["issues"].append("high_skew")
                    column_info["skew"] = float(skew)
            except Exception:
                pass

        # -----------------------------
        # DATE VALIDATION
        # -----------------------------
        if semantic_type == "datetime":

            invalid_dates = detect_invalid_dates(col_data)

            if invalid_dates > 0:
                column_info["issues"].append("invalid_dates")
                column_info["invalid_date_count"] = invalid_dates

        # -----------------------------
        # HIGH MISSING VALUES
        # -----------------------------
        if column_info["null_percentage"] > 20:
            column_info["issues"].append("high_missing_values")

        # -----------------------------
        # SINGLE VALUE COLUMN
        # -----------------------------
        if column_info["unique_count"] <= 1:
            column_info["issues"].append("single_value_column")

        profile["columns"].append(column_info)

    # --------------------------------
    # DATASET LEVEL ISSUES
    # --------------------------------

    if duplicate_rows > 0:
        profile["dataset_issues"].append("duplicate_rows_present")

    high_missing_cols = [
        col["name"] for col in profile["columns"]
        if col["null_percentage"] > 50
    ]

    if high_missing_cols:
        profile["dataset_issues"].append({
            "columns_with_over_50_percent_missing": high_missing_cols
        })

    single_value_cols = [
        col["name"] for col in profile["columns"]
        if "single_value_column" in col["issues"]
    ]

    if single_value_cols:
        profile["dataset_issues"].append({
            "single_value_columns": single_value_cols
        })

    # --------------------------------
    # DATASET HEALTH SCORE
    # --------------------------------

    profile["dataset_health_score"] = calculate_health_score(
        df,
        duplicate_rows,
        profile["columns"]
    )

    return profile