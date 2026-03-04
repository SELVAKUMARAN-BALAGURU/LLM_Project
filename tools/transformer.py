import pandas as pd


def apply_cleaning_plan(df: pd.DataFrame, plan: dict):
    """
    Apply cleaning actions suggested by the planner agent.
    Returns cleaned dataframe and summary of changes.
    """

    changes = {
        "rows_removed": 0,
        "nulls_filled": 0,
        "columns_modified": []
    }

    # Remove duplicates
    if plan.get("drop_duplicates"):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        changes["rows_removed"] = before - after

    # Column specific actions
    for action in plan.get("column_actions", []):

        col = action["column"]
        method = action["action"]

        if col not in df.columns:
            continue

        if method == "median_imputation":
            filled = df[col].isna().sum()
            df[col] = df[col].fillna(df[col].median())
            changes["nulls_filled"] += int(filled)
            changes["columns_modified"].append(col)

        elif method == "mean_imputation":
            filled = df[col].isna().sum()
            df[col] = df[col].fillna(df[col].mean())
            changes["nulls_filled"] += int(filled)
            changes["columns_modified"].append(col)

        elif method == "mode_imputation":
            filled = df[col].isna().sum()
            df[col] = df[col].fillna(df[col].mode()[0])
            changes["nulls_filled"] += int(filled)
            changes["columns_modified"].append(col)

        elif method == "normalize_text":
            df[col] = df[col].astype(str).str.strip().str.lower()
            changes["columns_modified"].append(col)

        elif method == "remove_outliers":

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            before = len(df)

            df = df[(df[col] >= lower) & (df[col] <= upper)]

            removed = before - len(df)

            changes["rows_removed"] += removed

    return df, changes