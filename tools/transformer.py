import pandas as pd
import numpy as np


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
    for item in plan.get("column_actions", []):

        col = item.get("column")
        if not col or col not in df.columns:
            continue

        methods = item.get("actions", [])
        if "action" in item:  # fallback for older format
            methods = [item["action"]]

        for method in methods:
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

            elif method == "fill_unknown":
                df[col] = df[col].fillna("Unknown")
                changes["columns_modified"].append(col)

            elif method == "replace_negatives_with_null":
                # Only apply to numeric representations
                numeric_col = pd.to_numeric(df[col], errors="coerce")
                df.loc[numeric_col < 0, col] = np.nan
                changes["columns_modified"].append(col)

            elif method == "coerce_datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
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

            elif method == "drop_column":
                df = df.drop(columns=[col])
                changes["columns_modified"].append(col)

    # ==========================================
    # AUTOMATIC GENERALIZED CLEANING MODULE
    # ==========================================
    # Hard-coded rules that apply unconditionally 
    # to every dataset, freeing up LLM bandwidth.

    for col in df.columns:
        # 1. Automatic Text Normalization
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            mask = df[col].notna()
            # Only apply to actual strings (skip mixed data types if they fail)
            try:
                df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip().str.title()
                if col not in changes["columns_modified"]:
                    changes["columns_modified"].append(col)
            except Exception:
                pass

        # 2. Automatic Outlier Clipping
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            # Avoid clipping ID columns or small categorical integers
            if col.lower().endswith("id") or col.lower() == "id" or df[col].nunique() <= 5:
                continue
                
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower=lower, upper=upper)
                if col not in changes["columns_modified"]:
                    changes["columns_modified"].append(col)

        elif method == "clip_outliers":

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            df[col] = df[col].clip(lower, upper)
            changes["columns_modified"].append(col)

        elif method == "normalize_text":
            df[col] = df[col].astype(str).str.strip().str.lower()
            changes["columns_modified"].append(col)
        
        elif method == "datetime_imputation":
            df[col] = pd.to_datetime(df[col], errors="coerce")
            filled = df[col].isna().sum()
            df[col] = df[col].fillna(df[col].mode()[0])
            changes["nulls_filled"] += int(filled)
            changes["columns_modified"].append(col)

    return df, changes
