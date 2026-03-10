import pandas as pd
from datetime import datetime


def run_feature_engineering(df: pd.DataFrame):

    print("\nRunning Feature Engineering Agent...")

    engineered_columns = []

    # ----------------------------
    # 1. Average Order Value
    # ----------------------------
    if "total_orders" in df.columns and "total_spent" in df.columns:

        df["avg_order_value"] = df["total_spent"] / df["total_orders"]
        df["avg_order_value"] = df["avg_order_value"].replace([float("inf")], 0)

        engineered_columns.append("avg_order_value")

    # ----------------------------
    # 2. Signup Date Features
    # ----------------------------
    if "signup_date" in df.columns:

        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

        df["signup_year"] = df["signup_date"].dt.year
        df["signup_month"] = df["signup_date"].dt.month

        engineered_columns.extend(["signup_year", "signup_month"])

    # ----------------------------
    # 3. Customer Tenure
    # ----------------------------
    if "signup_date" in df.columns:

        today = pd.Timestamp(datetime.today())

        df["customer_tenure_days"] = (today - df["signup_date"]).dt.days

        engineered_columns.append("customer_tenure_days")

    # ----------------------------
    # 4. High Value Customer Flag
    # ----------------------------
    if "total_spent" in df.columns:

        threshold = df["total_spent"].quantile(0.75)

        df["high_value_customer"] = df["total_spent"] > threshold

        engineered_columns.append("high_value_customer")

    # ----------------------------
    # Summary
    # ----------------------------
    print("\nFEATURE ENGINEERING SUMMARY")
    print("----------------------------")
    print("New Features Created:", engineered_columns)

    return df