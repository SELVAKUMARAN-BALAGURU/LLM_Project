import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df):

    print("\nRunning EDA Agent...")

    output_dir = "reports"
    plot_dir = os.path.join(output_dir, "plots")

    os.makedirs(plot_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "bool"]).columns

    insights = []

    # -----------------------------------
    # 1. Statistical Summary
    # -----------------------------------

    stats = df.describe(include="all")

    stats.to_csv(os.path.join(output_dir, "dataset_statistics.csv"))

    # -----------------------------------
    # 2. Numeric Distributions
    # -----------------------------------

    for col in numeric_cols:

        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)

        plt.title(f"Distribution of {col}")

        plt.savefig(f"{plot_dir}/{col}_distribution.png")
        plt.close()

        skew = df[col].skew()

        if skew > 1:
            insights.append(f"{col} is highly right skewed.")
        elif skew < -1:
            insights.append(f"{col} is highly left skewed.")

    # -----------------------------------
    # 3. Correlation Analysis
    # -----------------------------------

    if len(numeric_cols) > 1:

        corr = df[numeric_cols].corr()

        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")

        plt.title("Correlation Heatmap")

        plt.savefig(f"{plot_dir}/correlation_heatmap.png")
        plt.close()

        # Detect strong correlations
        for col1 in corr.columns:
            for col2 in corr.columns:

                if col1 != col2 and abs(corr.loc[col1, col2]) > 0.7:

                    insights.append(
                        f"Strong correlation detected between {col1} and {col2}"
                    )

    # -----------------------------------
    # 4. Categorical Analysis
    # -----------------------------------

    for col in categorical_cols:

        plt.figure()

        df[col].value_counts().plot(kind="bar")

        plt.title(f"{col} Distribution")

        plt.savefig(f"{plot_dir}/{col}_counts.png")

        plt.close()

        if df[col].nunique() < 10:

            top_value = df[col].value_counts().idxmax()

            insights.append(f"Most common value in {col} is {top_value}")

    # -----------------------------------
    # 5. Save Insights
    # -----------------------------------

    insights_path = os.path.join(output_dir, "eda_insights.txt")

    with open(insights_path, "w") as f:
        for insight in insights:
            f.write(insight + "\n")

    print("\nEDA Completed")
    print(f"Plots saved in: {plot_dir}")
    print(f"Insights saved in: {insights_path}")

    return insights