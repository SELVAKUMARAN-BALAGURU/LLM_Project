import sys
import json

from tools.profiler import load_dataset, generate_profile
from tools.transformer import apply_cleaning_plan
from agents.planner_agent import generate_cleaning_plan


def run_pipeline(file_path):

    print("\nLoading dataset...")
    df = load_dataset(file_path)

    print("Generating dataset profile...")
    profile = generate_profile(df)

    print("\nDATASET HEALTH SCORE:", profile["dataset_health_score"])

    print("\nPROFILE SUMMARY:")
    print(json.dumps(profile, indent=2))

    print("\nGenerating AI cleaning plan...")
    plan = generate_cleaning_plan(profile)

    print("\nAI CLEANING PLAN:")
    print(json.dumps(plan, indent=2))

    print("\nApplying cleaning transformations...")
    cleaned_df, summary = apply_cleaning_plan(df, plan)

    print("\nTRANSFORMATION SUMMARY:")
    print(summary)

    cleaned_df.to_csv("cleaned_output.csv", index=False)

    print("\nCleaned dataset saved as cleaned_output.csv")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python -m app.main <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    run_pipeline(dataset_path)