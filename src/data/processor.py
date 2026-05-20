# src/data/processor.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data-processor')

def load_data(file_path):
    """Load data from a CSV file."""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the dataset by handling missing values and outliers."""
    logger.info("Cleaning dataset")
    
    # Make a copy to avoid modifying the original dataframe
    df_cleaned = df.copy()
    
    # Handle missing values
    for column in df_cleaned.columns:
        missing_count = df_cleaned[column].isnull().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values in {column}")

    #drop NaN values and unnecessary columns in the dataset
    df_cleaned = df_cleaned.drop(columns=['id','name','host_id', 'host_name', 'last_review'])
    df_cleaned = df_cleaned.dropna()

    logger.info("Dropped unnecessary columns and NaNs")

    logger.info("Checking for duplicate rows...")

    duplicates = df_cleaned.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Found {duplicates} duplicate rows. Dropping them...")
        df_cleaned = df_cleaned.drop_duplicates()
    else:
        logger.info(" No duplicate rows found.")

    logger.info(f"Dataset shape before Outlier Removal: {df_cleaned.shape}")
    
    df_price_qc = df_cleaned.copy()

    segment_cols = ["neighbourhood_group", "room_type"]

    # Segment-level thresholds
    df_price_qc["segment_p99"] = df_price_qc.groupby(segment_cols)["price"].transform(
        lambda x: x.quantile(0.99)
    )

    df_price_qc["segment_p95"] = df_price_qc.groupby(segment_cols)["price"].transform(
        lambda x: x.quantile(0.95)
    )

    df_price_qc["segment_p935"] = df_price_qc.groupby(segment_cols)["price"].transform(
        lambda x: x.quantile(0.935)
    )

    df_price_qc["segment_p905"] = df_price_qc.groupby(segment_cols)["price"].transform(
        lambda x: x.quantile(0.905)
    )

    # Global extreme threshold
    global_p90 = df_price_qc["price"].quantile(0.90)

    df_price_qc["price_quality_flag"] = "normal"

    # Suspicious entires rooms
    entire_home_error = (
        (df_price_qc["room_type"] == "Entire home/apt") &
        (df_price_qc["price"] > df_price_qc["segment_p905"])
    )

    # Suspicious shared rooms
    shared_room_error = (
        (df_price_qc["room_type"] == "Shared room") &
        (df_price_qc["price"] > df_price_qc["segment_p905"])
    )

    # Suspicious private rooms
    private_room_error = (
        (df_price_qc["room_type"] == "Private room") &
        (
            (df_price_qc["price"] > df_price_qc["segment_p905"]) |
            (df_price_qc["price"] > 1000)
        )
    )

    # Globally extreme prices
    global_extreme = df_price_qc["price"] > global_p90

    df_price_qc.loc[entire_home_error, "price_quality_flag"] = "suspicious_entire_home_price"
    df_price_qc.loc[shared_room_error, "price_quality_flag"] = "suspicious_shared_room_price"
    df_price_qc.loc[private_room_error, "price_quality_flag"] = "suspicious_private_room_price"
    df_price_qc.loc[global_extreme, "price_quality_flag"] = "globally_extreme_price"

    # Suspicious brox neighbourhood group
    brox_error = (
        (df_price_qc["neighbourhood_group"] == "Bronx") &
        (df_price_qc["price"] > df_price_qc["segment_p905"])
    )

    # Suspicious brox neighbourhood group
    Brooklyn_error = (
        (df_price_qc["neighbourhood_group"] == "Brookyn") &
        (df_price_qc["price"] > df_price_qc["segment_p95"])
    )

    # Suspicious brox neighbourhood group
    Manhattan_error = (
        (df_price_qc["neighbourhood_group"] == "Manhattan") &
        (df_price_qc["price"] > df_price_qc["segment_p95"])
    )

    # Suspicious brox neighbourhood group
    Queens_error = (
        (df_price_qc["neighbourhood_group"] == "Queens") &
        (df_price_qc["price"] > df_price_qc["segment_p95"])
    )

    # Suspicious brox neighbourhood group
    State_Island_error = (
        (df_price_qc["neighbourhood_group"] == "State Island") &
        (df_price_qc["price"] > df_price_qc["segment_p905"])
    )

    df_price_qc.loc[entire_home_error, "price_quality_flag"] = "suspicious_entire_home_price"
    df_price_qc.loc[shared_room_error, "price_quality_flag"] = "suspicious_shared_room_price"
    df_price_qc.loc[private_room_error, "price_quality_flag"] = "suspicious_private_room_price"
    df_price_qc.loc[global_extreme, "price_quality_flag"] = "globally_extreme_price"
    df_price_qc.loc[brox_error, "price_quality_flag"] = "suspicious_bronx_price"
    df_price_qc.loc[Brooklyn_error, "price_quality_flag"] = "suspicious_brooklyin_price"
    df_price_qc.loc[Manhattan_error, "price_quality_flag"] = "suspicious_manhattan_price"
    df_price_qc.loc[Queens_error, "price_quality_flag"] = "suspicious_queens_price"
    df_price_qc.loc[State_Island_error, "price_quality_flag"] = "suspicious_state_islands_price"
    df_price_qc.loc[global_extreme, "price_quality_flag"] = "globally_extreme_price"

    df_price_qc["price_quality_flag"].value_counts()
    
    # Flags to remove
    remove_flags = [
        "suspicious_private_room_price",
        "globally_extreme_price",
        "suspicious_bronx_price",
        "suspicious_brooklyn_price",
        "suspicious_manhattan_price",
        "suspicious_queens_price",
        "suspicious_state_island_price",
        "suspicious_entire_home_price",
    ]

    # Create filtered dataframe
    df_price_filtered = df_price_qc[
        ~df_price_qc["price_quality_flag"].isin(remove_flags)
    ].copy()

    # Check removal
    print("Original rows:", len(df_price_qc))
    print("Filtered rows:", len(df_price_filtered))
    print("Rows removed:", len(df_price_qc) - len(df_price_filtered))
    print("Percentage removed:", ((len(df_price_qc) - len(df_price_filtered)) / len(df_price_qc)) * 100)

    # Check remaining flags
    df_price_filtered["price_quality_flag"].value_counts()

    df_cleaned=df_price_filtered

    logger.info(f"Removed outliers. New dataset shape: {df_cleaned.shape}")
    
    df_cleaned=df_cleaned.drop(columns=["price_quality_flag", "segment_p99", "segment_p995","price_modified_z", "segment_p95", "segment_p935","segment_p905"], errors="ignore")
    
    logger.info(f"Quality Assessed dataset shape: {df_cleaned.shape}")
    
    return df_cleaned

def process_data(input_file, output_file):
    """Full data processing pipeline."""
    # Create output directory if it doesn't exist
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(input_file)
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Save processed data
    df_cleaned.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")
    
    return df_cleaned

if __name__ == "__main__":
    # Example usage
    process_data(
        input_file="data/raw/AB_NYC_2019.csv", 
        output_file="data/processed/AB_NYC_2019_quality_assessed.csv"
    )

# python src/data/processor.py   --input data/raw/AB_NYC_2019.csv   --output data/processed/AB_NYC_2019_quality_assessed.csv