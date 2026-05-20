# src/features/engineer.py
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature-engineering')

def create_features(df):
    """Create new features from existing data."""
    logger.info("Creating new features")
    
    # Make a copy to avoid modifying the original dataframe
    df_featured = df.copy()

    #Feature 1: Log transformation on price to be the target variable trained on
    #Create Log Price
    df_featured["log_price"]=np.log1p(df_featured["price"])
    df_featured=df_featured.drop(columns=["price"], errors="ignore")
    logger.info("Created 'log_price' feature and dropped price feature")

    # Feature 2:combine number of reviews and reviews per month into a new feature: total_reviews with lo
    df_featured['total_reviews'] = df_featured['number_of_reviews'].fillna(0)

    #No negative
    df_featured['total_reviews'] = df_featured['number_of_reviews'].clip(lower=0)

    #Convert to integer
    df_featured['total_reviews'] = df_featured['number_of_reviews'].astype(int)

    #Create the activity flash from total reviews
    df_featured['has_reviews'] = (df_featured['total_reviews'] > 0 ).astype(int)

    #Transformed version for modelling
    df_featured['log_total_reviews'] = np.log1p(df_featured['total_reviews'])  # Log-transform to handle skewness
    df_featured=df_featured.drop(columns=["total_reviews"], errors="ignore")
    logger.info("Created 'total_reviews' and 'has reviews' features and dropped 'total_reviews")
    
    # Feature 3: combine latitude and longitude into a new feature: location_score which will be derived by the Haversine formula to calculate distance from neighbourhoods to various points of interest (POI)
    
    # POI Coordinates
    pois = {
        'times_square': (40.7580, -73.9855),
        'wall_street': (40.7061, -74.0091),
        'central_park': (40.7812, -73.9665)
    }

    # Vectorized Haversine for efficiency with large DataFrames
    def haversine_vec(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c

    # Apply to your DataFrame
    for name, coords in pois.items():
        df_featured[f'dist_km_{name}'] = haversine_vec(df_featured['latitude'], df_featured['longitude'], coords[0], coords[1])

    logger.info("Created 'dist_km_times_square, dist_km_wall_street, and dist_km_central_park' features")
    
    # Do NOT one-hot encode categorical variables here; let the preprocessor handle it
    return df_featured

def extract_and_save_neighbourhoods(df: pd.DataFrame) -> list[str]:
    
    logger.info("Extracting and saving neighbourhood metadata")
    neighbourhood_geo = (
        df.groupby(["neighbourhood_group", "neighbourhood"])
          .agg(
              latitude=("latitude", "median"),
              longitude=("longitude", "median")
              )
              .reset_index()
    )
    logger.info("Aggregated neighbourhood geo coordinates")
 
    neighbourhood_geo["neighbourhood_key"] = (
        neighbourhood_geo["neighbourhood_group"] + "_" +
        neighbourhood_geo["neighbourhood"]
    )
    logger.info("Created hierarchical neighbourhood keys")

    geo_lookup = (
        neighbourhood_geo
        .set_index("neighbourhood_key")[["latitude", "longitude"]]
        .to_dict(orient="index")
    )
    logger.info("Built neighbourhood geo lookup dictionary")    

    some_path=Path("configs/neighbourhood_geo.json")
    some_path.parent.mkdir(parents=True, exist_ok=True)

    with some_path.open("w", encoding="utf-8") as f:
        json.dump(geo_lookup, f, indent=2)
        
    logger.info("Persisted neighbourhood geo lookup to configs/neighbourhood_geo.json")
    
    df['neighbourhoods'] = df['neighbourhood_group'] + "_" + df['neighbourhood']
    logger.info("Created 'neighbourhoods' column in dataframe")

    unique_neighbourhoods = sorted(df["neighbourhoods"].dropna().unique().tolist())
    meta_path = Path("configs/neighbourhoods.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(unique_neighbourhoods, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(unique_neighbourhoods)} unique neighbourhoods to {meta_path}")

    return unique_neighbourhoods

def extract_and_save_neighbourhood_stats(df: pd.DataFrame) -> None:
    """
    Compute and save median statistics for numerical features grouped by neighbourhoods and room_type.
    """
    logger.info("Extracting and saving neighbourhood stats")
    
    stats = df.groupby(['neighbourhoods', 'room_type']).agg({
        'minimum_nights': 'median',
        'number_of_reviews': 'median',
        'reviews_per_month': 'median',
        'calculated_host_listings_count': 'median',
        'availability_365': 'median',
    }).reset_index()
    
    logger.info("Aggregated neighbourhood stats")
    
    # Convert to nested dict: {neighbourhoods: {room_type: {feature: value}}}
    stats_dict = {}
    for _, row in stats.iterrows():
        neigh = row['neighbourhoods']
        rt = row['room_type']
        if neigh not in stats_dict:
            stats_dict[neigh] = {}
        stats_dict[neigh][rt] = {
            'minimum_nights': float(row['minimum_nights']),
            'number_of_reviews': float(row['number_of_reviews']),
            'reviews_per_month': float(row['reviews_per_month']) if pd.notnull(row['reviews_per_month']) else 0.0,
            'calculated_host_listings_count': float(row['calculated_host_listings_count']),
            'availability_365': float(row['availability_365']),
        }
    
    stats_path = Path("configs/neighbourhood_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)
    
    logger.info(f"Saved neighbourhood stats to {stats_path}")

def create_preprocessor(df_featured: pd.DataFrame):
    """
    Create a sklearn preprocessing pipeline.
    Assumes feature engineering has already been applied.
    """
    logger.info("Creating preprocessor pipeline")
    df_featured=df_featured.copy()

    # ----------------------------
    # Column groups
    # ----------------------------
    categorical_low_card = [ "room_type"]
    categorical_high_card = ["neighbourhoods"]

    # ----------------------------
    # Transformers
    # ----------------------------

    low_card_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    high_card_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder( handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    # ----------------------------
    # ColumnTransformer
    # ----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("drop_cols", "drop", ["neighbourhood_group", "neighbourhood"]),
            ("low_cat", low_card_transformer, categorical_low_card),
            ("high_cat", high_card_transformer, categorical_high_card),
        ],
        remainder="passthrough"  # Keep numerical features unchanged
    )

    logger.info("Preprocessor pipeline created")

    return preprocessor


def run_feature_engineering(input_file, output_file, preprocessor_file):
    """
    End-to-end feature engineering workflow.

    Steps:
    1. Load cleaned dataset
    2. Create engineered numerical features
    3. Persist neighbourhood metadata
    4. Persist neighbourhood stats for numerical defaults
    5. Encode categorical variables (mapping + one-hot)
    6. Persist feature metadata
    7. Save transformed dataset
    """

    # ------------------------------------------------------------------
    # 1. Load cleaned dataset
    # ------------------------------------------------------------------
    logger.info(f"Loading cleaned data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Feature engineering (numerical features only)
    # ------------------------------------------------------------------
    df_featured = create_features(df)
    print(df_featured.shape)
    logger.info(f"Feature engineering complete. Shape: {df_featured.shape}")

    # ------------------------------------------------------------------
    # 3. Extract & persist neighbourhoods metadata
    # ------------------------------------------------------------------
    extract_and_save_neighbourhoods(df_featured)

    # ------------------------------------------------------------------
    # 4. Extract & persist neighbourhood stats
    # ------------------------------------------------------------------
    extract_and_save_neighbourhood_stats(df_featured)

    # ------------------------------------------------------------------
    # 5. Create & fit preprocessor
    # ------------------------------------------------------------------
    preprocessor = create_preprocessor(df_featured)

    X = df_featured.drop(columns=["log_price"], errors="ignore")
    y = df_featured["log_price"] if "log_price" in df_featured.columns else None

    X_transformed = preprocessor.fit_transform(X)
    logger.info("Preprocessor fitted and data transformed")

    # ------------------------------------------------------------------
    # 6. Save preprocessor (REAL transformer)
    # ------------------------------------------------------------------
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Saved preprocessor to {preprocessor_file}")

    # ------------------------------------------------------------------
    # 7. Persist feature metadata
    # ------------------------------------------------------------------
    feature_metadata = {
        "n_features": X_transformed.shape[1]
   }

    meta_path = Path("configs/feature_metadata.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with meta_path.open("w", encoding="utf-8") as f: 
        json.dump(feature_metadata, f, indent=2)

    # ------------------------------------------------------------------
    # 8. Persist transformed dataset
    # ------------------------------------------------------------------
    df_out = pd.DataFrame(X_transformed)
    if y is not None:
        df_out["log_price"] = y.values

    print(df_out.shape)

    df_out.to_csv(output_file, index=False)
    logger.info(f"Saved transformed dataset to {output_file}")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature engineering for Airbnb data.')
    parser.add_argument('--input', required=True, help='Path to cleaned CSV file')
    parser.add_argument('--output', required=True, help='Path for output CSV file (engineered features)')
    parser.add_argument('--preprocessor', required=False, help='Path to save fitted preprocessor')
    
    args = parser.parse_args()
    
    run_feature_engineering(args.input, args.output, args.preprocessor)

# python src/features/engineer.py   --input data/processed/AB_NYC_2019_quality_assessed.csv   --output data/processed/AB_NYC_2019_feature_engineered.csv   --preprocessor models/trained/preprocessor.pkl