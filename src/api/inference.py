# src/api/inference.py
import joblib
import pandas as pd
import json
import numpy as np
from datetime import datetime
from schemas import AirbnbPredictionRequest, PredictionResponse

# Load model and preprocessor
MODEL_PATH = "models/trained/NYC_Airbnb_log_price_model.pkl"
PREPROCESSOR_PATH = "models/trained/preprocessor.pkl"
GEO_PATH = "configs/neighbourhood_geo.json"
STATS_PATH = "configs/neighbourhood_stats.json"

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with open(GEO_PATH, "r") as f:
        geo_data = json.load(f)
    with open(STATS_PATH, "r") as f:
        stats_data = json.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model, preprocessor, or configs: {str(e)}")

def haversine(lat1, lon1, lat2, lon2):
    """Vectorized Haversine formula for distance calculation."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # Earth radius in km

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features: total_reviews and has_reviews and POI distances."""
    df = df.copy()

    # total_reviews
    # Feature 1:combine number of reviews and reviews per month into a new feature: total_reviews with lo
    df['total_reviews'] = df['number_of_reviews'].fillna(0)

    #No negative
    df['total_reviews'] = df['number_of_reviews'].clip(lower=0)

    #Convert to integer
    df['total_reviews'] = df['number_of_reviews'].astype(int)

    #Create the activity flash from total reviews
    df['has_reviews'] = (df['total_reviews'] > 0 ).astype(int)

    #Transformed version for modelling
    df['log_total_reviews'] = np.log1p(df['total_reviews'])  # Log-transform to handle skewness
    df=df.drop(columns=["total_reviews"], errors="ignore")
    
    # POI coordinates
    pois = {
        'times_square': (40.7580, -73.9855),
        'wall_street': (40.7061, -74.0091),
        'central_park': (40.7812, -73.9665)
    }
    
    # Distances
    for name, coords in pois.items():
        df[f'dist_km_{name}'] = haversine(df['latitude'], df['longitude'], coords[0], coords[1])
    
    return df

def predict_price(request: AirbnbPredictionRequest) -> PredictionResponse:
    """
    Predict Airbnb price based on input features.
    """
    # Compute hierarchical keys
    neighbourhoods = f"{request.neighbourhood_group}_{request.neighbourhood}"
    
    # Validate and fetch geo data
    if neighbourhoods not in geo_data:
        raise ValueError(f"Unknown neighbourhood combination: {neighbourhoods}")
    
    lat = geo_data[neighbourhoods]["latitude"]
    long = geo_data[neighbourhoods]["longitude"]
    
    # Validate and fetch stats
    if neighbourhoods not in stats_data or request.room_type not in stats_data[neighbourhoods]:
        raise ValueError(f"No stats available for {neighbourhoods} and {request.room_type}")
    
    room_stats = stats_data[neighbourhoods][request.room_type]
    
    # Prepare input data with median values for numerical features
    data = {
        "neighbourhood_group": request.neighbourhood_group,
        "neighbourhood": request.neighbourhood,
        "room_type": request.room_type,
        "latitude": lat,
        "longitude": long,
        "minimum_nights": room_stats["minimum_nights"],
        "number_of_reviews": room_stats["number_of_reviews"],
        "reviews_per_month": room_stats["reviews_per_month"],
        "calculated_host_listings_count": room_stats["calculated_host_listings_count"],
        "availability_365": room_stats["availability_365"],
    }
    
    input_data = pd.DataFrame([data])
    
    # Add engineered features (total_reviews, distances)
    input_data = add_engineered_features(input_data)
    
    # Add neighbourhoods column
    input_data["neighbourhoods"] = neighbourhoods
    
    # Preprocess input data
    processed_features = preprocessor.transform(input_data)

    print("Raw inference columns:", list(input_data.columns))
    print("Processed feature shape:", processed_features.shape)

    try:
        print("Model expected features:", len(model.feature_names_))
        print("Last model features:", model.feature_names_[-10:])
    except Exception as e:
        print("Could not inspect model feature names:", e)

    try:
        preprocessor_features = preprocessor.get_feature_names_out()
        print("Preprocessor output features:", len(preprocessor_features))
        print("Last preprocessor features:", preprocessor_features[-10:])
    except Exception as e:
        print("Could not inspect preprocessor features:", e)

    # Make prediction in log-price scale
    predicted_log_price = model.predict(processed_features)[0]

    # Convert log-price prediction back to original price scale
    predicted_price = np.expm1(predicted_log_price)

    # Convert numpy type to Python float and round to 2 decimal places
    predicted_price = round(float(predicted_price), 2)

    # Confidence interval (10% range)
    confidence_interval = [predicted_price * 0.9, predicted_price * 1.1]

    # Convert confidence interval values to Python float and round to 2 decimal places
    confidence_interval = [round(float(value), 2) for value in confidence_interval]

    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_interval=confidence_interval,
        features_importance={},
        prediction_time=datetime.now().isoformat()
    )

def batch_predict(requests: list[AirbnbPredictionRequest]) -> list[float]:
    """
    Perform batch predictions.
    """
    input_datas = []
    for request in requests:
        neighbourhoods = f"{request.neighbourhood_group}_{request.neighbourhood}"
        
        if neighbourhoods not in geo_data:
            raise ValueError(f"Unknown neighbourhood combination: {neighbourhoods}")
        
        lat = geo_data[neighbourhoods]["latitude"]
        long = geo_data[neighbourhoods]["longitude"]
        
        if neighbourhoods not in stats_data or request.room_type not in stats_data[neighbourhoods]:
            raise ValueError(f"No stats available for {neighbourhoods} and {request.room_type}")
        
        room_stats = stats_data[neighbourhoods][request.room_type]
        
        data = {
            "neighbourhood_group": request.neighbourhood_group,
            "neighbourhood": request.neighbourhood,
            "room_type": request.room_type,
            "latitude": lat,
            "longitude": long,
            "minimum_nights": room_stats["minimum_nights"],
            "number_of_reviews": room_stats["number_of_reviews"],
            "reviews_per_month": room_stats["reviews_per_month"],
            "calculated_host_listings_count": room_stats["calculated_host_listings_count"],
            "availability_365": room_stats["availability_365"],
        }
        input_datas.append(data)
    
    input_data = pd.DataFrame(input_datas)
    
    # Add engineered features
    input_data = add_engineered_features(input_data)
    
    # Add neighbourhoods column
    input_data["neighbourhoods"] = input_data["neighbourhood_group"] + "_" + input_data["neighbourhood"]
    
    # Preprocess input data
    processed_features = preprocessor.transform(input_data)

    # Make predictions in log-price scale
    predicted_log_prices = model.predict(processed_features)

    # Convert log-price predictions back to original price scale
    predicted_prices = np.expm1(predicted_log_prices)

    return predicted_prices.tolist()