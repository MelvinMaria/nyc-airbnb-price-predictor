# src/api/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
import json
from pathlib import Path

# Load valid locations from the JSON file (defensive)
VALID_LOCATIONS = []
meta_path = Path("configs/neighbourhoods.json")
if meta_path.exists():
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            VALID_LOCATIONS = json.load(f)
    except Exception:
        VALID_LOCATIONS = []

ALLOWED_ROOM_TYPES = {"Entire home/apt", "Private room", "Shared room"}
ALLOWED_NEIGHBOURHOOD_GROUPS = {"Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"}

class AirbnbPredictionRequest(BaseModel):
    neighbourhood_group: str = Field(..., description="Neighbourhood group name")
    neighbourhood: str = Field(..., description="Neighbourhood name")
    room_type: str = Field(..., description="Type of room offered")

    @field_validator("neighbourhood", mode="after")
    def validate_hierarchical_neighbourhood(cls, v, info):
        group = info.data.get("neighbourhood_group")
        if not group:
            return v

        hierarchical = f"{group}_{v}"

        if VALID_LOCATIONS and hierarchical not in VALID_LOCATIONS:
            raise ValueError(
                f"Unknown neighbourhood combination: {hierarchical}. "
                "Model may not generalize well."
            )

        return v

    @field_validator("room_type")
    def room_type_must_be_allowed(cls, v):
        if v not in ALLOWED_ROOM_TYPES:
            raise ValueError(f"room_type must be one of {sorted(ALLOWED_ROOM_TYPES)}")
        return v

    @field_validator("neighbourhood_group")
    def neighbourhood_group_must_be_allowed(cls, v):
        if v not in ALLOWED_NEIGHBOURHOOD_GROUPS:
            raise ValueError(f"neighbourhood_group must be one of {sorted(ALLOWED_NEIGHBOURHOOD_GROUPS)}")
        return v

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: List[float]
    features_importance: Dict[str, float]
    prediction_time: str