from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_price, batch_predict
from schemas import AirbnbPredictionRequest, PredictionResponse

# Initialize FastAPI app with metadata
app = FastAPI(
    title="New York City Airbnb Prediction API",
    description=(
        "An API for predicting Airbnb Benchmark Market prices in New York City based on various features. "
        "This application is being developed as part of an end-to-end machine learning project, inspired by my Data Science Master's Degree course at University of Miami"
        "Authored by Melvin Maria."
    ),
    version="1.0.0",
    contact={
        "name": "Melvin Maria",
        "email": "melvinjmaria@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # In production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],           # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],           # Allows custom headers like Content-Type
)

@app.get("/health")
async def health_check():
    from inference import model # Assuming 'model' is defined globally in inference.py
    is_ready = model is not None
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
        "version": "1.0.0"
    }

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: AirbnbPredictionRequest):
    return predict_price(request)

# Batch prediction endpoint
@app.post("/batch-predict", response_model=list)
async def batch_predict_endpoint(requests: list[AirbnbPredictionRequest]):
    return batch_predict(requests)

