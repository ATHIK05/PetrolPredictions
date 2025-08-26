from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for frontend (adjust origins in production)
origins = [
    "http://localhost:5173",  # Your Vite frontend development server
    "http://127.0.0.1:5173",
    # Add your deployed frontend URL here when available
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
try:
    model = joblib.load('petrol_demand_model.pkl')
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")

# ------------------ OSM HELPERS (Copied from previous app.py) ------------------
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def query_overpass(lat: float, lon: float, radius: int = 1000, tags: dict | None = None):
    if tags is None:
        tags = {}
    tag_query = "".join(f'["{k}"="{v}"]' for k, v in tags.items())
    query = f"""
    [out:json][timeout:25];
    (
      node{tag_query}(around:{radius},{lat},{lon});
      way{tag_query}(around:{radius},{lat},{lon});
      relation{tag_query}(around:{radius},{lat},{lon});
    );
    out center;
    """
    try:
        res = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
        res.raise_for_status()
        return res.json().get("elements", [])
    except Exception:
        return []


def get_nearest_petrol_station(lat: float, lon: float):
    elements = query_overpass(lat, lon, radius=5000, tags={"amenity": "fuel"})
    min_dist = None
    for e in elements:
        elat = e.get("lat") or (e.get("center") or {}).get("lat")
        elon = e.get("lon") or (e.get("center") or {}).get("lon")
        if elat is None or elon is None:
            continue
        try:
            dlat = np.radians(float(elat) - lat)
            dlon = np.radians(float(elon) - lon)
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(np.radians(lat))
                * np.cos(np.radians(float(elat)))
                * np.sin(dlon / 2) ** 2
            )
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            d_km = 6371.0 * c
            if min_dist is None or d_km < min_dist:
                min_dist = d_km
        except Exception:
            continue
    return (round(float(min_dist), 2) if min_dist is not None else None, len(elements))


def get_nearby_pois(lat: float, lon: float):
    schools = query_overpass(lat, lon, tags={"amenity": "school"})
    offices = query_overpass(lat, lon, tags={"office": "company"})
    factories = query_overpass(lat, lon, tags={"building": "industrial"})
    return len(schools), len(offices), len(factories)


def get_macro_features(lat: float, lon: float):
    return {
        "population_density": 9500,
        "vehicle_ownership_per_1000": 350,
        "avg_income": 25000,
    }

# ------------------ FEATURE ENGINEERING & PREDICTION (Copied from previous app.py) ------------------

def build_and_engineer_features(raw_values: dict, lat: float, lon: float):
    currentSpeed = float(raw_values.get('currentSpeed', 35.0) or 35.0)
    freeFlowSpeed = float(raw_values.get('freeFlowSpeed', 55.0) or 55.0)
    confidence = float(raw_values.get('confidence', 0.9) or 0.9)
    congestionIndex = float(raw_values.get('congestionIndex', 20.0) or 20.0)
    nearest_petrol_distance_km = float(raw_values.get('nearest_petrol_distance_km', 1.0) or 1.0)
    petrol_stations_within_1km = int(raw_values.get('petrol_stations_within_1km', 2) or 2)
    nearby_schools = int(raw_values.get('nearby_schools', 5) or 5)
    nearby_offices = int(raw_values.get('nearby_offices', 10) or 10)
    nearby_factories = int(raw_values.get('nearby_factories', 2) or 2)
    population_density = float(raw_values.get('population_density', 9500) or 9500)
    vehicle_ownership_per_1000 = float(raw_values.get('vehicle_ownership_per_1000', 350) or 350)
    avg_income = float(raw_values.get('avg_income', 25000) or 25000)

    now = datetime.now()
    hour = float(now.hour)
    day_of_week = float(now.weekday())
    day_of_year = float(now.timetuple().tm_yday)
    speed_congestion_interaction = currentSpeed * congestionIndex
    inverse_nearest_petrol_distance = 1 / (nearest_petrol_distance_km + 0.01)
    total_nearby_pois = nearby_schools + nearby_offices + nearby_factories

    features = [
        lat,
        lon,
        currentSpeed,
        freeFlowSpeed,
        confidence,
        congestionIndex,
        nearest_petrol_distance_km,
        petrol_stations_within_1km,
        nearby_schools,
        nearby_offices,
        nearby_factories,
        population_density,
        vehicle_ownership_per_1000,
        avg_income,
        hour,
        day_of_week,
        day_of_year,
        speed_congestion_interaction,
        inverse_nearest_petrol_distance,
        total_nearby_pois,
    ]
    return np.array(features, dtype=float).reshape(1, -1)


def predict_petrol_demand(feature_array: np.ndarray):
    try:
        prediction = model.predict(feature_array)
        return float(prediction[0])
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")


def calculate_suitability(predicted_demand, raw_values: dict):
    nearest_d = raw_values.get('nearest_petrol_distance_km', 0.0)
    inverse_nearest_petrol_distance = 1.0 / (float(nearest_d) + 0.01)
    total_nearby_pois = (
        raw_values.get('nearby_schools', 0) +
        raw_values.get('nearby_offices', 0) +
        raw_values.get('nearby_factories', 0)
    )

    temp_values = {
        'predicted_petrol_demand_L': predicted_demand,
        'inverse_nearest_petrol_distance': inverse_nearest_petrol_distance,
        'total_nearby_pois': total_nearby_pois,
        'population_density': raw_values.get('population_density', 0),
        'avg_income': raw_values.get('avg_income', 0),
    }

    weights = {
        'predicted_petrol_demand_L': 0.4,
        'inverse_nearest_petrol_distance': 0.3,
        'total_nearby_pois': 0.1,
        'population_density': 0.1,
        'avg_income': 0.1,
    }

    suitability_score = sum(temp_values.get(feature, 0) * weights[feature] for feature in weights)

    highly_suitable_threshold = 5000.0
    moderately_suitable_threshold = 4000.0

    if suitability_score >= highly_suitable_threshold:
        suitability_classification = "Highly Suitable"
    elif suitability_score >= moderately_suitable_threshold:
        suitability_classification = "Moderately Suitable"
    else:
        suitability_classification = "Less Suitable"

    return suitability_score, suitability_classification

# ------------------ API ENDPOINT ------------------

class PredictRequest(BaseModel):
    city: str
    area_name: str

class PredictResponse(BaseModel):
    predicted_demand_L: float
    suitability_score: float
    suitability_classification: str
    explanation: str

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    city = request.city
    area_name = request.area_name

    if not city or not area_name:
        raise HTTPException(status_code=400, detail="Please provide both city and area name.")

    try:
        geolocator = Nominatim(user_agent="petrol_demand_fastapi_app")
        location = geolocator.geocode(f"{area_name}, {city}", timeout=10)
    except GeocoderServiceError as e:
        raise HTTPException(status_code=500, detail=f"Geocoding error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geocoding failed: {e}")

    if not location:
        raise HTTPException(status_code=404, detail="Could not locate the given area. Try a more specific name.")

    lat, lon = float(location.latitude), float(location.longitude)

    nearest_km, stations_within = get_nearest_petrol_station(lat, lon)
    schools, offices, factories = get_nearby_pois(lat, lon)
    macro = get_macro_features(lat, lon)

    current_speed = 35.0 # Placeholder
    free_flow_speed = 55.0 # Placeholder
    confidence = 0.9 # Placeholder
    congestion_index = 20.0 # Placeholder

    raw_values = {
        'currentSpeed': current_speed,
        'freeFlowSpeed': free_flow_speed,
        'confidence': confidence,
        'congestionIndex': congestion_index,
        'nearest_petrol_distance_km': nearest_km or 1.0,
        'petrol_stations_within_1km': stations_within or 0,
        'nearby_schools': schools,
        'nearby_offices': offices,
        'nearby_factories': factories,
        'population_density': macro['population_density'],
        'vehicle_ownership_per_1000': macro['vehicle_ownership_per_1000'],
        'avg_income': macro['avg_income'],
    }

    try:
        feature_array = build_and_engineer_features(raw_values, lat, lon)
        predicted_demand = predict_petrol_demand(feature_array)
        suitability_score, suitability_classification = calculate_suitability(predicted_demand, raw_values)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    explanation = (
        f"### Location Details:\n"
        f"- **City**: {city}\n"
        f"- **Area**: {area_name}\n"
        f"- **Coordinates**: {lat:.5f}, {lon:.5f}\n\n"
        f"### Key Factors Influencing Demand & Suitability:\n"
        f"- **Distance to Nearest Petrol Station**: {raw_values['nearest_petrol_distance_km']:.2f} km\n"
        f"- **Petrol Stations within 1km**: {raw_values['petrol_stations_within_1km']}\n"
        f"- **Nearby Infrastructure**: {raw_values['nearby_schools']} schools, {raw_values['nearby_offices']} offices, {raw_values['nearby_factories']} factories\n"
        f"- **Population Density**: {raw_values['population_density']:.0f} per sq. km\n"
        f"- **Vehicle Ownership**: {raw_values['vehicle_ownership_per_1000']:.0f} vehicles per 1000 people\n"
        f"- **Average Income**: INR {raw_values['avg_income']:.0f} per month\n\n"
        f"**Reasoning**: Locations with higher population density, average income, and a good balance of nearby points of interest tend to have higher petrol demand. Proximity to existing petrol stations is also a factor, with a greater inverse distance often implying more potential for a new bunk.\""
    )

    return PredictResponse(
        predicted_demand_L=predicted_demand,
        suitability_score=suitability_score,
        suitability_classification=suitability_classification,
        explanation=explanation,
    )
