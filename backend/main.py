from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from . import models, database, ai_engine, ml_hub, nlp_pipeline
from .database import engine, get_db
import pandas as pd
import math

# Initialize data
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="TouroPulse Enterprise API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HELPER: Absolute JSON Sanity
def sanitize_json(data):
    if isinstance(data, dict):
        return {k: sanitize_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json(i) for i in data]
    elif isinstance(data, float):
        if not math.isfinite(data):
            return 0.0
    return data

@app.get("/")
def read_root():
    return {"message": "Welcome to TouroPulse Enterprise API", "status": "online"}

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    try:
        total_bookings = db.query(models.Booking).count()
        total_reviews = db.query(models.Review).count()
        
        # Calculate Average ADR safely
        avg_adr_query = db.query(models.Booking).with_entities(models.Booking.adr).all()
        adr_values = [x[0] for x in avg_adr_query if x[0] is not None and math.isfinite(float(x[0]))]
        
        avg_adr_val = sum(adr_values) / len(adr_values) if adr_values else 0.0
            
        return sanitize_json({
            "total_bookings": total_bookings,
            "total_reviews": total_reviews,
            "avg_daily_rate": round(float(avg_adr_val), 2)
        })
    except Exception as e:
        print(f"Stats API Error: {e}")
        return {"total_bookings": 0, "total_reviews": 0, "avg_daily_rate": 0}

@app.get("/predict/price")
def predict_price(
    hotel: str, lead_time: int, month: str, weekend_nights: int, week_nights: int,
    market_segment: str = "Online TA", adults: int = 2, is_repeated_guest: int = 0
):
    """Dual Ensemble ADR Prediction (Report Section 5.1.3)"""
    price = ml_hub.hub.predict_price(
        hotel, lead_time, month, weekend_nights, week_nights,
        market_segment=market_segment, adults=adults,
        is_repeated_guest=is_repeated_guest
    )
    return {"suggested_price": price}

@app.get("/model-metrics")
def get_model_metrics():
    """Return validated ML model performance metrics (Report Section 7.1)"""
    return sanitize_json(ml_hub.hub.get_model_metrics())

@app.get("/nlp/keywords")
def get_nlp_keywords():
    """TF-IDF weighted keyword extraction from NLP pipeline (Report Section 5.2)"""
    try:
        reviews = pd.read_csv("data/reviews_with_sentiment.csv")
        word_pulse_data = nlp_pipeline.nlp_pipeline.generate_word_pulse_data(reviews, top_n=15)
        return sanitize_json(word_pulse_data.to_dict(orient="records"))
    except Exception as e:
        print(f"NLP Keywords Error: {e}")
        return []

@app.get("/forecast")
def get_forecast():
    return sanitize_json(ml_hub.hub.get_forecast())

@app.post("/chat")
def chat_with_ai(query: dict):
    response = ai_engine.ai_engine.chat(query.get("message", ""))
    return {"response": response}

@app.get("/trend")
def get_trend(db: Session = Depends(get_db)):
    try:
        # Monthly aggregation for ADR
        bookings = db.query(models.Booking).with_entities(
            models.Booking.arrival_date_year, 
            models.Booking.arrival_date_month, 
            models.Booking.adr
        ).all()
        
        if not bookings:
            return []
            
        df = pd.DataFrame(bookings, columns=['year', 'month', 'adr'])
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        df['month'] = pd.Categorical(df['month'], categories=months, ordered=True)
        
        # Aggregate
        trend = df.groupby(['year', 'month'], observed=True)['adr'].mean().reset_index()
        
        # Sanitize and return
        return sanitize_json(trend.to_dict(orient="records"))
    except Exception as e:
        print(f"Trend API Error: {e}")
        return []
