from sqlalchemy import Column, Integer, String, Float, Boolean, Date
from .database import Base

class Booking(Base):
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True, index=True)
    hotel = Column(String)
    is_canceled = Column(Integer)
    lead_time = Column(Integer)
    arrival_date_year = Column(Integer)
    arrival_date_month = Column(String)
    arrival_date_week_number = Column(Integer)
    arrival_date_day_of_month = Column(Integer)
    stays_in_weekend_nights = Column(Integer)
    stays_in_week_nights = Column(Integer)
    adults = Column(Integer)
    children = Column(Float)
    babies = Column(Integer)
    meal = Column(String)
    country = Column(String)
    market_segment = Column(String)
    distribution_channel = Column(String)
    is_repeated_guest = Column(Integer)
    previous_cancellations = Column(Integer)
    previous_bookings_not_canceled = Column(Integer)
    reserved_room_type = Column(String)
    assigned_room_type = Column(String)
    booking_changes = Column(Integer)
    deposit_type = Column(String)
    agent = Column(Float)
    company = Column(Float)
    days_in_waiting_list = Column(Integer)
    customer_type = Column(String)
    adr = Column(Float)
    required_car_parking_spaces = Column(Integer)
    total_of_special_requests = Column(Integer)
    reservation_status = Column(String)
    reservation_status_date = Column(String)

class Review(Base):
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    Review = Column(String)
    Rating = Column(Integer)
    sentiment_score = Column(Float)
    sentiment_label = Column(String)
