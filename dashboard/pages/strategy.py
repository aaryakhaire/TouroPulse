import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

dash.register_page(__name__, path="/strategy")

# Load data
bookings = pd.read_csv("data/cleaned_bookings.csv")

# 1. LEAD TIME DISTRIBUTION
fig_lead = px.histogram(
    bookings, 
    x="lead_time", 
    nbins=100,
    title="Planner vs. Last-Minute Window (Frequencies)",
    color_discrete_sequence=['#0EA5E9']
)
fig_lead.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Inter")

# 2. CANCELLATION TRENDS (Rolling Avg)
bookings['is_canceled_num'] = bookings['is_canceled'].apply(lambda x: 1 if x == 'yes' or x == 1 or x == True else 0)
# Group by arrival month for trends
cancel_trend = bookings.groupby('arrival_date_month')['is_canceled_num'].mean().reset_index()
# Sort months correctly
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
cancel_trend['arrival_date_month'] = pd.Categorical(cancel_trend['arrival_date_month'], categories=months, ordered=True)
cancel_trend = cancel_trend.sort_values('arrival_date_month')

fig_cancel = px.line(
    cancel_trend, 
    x="arrival_date_month", 
    y="is_canceled_num",
    title="Monthly Cancellation Velocity Radar",
    markers=True,
    color_discrete_sequence=['#F43F5E']
)
fig_cancel.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Inter")

# 3. STAY COMPOSITION (Weekend vs Weekday)
stay_comp = pd.DataFrame({
    'Type': ['Weekend Stays', 'Weekday Stays'],
    'Total': [bookings['stays_in_weekend_nights'].sum(), bookings['stays_in_week_nights'].sum()]
})
fig_stay = px.pie(
    stay_comp, 
    values='Total', 
    names='Type', 
    title="Guest Behavior Structure: Nightly Breakdown",
    hole=0.6,
    color_discrete_sequence=['#0EA5E9', '#F43F5E']
)
fig_stay.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Inter")

# 4. SPECIAL REQUESTS vs ADR
fig_requests = px.scatter(
    bookings.sample(min(2000, len(bookings))),
    x="total_of_special_requests", 
    y="adr",
    color="hotel",
    trendline="ols",
    title="Special Requirements vs. Price Pressure (ADR)",
    opacity=0.3,
    color_discrete_sequence=['#0EA5E9', '#F43F5E']
)
fig_requests.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Inter")

# LAYOUT
layout = html.Div([

    # HERO HEADER
    html.Div([
        html.H1("Strategy Intelligence."),
        html.P("Deep-dive strategy mapping encompassing cancellation velocity, booking windows, and behavioral night-stay analysis for revenue maximization.")
    ], className="dash-hero"),

    # BENTO CONTAINER
    html.Div([
        
        # Row 1: Heavy Charts
        html.Div([
            html.Div([
                dcc.Graph(figure=fig_lead)
            ], className="dash-card", style={"flex": "1"}),
            
            html.Div([
                dcc.Graph(figure=fig_cancel)
            ], className="dash-card", style={"flex": "1.5"}),
        ], style={"display": "flex", "gap": "24px", "marginBottom": "24px"}),

        # Row 2: Behavioral Distribution
        html.Div([
            html.Div([
                dcc.Graph(figure=fig_stay)
            ], className="dash-card", style={"flex": "1"}),
            
            html.Div([
                dcc.Graph(figure=fig_requests)
            ], className="dash-card", style={"flex": "1.8"}),
        ], style={"display": "flex", "gap": "24px"}),

    ], className="bento-container")
])
