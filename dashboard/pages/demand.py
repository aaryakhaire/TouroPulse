import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path="/demand")

# Load data
bookings = pd.read_csv("../data/cleaned_bookings.csv")

# Sort months correctly
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
bookings['arrival_date_month'] = pd.Categorical(bookings['arrival_date_month'], categories=months, ordered=True)

# 1. MONTHLY BOOKING VOLUME
fig_vol = px.histogram(
    bookings,
    x="arrival_date_month",
    title="Seasonal Demand Pulse (Total Bookings)",
    color_discrete_sequence=['#0EA5E9']
)
fig_vol.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Inter", margin=dict(l=0, r=0, t=50, b=0))

# 3. GLOBAL DEMAND HEATMAP
country_counts = bookings['country'].value_counts().reset_index()
country_counts.columns = ['country', 'bookings']

fig_map = px.choropleth(
    country_counts,
    locations="country",
    color="bookings",
    hover_name="country",
    title="Global Booking Origins (Demand Density)",
    color_continuous_scale=['#0EA5E9', '#F43F5E'],
    projection="natural earth"
)
fig_map.update_layout(
    template="plotly_white", 
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)', 
    font_family="Inter",
    margin=dict(l=0, r=0, t=50, b=0),
    coloraxis_showscale=False
)
fig_map.update_geos(
    showocean=True, oceancolor="rgba(14, 165, 233, 0.05)",
    showcountries=True, countrycolor="rgba(255,255,255,0.1)",
    showframe=False
)

# 2. LEAD TIME VS MONTH (Heatmap/Agg)
lead_month = bookings.groupby(['arrival_date_month', 'hotel'])['lead_time'].mean().reset_index()
fig_lead_month = px.bar(
    lead_month,
    x='arrival_date_month',
    y='lead_time',
    color='hotel',
    barmode='group',
    title="Global Planning Windows (Lead Time by Month)",
    color_discrete_sequence=['#0EA5E9', '#F43F5E']
)
fig_lead_month.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Inter")

# LAYOUT
layout = html.Div([

    # HERO HEADER
    html.Div([
        html.H1("Booking Analytics."),
        html.P("Predictive demand modeling tracking global tourism volumes, seasonal planning windows, and market fluctuations.")
    ], className="dash-hero"),

    # BENTO CONTAINER
    html.Div([
        
        # Row 1: The Global Map (Hero Feature)
        html.Div([
            html.Div([
                dcc.Graph(figure=fig_map)
            ], className="dash-card", style={"flex": "1"}),
        ], style={"marginBottom": "24px"}),

        # Row 2: Secondary Charts
        html.Div([
            html.Div([
                dcc.Graph(figure=fig_vol)
            ], className="dash-card", style={"flex": "1"}),
            
            html.Div([
                dcc.Graph(figure=fig_lead_month)
            ], className="dash-card", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "24px"}),

    ], className="bento-container")
])