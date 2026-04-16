import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path="/pricing")

# Load data
bookings = pd.read_csv("../data/cleaned_bookings.csv")

# 1. ADR DISTRIBUTION BOX PLOT (Heavy Analysis)
fig_box = px.box(
    bookings, 
    x="hotel", 
    y="adr", 
    points="all", 
    title="Price Dispersion & Yield Opportunity",
    color="hotel",
    color_discrete_sequence=['#0EA5E9', '#F43F5E']
)
fig_box.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Inter")

# 2. ADR BY ROOM TYPE
fig_room = px.bar(
    bookings.groupby('reserved_room_type')['adr'].mean().reset_index(),
    x='reserved_room_type',
    y='adr',
    title="Yield by Inventory Class (Reserved Room Type)",
    color='adr',
    color_continuous_scale=['#0EA5E9', '#F43F5E']
)
fig_room.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Inter")

# LAYOUT
layout = html.Div([

    # HERO HEADER
    html.Div([
        html.H1("Pricing Intelligence."),
        html.P("Dynamic pricing optimization tracking ADR corridors, yield dispersion, and inventory class performance across global markets.")
    ], className="dash-hero"),

    # BENTO CONTAINER
    html.Div([
        
        # Row 1: Heavy Charts
        html.Div([
            html.Div([
                dcc.Graph(figure=fig_box)
            ], className="dash-card", style={"flex": "1.5"}),
            
            html.Div([
                dcc.Graph(figure=fig_room)
            ], className="dash-card", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "24px", "marginBottom": "24px"}),

    ], className="bento-container")
])