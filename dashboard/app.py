import dash
from dash import html, dcc

from components.chatbot import render_chatbot

app = dash.Dash(
    __name__, 
    use_pages=True,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=Playfair+Display:wght@700;800&display=swap"
    ]
)

app.layout = html.Div([

    # HORIZON OVERDRIVE: ATMOSPHERIC BACKDROP
    html.Div([
        html.Div(className="blob"),
        html.Div(className="blob", style={"animationDelay": "-5s", "background": "var(--accent-primary)"}),
        html.Div(className="blob", style={"animationDelay": "-10s", "background": "var(--accent-secondary)"}),
    ], className="aurora-container"),

    # SIDEBAR
    html.Div([
        html.H2("TouroPulse"),
        
        dcc.Link("Overview", href="/", className="nav-link"),
        dcc.Link("Prediction Hub", href="/prediction", className="nav-link"),
        dcc.Link("Strategy Intelligence", href="/strategy", className="nav-link"),
        dcc.Link("Sentiment Intelligence", href="/sentiment", className="nav-link"),
        dcc.Link("Demand Analytics", href="/demand", className="nav-link"),
        dcc.Link("Pricing Intelligence", href="/pricing", className="nav-link"),

    ], className="sidebar"),

    # CONTENT
    html.Div([
        dash.page_container
    ], className="content"),

    # AI Chatbot
    render_chatbot()

])

if __name__ == "__main__":
    app.run(debug=True)