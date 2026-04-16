from dash import html, dcc, callback, Input, Output, State, ALL
import requests

def render_chatbot():
    return html.Div([
        # Chat Button
        html.Button(
            "AI",
            id="chat-toggle",
            className="chat-toggle-btn",
            n_clicks=0
        ),
        
        # Chat Window
        html.Div([
            html.Div([
                html.H3("TouroAI Assistant", style={"margin": "0", "fontSize": "16px"}),
                html.P("Real-time strategic insights", style={"margin": "0", "fontSize": "10px", "opacity": "0.7"})
            ], className="chat-header"),
            
            html.Div(id="chat-history", className="chat-body"),
            
            html.Div([
                dcc.Input(
                    id="chat-input",
                    placeholder="Ask about your data...",
                    type="text",
                    className="chat-input-field"
                ),
                html.Button("→", id="chat-send", className="chat-send-btn")
            ], className="chat-footer")
        ], id="chat-window", className="chat-window-hidden")
    ], className="chatbot-container")

@callback(
    Output("chat-window", "className"),
    Input("chat-toggle", "n_clicks"),
    State("chat-window", "className"),
    prevent_initial_call=True
)
def toggle_chat(n, current_class):
    if n % 2 == 1:
        return "chat-window"
    return "chat-window-hidden"

@callback(
    Output("chat-history", "children"),
    Input("chat-send", "n_clicks"),
    State("chat-input", "value"),
    State("chat-history", "children"),
    prevent_initial_call=True
)
def handle_chat(n, user_msg, history):
    if not user_msg:
        return history
    
    if history is None:
        history = []
    
    # Add user message
    history.append(html.Div(user_msg, className="msg msg-user"))
    
    # Call AI API
    try:
        import os
        api_url = os.getenv("API_URL", "http://127.0.0.1:8001")
        response = requests.post(f"{api_url}/chat", json={"message": user_msg})
        ai_msg = response.json().get("response", "Error: No response from AI.")
    except Exception as e:
        ai_msg = f"API Error: {str(e)}"
    
    # Add AI message
    history.append(html.Div(ai_msg, className="msg msg-ai"))
    
    return history
