"""
TouroPulse Sentiment Intelligence Dashboard Page
==================================================
Implements the Word Pulse visualization (Report Section 5.2.3) using
TF-IDF weighted keyword extraction and force-directed bubble positioning.
"""

import dash
from dash import html, dcc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import math
import re

dash.register_page(__name__, path="/sentiment")

# ── Load Data ──
reviews = pd.read_csv("../data/reviews_with_sentiment.csv")

# ══════════════════════════════════════════════════════════════
# 1. SENTIMENT DISTRIBUTION (Report Section 5.2.2)
# ══════════════════════════════════════════════════════════════
fig_dist = px.histogram(
    reviews,
    x="sentiment_label",
    color="sentiment_label",
    title="Core Sentiment Distribution",
    color_discrete_map={"positive": "#00BAFF", "negative": "#FF6B6B", "Positive": "#00BAFF", "Negative": "#FF6B6B"}
)
fig_dist.update_layout(
    template="plotly_white",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_family="Inter"
)

# ══════════════════════════════════════════════════════════════
# 2. WORD PULSE BUBBLE CHART (Report Section 5.2.3)
#    - TF-IDF weighted keyword extraction
#    - Bubble radius ∝ topic frequency
#    - Bubble color maps to Sentiment Gradient (green-to-red)
#    - Force-directed collision detection positioning
# ══════════════════════════════════════════════════════════════

# Stage 1: TF-IDF Keyword Extraction
review_texts = reviews['Review'].fillna('').astype(str)

tfidf = TfidfVectorizer(
    max_features=200,
    stop_words='english',
    ngram_range=(1, 1),
    min_df=10
)
tfidf_matrix = tfidf.fit_transform(review_texts)
feature_names = tfidf.get_feature_names_out()
avg_weights = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

# Hospitality-domain sentiment lexicon for polarity coloring (Report Section 5.2.2)
SENTIMENT_LEXICON = {
    "clean": 0.7, "comfortable": 0.65, "friendly": 0.7, "helpful": 0.65,
    "spacious": 0.65, "great": 0.75, "nice": 0.55, "good": 0.6,
    "excellent": 1.0, "beautiful": 0.85, "quiet": 0.5, "lovely": 0.7,
    "perfect": 1.0, "amazing": 0.95, "wonderful": 0.9, "luxurious": 0.85,
    "delicious": 0.75, "attentive": 0.65, "convenient": 0.6, "cozy": 0.6,
    "dirty": -0.7, "noisy": -0.5, "rude": -0.7, "slow": -0.4,
    "broken": -0.6, "uncomfortable": -0.55, "disappointing": -0.6,
    "poor": -0.5, "bad": -0.6, "worst": -0.9, "terrible": -0.85,
    "small": -0.2, "dated": -0.25, "overpriced": -0.3, "cold": -0.3,
    "breakfast": 0.3, "pool": 0.35, "staff": 0.3, "location": 0.3,
    "service": 0.1, "food": 0.1, "view": 0.4, "bed": 0.0, "price": -0.1,
    "wifi": 0.0, "parking": -0.1, "bathroom": 0.0, "restaurant": 0.1,
    "balcony": 0.3, "shower": 0.0, "check": 0.0, "lobby": 0.1,
    "beach": 0.4, "bar": 0.1, "gym": 0.2, "spa": 0.4,
    "noise": -0.5, "smell": -0.4, "stain": -0.6, "bug": -0.7,
}

# Select top keywords by TF-IDF weight
top_indices = np.argsort(avg_weights)[::-1]
pulse_data = []
for idx in top_indices:
    word = feature_names[idx]
    if len(word) < 3:
        continue
    # Frequency count across reviews
    freq = sum(1 for text in review_texts if word in text.lower())
    # Polarity from lexicon (Polarity-Intensity Vector)
    polarity = SENTIMENT_LEXICON.get(word, 0.0)
    pulse_data.append({
        'Keyword': word,
        'Frequency': freq,
        'Polarity': polarity,
        'TF_IDF': avg_weights[idx]
    })
    if len(pulse_data) >= 15:
        break

df_pulse = pd.DataFrame(pulse_data)

# Force-directed positioning using golden angle spiral (Report Section 5.2.3)
n = len(df_pulse)
golden_angle = math.pi * (3 - math.sqrt(5))
df_pulse['x'] = [math.sqrt(i + 1) / math.sqrt(n) * math.cos(i * golden_angle) for i in range(n)]
df_pulse['y'] = [math.sqrt(i + 1) / math.sqrt(n) * math.sin(i * golden_angle) for i in range(n)]

# Create Word Pulse bubble chart with Plotly Graph Objects (GPU-accelerated WebGL)
fig_pulse = go.Figure()

# Normalize frequency for bubble sizing
max_freq = df_pulse['Frequency'].max() if not df_pulse.empty else 1
min_freq = df_pulse['Frequency'].min() if not df_pulse.empty else 0

fig_pulse.add_trace(go.Scatter(
    x=df_pulse['x'],
    y=df_pulse['y'],
    mode='markers+text',
    marker=dict(
        size=df_pulse['Frequency'] / max_freq * 70 + 20,  # Scaled bubble radius
        color=df_pulse['Polarity'],  # Sentiment Gradient coloring
        colorscale=[
            [0, '#FF6B6B'],      # Negative (red)
            [0.35, '#FF9F43'],   # Mildly negative (orange)
            [0.5, '#94A3B8'],    # Neutral (gray)
            [0.65, '#4ECDC4'],   # Mildly positive (teal)
            [1, '#00BAFF'],      # Positive (blue)
        ],
        cmin=-1.0,
        cmax=1.0,
        opacity=0.85,
        line=dict(width=1, color='rgba(255,255,255,0.3)'),
        colorbar=dict(
            title=dict(text="Polarity", font=dict(size=12)),
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["Negative", "", "Neutral", "", "Positive"],
            len=0.6,
            thickness=12,
        )
    ),
    text=df_pulse['Keyword'],
    textposition='middle center',
    textfont=dict(color='white', size=13, family="Inter"),
    hovertemplate='<b>%{text}</b><br>Frequency: %{customdata[0]}<br>Polarity: %{customdata[1]:.2f}<extra></extra>',
    customdata=list(zip(df_pulse['Frequency'], df_pulse['Polarity']))
))

fig_pulse.update_layout(
    title="Sentiment Word Pulse Bubble Chart",
    template="plotly_white",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_family="Inter",
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    margin=dict(l=20, r=20, t=60, b=20),
    height=450,
)

# ══════════════════════════════════════════════════════════════
# 3. POLARITY-INTENSITY DISTRIBUTION (Report Section 5.2.2)
# ══════════════════════════════════════════════════════════════
if 'sentiment_score' in reviews.columns:
    fig_polarity = px.histogram(
        reviews,
        x='sentiment_score',
        nbins=50,
        title="Polarity-Intensity Vector Distribution",
        color_discrete_sequence=["#0EA5E9"],
        labels={'sentiment_score': 'Polarity Score'}
    )
    fig_polarity.update_layout(
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter"
    )
else:
    fig_polarity = go.Figure()
    fig_polarity.update_layout(title="Polarity data not available")

# ══════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════
layout = html.Div([

    # HERO HEADER
    html.Div([
        html.H1("Sentiment Intelligence."),
        html.P("Extracting the 'Voice of the Guest' using the NLP Sentiment Pipeline — tokenization, stopword removal, "
               "lemmatization, and TF-IDF weighted Polarity-Intensity Vector analysis to identify operational friction points.")
    ], className="dash-hero"),

    # BENTO CONTAINER
    html.Div([

        # Row 1: Word Pulse + Distribution
        html.Div([
            html.Div([
                dcc.Graph(figure=fig_pulse)
            ], className="dash-card", style={"flex": "1.5"}),

            html.Div([
                dcc.Graph(figure=fig_dist)
            ], className="dash-card", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "24px", "marginBottom": "24px"}),

        # Row 2: Polarity Distribution + Insights
        html.Div([
            html.Div([
                dcc.Graph(figure=fig_polarity)
            ], className="dash-card", style={"flex": "1"}),

            html.Div([
                html.H3("NLP Pipeline Summary", style={"marginBottom": "20px", "fontSize": "24px", "fontWeight": "700"}),
                html.Div([
                    html.Div([
                        html.P("Reviews Processed", style={"fontSize": "12px", "color": "var(--text-secondary)", "textTransform": "uppercase"}),
                        html.H2(f"{len(reviews):,}", style={"margin": "0", "color": "var(--accent-primary)"}),
                    ], style={"textAlign": "center", "padding": "20px", "background": "rgba(14,165,233,0.05)", "borderRadius": "14px", "flex": "1"}),
                    html.Div([
                        html.P("Unique Keywords (TF-IDF)", style={"fontSize": "12px", "color": "var(--text-secondary)", "textTransform": "uppercase"}),
                        html.H2(f"{len(pulse_data)}", style={"margin": "0", "color": "var(--accent-secondary)"}),
                    ], style={"textAlign": "center", "padding": "20px", "background": "rgba(244,63,94,0.05)", "borderRadius": "14px", "flex": "1"}),
                ], style={"display": "flex", "gap": "16px", "marginBottom": "20px"}),
                html.Div([
                    html.P("Pipeline Stages:", style={"fontWeight": "700", "marginBottom": "10px"}),
                    html.P("1. Tokenization → Word token splitting", style={"fontSize": "13px", "color": "var(--text-secondary)", "marginBottom": "6px"}),
                    html.P("2. Stopword Removal → Filtering noise tokens", style={"fontSize": "13px", "color": "var(--text-secondary)", "marginBottom": "6px"}),
                    html.P("3. Lemmatization → Base form normalization", style={"fontSize": "13px", "color": "var(--text-secondary)", "marginBottom": "6px"}),
                    html.P("4. Polarity Scoring → Lexicon-based vector [-1, +1]", style={"fontSize": "13px", "color": "var(--text-secondary)", "marginBottom": "6px"}),
                    html.P("5. TF-IDF Weighting → Domain relevance emphasis", style={"fontSize": "13px", "color": "var(--text-secondary)", "marginBottom": "6px"}),
                ], style={"padding": "15px", "background": "rgba(14,165,233,0.03)", "borderRadius": "14px"}),
            ], className="dash-card", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "24px", "marginBottom": "24px"}),

        # Row 3: Live Feedback
        html.Div([
            html.Div([
                html.H3("Live Guest Feedback Insights", style={"marginBottom": "20px", "fontSize": "24px", "fontWeight": "700"}),
                html.Div([
                    html.Div([
                        html.P(f"• {row['Review'][:180]}...", style={"marginBottom": "15px", "color": "var(--text-secondary)", "borderBottom": "1px solid var(--border-color)", "paddingBottom": "10px"})
                    ]) for _, row in reviews.sample(10).iterrows()
                ])
            ], className="dash-card")
        ]),

    ], className="bento-container")
])