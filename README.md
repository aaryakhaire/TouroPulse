# TouroPulse — Hotel Analytics & AI Consulting Platform

TouroPulse is a cloud-native **Integrated Strategic Decision-Support System (ISDSS)** for the hospitality industry. It unifies booking history, real-time pricing signals, and guest sentiment into a single platform, replacing static reports and siloed data with real-time, data-backed decision-making.

🔗 **Live Demo:** [touro-pulse.vercel.app](https://touro-pulse.vercel.app)

## Features

- **Predictive Revenue Engine** — Random Forest / Gradient Boosting regression ensemble forecasting hotel ADR (Average Daily Rate), achieving **94.2% prediction accuracy** across **117,138 booking records**
- **NLP Sentiment Pipeline** — Custom Latent Sentiment Analysis (LSA) engine mapping guest reviews into Polarity-Intensity Vectors (-1.0 to +1.0); processed 10,000+ reviews, visualized via a "Word Pulse" bubble chart
- **TouroAI** — RAG-based AI consultant powered by Google Gemini-Pro, grounded in live dataset trends for strategic, hallucination-resistant business advice
- **Sub-120ms query latency** on full 117K-record dataset via an async FastAPI backend
- Modular **Decoupled Asynchronous Micro-Architecture (DAMA)** separating backend, dashboard, and AI layers

## Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Dash, Plotly, HTML/CSS (Glassmorphism UI, CSS Grid Bento layout) |
| **Backend** | FastAPI (async/await), Python, dependency-injected DB sessions |
| **ML & AI** | Scikit-learn (Random Forest, Gradient Boosting), custom LSA sentiment pipeline, Google Gemini-Pro (RAG) |
| **Data** | Pandas, NumPy, Plotly Graph Objects — 117,138 booking records |
| **Deployment** | Backend on Render, Frontend on Vercel |

## Project Structure

```
├── backend/       # FastAPI backend, API endpoints, ML/NLP logic
├── dashboard/      # Analytics dashboard (Dash/Plotly)
├── frontend/       # Frontend application
├── data/           # Datasets
├── notebooks/       # Jupyter notebooks — EDA, model training
├── report/          # Project report and documentation
├── app.py
├── requirements.txt
└── render.yaml       # Render deployment config
```

## Getting Started

```bash
git clone https://github.com/aaryakhaire/TouroPulse.git
cd TouroPulse
pip install -r requirements.txt
python app.py
```

> **Note:** If the app needs a `.env` file (e.g. Gemini API key), create one in the root directory before running. See `app.py` for required environment variables.

## Results

| Metric | Value |
|---|---|
| ADR Prediction Accuracy | 94.2% |
| Query Latency (full dataset) | <120ms |
| Booking Records Processed | 117,138 |
| Guest Reviews Analyzed | 10,000+ |
| Sentiment Score Range | -1.0 to +1.0 |

## Future Scope

- LSTM networks for deeper time-series seasonality forecasting
- Kubernetes scaling for multi-resort enterprise deployment
- Companion mobile app (iOS/Android)
- Fine-tuned domain-specific LLM for TouroAI

## Team

- Jatin Rathod — 23101A0050
- Aarya Khaire — 23101A0059
- Ayush Gujar — 23101A0071
