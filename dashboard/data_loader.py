import pandas as pd

# Load heavy datasets exactly once in memory to prevent Render OOM
bookings = pd.read_csv("data/cleaned_bookings.csv")
reviews = pd.read_csv("data/reviews_with_sentiment.csv")
