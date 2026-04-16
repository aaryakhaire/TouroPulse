"""
TouroPulse NLP Sentiment Pipeline
===================================
Implements the organic Latent Sentiment Analysis (LSA) pipeline
as described in Report Section 5.2.

Pipeline Stages:
  1. Text Preprocessing: Tokenization, stopword removal, lemmatization
  2. Sentiment Quantification: Polarity-Intensity Vector (-1.0 to +1.0)
  3. TF-IDF Keyword Extraction: Domain-weighted topic identification
  4. Word Pulse Data Generation: Frequency + sentiment per keyword
"""

import re
import math
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


# ── Hospitality-Domain Sentiment Lexicon ──
# Maps tokens to a Polarity-Intensity Vector value ranging from
# -1.0 (strongly negative) to +1.0 (strongly positive)
SENTIMENT_LEXICON = {
    # Strongly Positive (+0.7 to +1.0)
    "excellent": 1.0, "amazing": 0.95, "outstanding": 0.95, "wonderful": 0.9,
    "fantastic": 0.9, "perfect": 1.0, "luxurious": 0.85, "beautiful": 0.85,
    "spotless": 0.9, "immaculate": 0.9, "exceptional": 0.95, "superb": 0.9,
    "delightful": 0.85, "gorgeous": 0.85, "impeccable": 0.9, "magnificent": 0.9,

    # Positive (+0.3 to +0.7)
    "good": 0.6, "nice": 0.55, "comfortable": 0.65, "clean": 0.7,
    "friendly": 0.7, "helpful": 0.65, "pleasant": 0.6, "quiet": 0.5,
    "spacious": 0.65, "convenient": 0.6, "cozy": 0.6, "great": 0.75,
    "recommended": 0.6, "enjoyable": 0.65, "refreshing": 0.55, "welcoming": 0.7,
    "lovely": 0.7, "attentive": 0.65, "tasty": 0.6, "delicious": 0.75,

    # Mildly Positive (+0.1 to +0.3)
    "okay": 0.2, "adequate": 0.15, "decent": 0.25, "satisfactory": 0.2,
    "average": 0.1, "fine": 0.2, "acceptable": 0.15, "reasonable": 0.2,

    # Mildly Negative (-0.1 to -0.3)
    "small": -0.2, "basic": -0.15, "dated": -0.25, "tired": -0.2,
    "mediocre": -0.25, "overpriced": -0.3, "crowded": -0.25, "bland": -0.2,

    # Negative (-0.3 to -0.7)
    "dirty": -0.7, "noisy": -0.5, "rude": -0.7, "slow": -0.4,
    "broken": -0.6, "uncomfortable": -0.55, "disappointing": -0.6,
    "cold": -0.3, "stained": -0.6, "smelly": -0.65, "unfriendly": -0.6,
    "unhelpful": -0.55, "cramped": -0.5, "poor": -0.5, "bad": -0.6,
    "worst": -0.9, "terrible": -0.85, "horrible": -0.85, "disgusting": -0.9,

    # Strongly Negative (-0.7 to -1.0)
    "awful": -0.85, "filthy": -0.9, "unacceptable": -0.8, "dreadful": -0.85,
    "appalling": -0.9, "nightmare": -0.9, "cockroach": -0.95, "bug": -0.7,
    "mold": -0.8, "dangerous": -0.85, "scam": -0.9, "fraud": -0.9,

    # Hospitality-specific terms
    "breakfast": 0.3, "pool": 0.35, "spa": 0.4, "view": 0.4,
    "parking": -0.1, "wifi": 0.0, "location": 0.3, "staff": 0.3,
    "room": 0.0, "bed": 0.0, "bathroom": 0.0, "restaurant": 0.1,
    "lobby": 0.1, "elevator": 0.0, "housekeeping": 0.2, "concierge": 0.2,
    "check-in": 0.0, "checkout": 0.0, "reception": 0.1, "service": 0.1,
    "food": 0.1, "balcony": 0.3, "shower": 0.0, "towel": 0.0,
    "pillow": 0.0, "minibar": 0.1, "gym": 0.2, "beach": 0.4,
    "price": -0.1, "value": 0.2, "noise": -0.5, "smell": -0.4,
}

# Standard English stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd',
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn',
    'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
    'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
    'hotel', 'stay', 'stayed', 'room', 'night', 'one', 'two', 'got', 'get',
    'would', 'could', 'also', 'like', 'go', 'went', 'back', 'even', 'really',
    'much', 'well', 'said', 'told', 'asked', 'time', 'day', 'days', 'first',
    'n', 'nt', 'did', 'make', 'made',
}

# Simple lemmatization rules (suffix stripping)
LEMMA_RULES = [
    ('ies', 'y'), ('ves', 'f'), ('ing', ''), ('tion', 'te'),
    ('ed', ''), ('ly', ''), ('ers', ''), ('er', ''), ('es', ''),
    ('s', ''),
]


class NLPPipeline:
    """
    Organic NLP Sentiment Analysis Pipeline (Report Section 5.2).
    
    Processes qualitative guest review text into quantifiable sentiment
    metrics surfaced through the Word Pulse visualization module.
    """

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 1),
            min_df=5
        )
        self.keyword_sentiments = {}
        self.keyword_frequencies = {}

    @staticmethod
    def tokenize(text):
        """Stage 1a: Tokenization — split text into individual word tokens."""
        text = str(text).lower()
        tokens = re.findall(r'\b[a-z]{2,}\b', text)
        return tokens

    @staticmethod
    def remove_stopwords(tokens):
        """Stage 1b: Stopword removal — filter common words with no sentiment value."""
        return [t for t in tokens if t not in STOPWORDS]

    @staticmethod
    def lemmatize(token):
        """Stage 1c: Lemmatization — reduce words to base form to normalize vocabulary."""
        if len(token) <= 3:
            return token
        for suffix, replacement in LEMMA_RULES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                base = token[:-len(suffix)] + replacement
                return base if len(base) >= 2 else token
        return token

    @staticmethod
    def compute_polarity(tokens):
        """
        Stage 2: Sentiment Quantification (Report Section 5.2.2).
        
        Each token is matched against the hospitality-domain lexicon.
        Compound review sentiment = weighted mean of individual token polarity scores.
        
        Returns: Polarity-Intensity Vector value from -1.0 to +1.0
        """
        scores = []
        for token in tokens:
            if token in SENTIMENT_LEXICON:
                scores.append(SENTIMENT_LEXICON[token])
        
        if not scores:
            return 0.0
        
        # Weighted mean polarity
        return np.clip(np.mean(scores), -1.0, 1.0)

    def preprocess(self, text):
        """Full preprocessing pipeline: tokenize → stopwords → lemmatize."""
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = [self.lemmatize(t) for t in tokens]
        return tokens

    def process_reviews(self, reviews_df):
        """
        Process a DataFrame of reviews through the full NLP pipeline.
        
        Args:
            reviews_df: DataFrame with 'Review' column
            
        Returns:
            DataFrame with added 'polarity_score' and 'sentiment_label' columns
        """
        print("  NLP Pipeline — Processing Reviews...")
        
        df = reviews_df.copy()
        
        # Preprocess all reviews
        df['tokens'] = df['Review'].apply(self.preprocess)
        
        # Compute polarity for each review
        df['polarity_score'] = df['tokens'].apply(self.compute_polarity)
        
        # Classify: positive (>0.05), negative (<-0.05), neutral
        df['computed_label'] = df['polarity_score'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )
        
        print(f"  → Processed {len(df):,} reviews")
        print(f"  → Sentiment Distribution: {df['computed_label'].value_counts().to_dict()}")
        
        return df

    def extract_tfidf_keywords(self, reviews_df, top_n=20):
        """
        Stage 3: TF-IDF Keyword Extraction (Report Section 5.2.2).
        
        Uses TF-IDF weighting to emphasize domain-relevant terms.
        
        Returns:
            DataFrame with columns: Keyword, Frequency, Avg_Polarity, TF_IDF_Weight
        """
        # Fit TF-IDF on review corpus
        review_texts = reviews_df['Review'].fillna('').astype(str)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(review_texts)
        
        # Get feature names and their average TF-IDF weights
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        avg_weights = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        
        # Build keyword data
        keyword_data = []
        for idx in np.argsort(avg_weights)[::-1]:
            word = feature_names[idx]
            if len(word) < 3:
                continue
            
            # Count frequency across all reviews
            freq = sum(1 for text in review_texts if word in text.lower())
            
            # Get polarity from lexicon
            polarity = SENTIMENT_LEXICON.get(word, 0.0)
            
            # Compute polarity from reviews containing this word
            matching_reviews = reviews_df[review_texts.str.contains(word, case=False, na=False)]
            if 'sentiment_score' in matching_reviews.columns:
                avg_review_polarity = matching_reviews['sentiment_score'].mean()
            else:
                avg_review_polarity = polarity
            
            keyword_data.append({
                'Keyword': word,
                'Frequency': freq,
                'Lexicon_Polarity': polarity,
                'Avg_Review_Polarity': avg_review_polarity,
                'TF_IDF_Weight': avg_weights[idx]
            })
            
            if len(keyword_data) >= top_n:
                break
        
        return pd.DataFrame(keyword_data)

    def generate_word_pulse_data(self, reviews_df, top_n=15):
        """
        Generate data for the Word Pulse bubble chart (Report Section 5.2.3).
        
        Each bubble represents a unique topic keyword:
          - Bubble radius ∝ topic frequency across all reviews  
          - Bubble color maps to Sentiment Gradient (green-to-red diverging scale)
          - Force-directed collision detection positions
        
        Returns:
            DataFrame with: Keyword, Frequency, Sentiment, x, y (positions)
        """
        tfidf_df = self.extract_tfidf_keywords(reviews_df, top_n=top_n)
        
        if tfidf_df.empty:
            return pd.DataFrame()
        
        # Generate force-directed-like positions using golden angle distribution
        n = len(tfidf_df)
        golden_angle = math.pi * (3 - math.sqrt(5))
        
        positions_x = []
        positions_y = []
        for i in range(n):
            r = math.sqrt(i + 1) / math.sqrt(n)  # Radial distance
            theta = i * golden_angle  # Angular position
            positions_x.append(r * math.cos(theta))
            positions_y.append(r * math.sin(theta))
        
        tfidf_df['x'] = positions_x
        tfidf_df['y'] = positions_y
        
        return tfidf_df


# Module-level singleton
nlp_pipeline = NLPPipeline()
