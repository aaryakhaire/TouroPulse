import pandas as pd
import sqlite3
import os

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "touropulse.db")
DATA_DIR = os.path.join(BASE_DIR, "data")

def migrate():
    print("Starting Data Migration to Enterprise SQLite (Run from project root)...")
    
    # 1. Connect to DB
    conn = sqlite3.connect(DB_PATH)
    
    # 2. Migrate Bookings
    print("Migrating Bookings...")
    bookings_path = os.path.join(DATA_DIR, "cleaned_bookings.csv")
    if os.path.exists(bookings_path):
        df_bookings = pd.read_csv(bookings_path)
        df_bookings.to_sql("bookings", conn, if_exists="replace", index=True, index_label="id")
        print(f"Migrated {len(df_bookings)} booking records.")
    else:
        print("bookings file not found.")

    # 3. Migrate Reviews
    print("Migrating Reviews...")
    reviews_path = os.path.join(DATA_DIR, "reviews_with_sentiment.csv")
    if os.path.exists(reviews_path):
        df_reviews = pd.read_csv(reviews_path)
        df_reviews.to_sql("reviews", conn, if_exists="replace", index=True, index_label="id")
        print(f"Migrated {len(df_reviews)} review records.")
    else:
        print("reviews file not found.")

    conn.close()
    print("Migration Complete.")

if __name__ == "__main__":
    migrate()
