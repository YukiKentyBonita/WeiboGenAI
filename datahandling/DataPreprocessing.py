import os
import re
import boto3
import pandas as pd
import datahandling.DataHandling as dh
from pathlib import Path


# ---------- Initialize AWS Translate client ----------

translate = boto3.client("translate", region_name="us-east-1")


# ---------- Translation helper ----------

def content_translation(text: str):
    """
    Clean a Weibo post's content and translate Chinese -> English.
    Returns None if text is empty or cannot be translated.
    """
    if pd.isnull(text):
        return None

    # Ensure it's a string
    text = str(text)

    # Remove hashtag blocks like #xxx# (Weibo style)
    text_clean = re.sub(r'#.*?#', '', text).strip()
    if text_clean == "":
        return None

    try:
        response = translate.translate_text(
            Text=text_clean,
            SourceLanguageCode="zh",
            TargetLanguageCode="en"
        )
        return response["TranslatedText"]
    except Exception as e:
        # Print only a short preview of the text to avoid messy logs
        preview = text_clean[:30].replace("\n", " ")
        print(f"Error translating '{preview}...': {e}")
        return None


# ---------- Main preprocessing ----------

def preprocess_posts(input_path: str, output_path: str):
    """
    Load raw posts CSV, drop unused columns, add English translation,
    and save to a new processed CSV.
    """
    print(f"Loading posts from: {input_path}")
    posts_df = dh.load_data(input_path)

    # Drop clearly irrelevant or noisy columns, if they exist
    columns_to_drop = ['product', 'ratescore', 'crawl_time', 'device', 'location']
    existing_drop_cols = [c for c in columns_to_drop if c in posts_df.columns]
    if existing_drop_cols:
        print(f"Dropping columns: {existing_drop_cols}")
        posts_df = posts_df.drop(columns=existing_drop_cols, axis=1)
    else:
        print("No extra columns to drop.")

    # Apply translation to create English content column
    print("Translating 'content' column from Chinese to English...")
    posts_df["content_en"] = posts_df["content"].apply(content_translation)

    # Optional: quick info to sanity check
    print("\nSample of translated posts:")
    print(posts_df.loc[0:5, ["content", "content_en"]])

    # Save processed data
    print(f"\nSaving processed posts to: {output_path}")
    posts_df.to_csv(output_path, index=False)
    print("Done! ✅")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    input_csv = BASE_DIR.parent / "data" / "raw" / "posts.csv"
    output_csv = BASE_DIR.parent / "data" / "processed" / "posts_processed.csv"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    preprocess_posts(input_csv, output_csv)