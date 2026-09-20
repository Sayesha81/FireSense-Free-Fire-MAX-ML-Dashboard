# ==============================
# CELL 3: Load Dataset
# ==============================

import pandas as pd

# Locate the CSV file inside the downloaded folder
import glob
csv_path = glob.glob('/content/data/*.csv')[0]
print("Found file:", csv_path)

# Load dataset into a DataFrame
df = pd.read_csv(csv_path)

# Quick look at the data
print("Shape:", df.shape)
df.head()

# ==============================
# CELL 4: Dataset Inspection
# ==============================

# Column names and data types
print("Columns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)

# General info
print("\nInfo:")
df.info()

# Basic statistics for numeric column(s)
print("\nScore statistics:\n", df['score'].describe())

# ==============================
# CELL 5: Missing Values & Duplicates Check
# ==============================

# Check missing values per column
print("Missing values per column:\n", df.isnull().sum())

# Check duplicate rows
print("\nTotal duplicate rows:", df.duplicated().sum())

# Check duplicate reviewIds specifically
print("Duplicate reviewId count:", df['reviewId'].duplicated().sum())

# Check for empty/whitespace-only content
empty_content = df['content'].isnull() | (df['content'].astype(str).str.strip() == '')
print("Empty/blank content rows:", empty_content.sum())

# ==============================
# CELL 6: Data Cleaning
# ==============================

print("Before cleaning:", df.shape)

# Drop rows with missing or blank content
df = df.dropna(subset=['content'])
df = df[df['content'].astype(str).str.strip() != '']

# Drop duplicate rows based on reviewId (keep first occurrence)
df = df.drop_duplicates(subset=['reviewId'], keep='first')

# Drop duplicate content (exact same review text)
df = df.drop_duplicates(subset=['content'], keep='first')

# Ensure score is numeric and within valid range (1-5)
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df = df.dropna(subset=['score'])
df = df[df['score'].between(1, 5)]
df['score'] = df['score'].astype(int)

# Parse 'at' column as datetime
df['at'] = pd.to_datetime(df['at'], errors='coerce')
df = df.dropna(subset=['at'])

# Reset index after cleaning
df = df.reset_index(drop=True)

print("After cleaning:", df.shape)
df.head()

# ==============================
# CELL 7: Add Sentiment Labels
# ==============================

def map_sentiment(score):
    # 1-2 = Negative, 3 = Neutral, 4-5 = Positive
    if score <= 2:
        return 'Negative'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Positive'

# Vectorized mapping using pandas (fast, no loops)
df['sentiment'] = df['score'].map({1:'Negative', 2:'Negative', 3:'Neutral', 4:'Positive', 5:'Positive'})

# Add review length column (word count) - vectorized
df['review_length'] = df['content'].astype(str).str.split().str.len()

print(df[['score', 'sentiment', 'review_length']].head())
print("\nSentiment counts:\n", df['sentiment'].value_counts())

# ==============================
# CELL 8: EDA - Rating Distribution
# ==============================

import matplotlib.pyplot as plt

# Count of each rating (1-5) - vectorized value_counts on full dataset (cheap)
rating_counts = df['score'].value_counts().sort_index()
print("Rating distribution:\n", rating_counts)

# Plot rating distribution
plt.figure(figsize=(6,4))
rating_counts.plot(kind='bar', color='orange')
plt.title('Rating Distribution')
plt.xlabel('Score')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ==============================
# CELL 9: EDA - Sentiment Distribution
# ==============================

# Sentiment counts (cheap, already computed column) - vectorized
sentiment_counts = df['sentiment'].value_counts()
print("Sentiment distribution:\n", sentiment_counts)

# Plot sentiment distribution as pie chart
plt.figure(figsize=(5,5))
colors = {'Positive':'green', 'Neutral':'gray', 'Negative':'red'}
plt.pie(sentiment_counts.values, labels=sentiment_counts.index,
        autopct='%1.1f%%', colors=[colors[s] for s in sentiment_counts.index])
plt.title('Sentiment Distribution')
plt.tight_layout()
plt.show()

# ==============================
# CELL 10: EDA - Review Length Analysis
# ==============================

# Review length statistics (already computed column, cheap)
print("Review length statistics:\n", df['review_length'].describe())

# Plot review length distribution (capped at 100 words for readability)
plt.figure(figsize=(6,4))
df[df['review_length'] <= 100]['review_length'].hist(bins=30, color='skyblue')
plt.title('Review Length Distribution (<=100 words)')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Average review length by sentiment - vectorized groupby
print("\nAverage review length by sentiment:\n", df.groupby('sentiment')['review_length'].mean())

# ==============================
# CELL 11: EDA - Monthly Review Trend
# ==============================

# Extract year-month from 'at' column - vectorized
df['year_month'] = df['at'].dt.to_period('M')

# Count reviews per month - vectorized groupby (cheap on full dataset)
monthly_trend = df.groupby('year_month').size()

print("Monthly review counts (first 10):\n", monthly_trend.head(10))

# Plot monthly trend
plt.figure(figsize=(10,4))
monthly_trend.plot(kind='line', color='purple')
plt.title('Monthly Review Trend')
plt.xlabel('Month')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.show()

# ==============================
# CELL 12: Create Balanced ML Dataset (25k/25k/25k)
# ==============================

# Sample up to 25,000 rows per sentiment class using random_state=42
n_per_class = 25000

balanced_parts = []
for sentiment_label in ['Negative', 'Neutral', 'Positive']:
    subset = df[df['sentiment'] == sentiment_label]
    sample_size = min(n_per_class, len(subset))
    sampled = subset.sample(n=sample_size, random_state=42)
    balanced_parts.append(sampled)
    print(f"{sentiment_label}: available={len(subset)}, sampled={sample_size}")

# Combine into one balanced dataframe
balanced_df = pd.concat(balanced_parts, axis=0)

# Shuffle the combined dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced dataset shape:", balanced_df.shape)
print(balanced_df['sentiment'].value_counts())

# ==============================
# CELL 13: Text Cleaning (on Balanced Dataset only)
# ==============================

import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)      # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)             # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()          # remove extra whitespace
    return text

# Apply cleaning only on the 75k balanced dataset (fast, vectorized apply)
balanced_df['clean_content'] = balanced_df['content'].apply(clean_text)

# Remove rows that became empty after cleaning
balanced_df = balanced_df[balanced_df['clean_content'].str.strip() != ''].reset_index(drop=True)

print("Balanced dataset shape after cleaning:", balanced_df.shape)
balanced_df[['content', 'clean_content', 'sentiment']].head()

