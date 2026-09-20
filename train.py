# ==============================
# CELL 2: Setup Kaggle API & Download Dataset
# ==============================

import kagglehub

print("Downloading dataset from Kaggle...")
# Download dataset automatically (kagglehub handles authentication if needed)
path = kagglehub.dataset_download("bagush/free-fire-max-mobile-game-reviews-google-play")
print("Dataset downloaded to:", path)

# ==============================
# CELL 3: Load Dataset
# ==============================

import pandas as pd
import glob
import os

# Locate the CSV file inside the downloaded kagglehub folder
csv_path = glob.glob(os.path.join(path, '*.csv'))[0]
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

# ==============================
# CELL 14: Train/Test Split
# ==============================

from sklearn.model_selection import train_test_split

X = balanced_df['clean_content']
y = balanced_df['sentiment']

# Stratified split to preserve class balance, random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("\nTrain class distribution:\n", y_train.value_counts())
print("\nTest class distribution:\n", y_test.value_counts())

# ==============================
# CELL 15: TF-IDF Vectorization
# ==============================

from sklearn.feature_extraction.text import TfidfVectorizer

# Optimized TF-IDF settings for Colab performance
tfidf = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)

# Fit only on training data, transform both train and test
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)

# ==============================
# CELL 16: Train Logistic Regression Model
# ==============================

from sklearn.linear_model import LogisticRegression
import time

start_time = time.time()

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_tfidf, y_train)

print(f"Logistic Regression trained in {time.time() - start_time:.2f} seconds")

# ==============================
# CELL 17: Train LinearSVC Model
# ==============================

from sklearn.svm import LinearSVC
import time

start_time = time.time()

linear_svc = LinearSVC(random_state=42, max_iter=2000)
linear_svc.fit(X_train_tfidf, y_train)

print(f"LinearSVC trained in {time.time() - start_time:.2f} seconds")

# ==============================
# CELL 18: Compare Accuracy & Weighted F1
# ==============================

from sklearn.metrics import accuracy_score, f1_score

# Predictions from both models
log_reg_preds = log_reg.predict(X_test_tfidf)
svc_preds = linear_svc.predict(X_test_tfidf)

# Compute metrics
log_reg_acc = accuracy_score(y_test, log_reg_preds)
log_reg_f1 = f1_score(y_test, log_reg_preds, average='weighted')

svc_acc = accuracy_score(y_test, svc_preds)
svc_f1 = f1_score(y_test, svc_preds, average='weighted')

# Store results in a comparison table
model_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'LinearSVC'],
    'Accuracy': [log_reg_acc, svc_acc],
    'Weighted F1': [log_reg_f1, svc_f1]
})

print(model_comparison)

# ==============================
# CELL 19: Select Best Model
# ==============================

# Pick the model with the highest weighted F1 score
if svc_f1 >= log_reg_f1:
    best_model = linear_svc
    best_model_name = 'LinearSVC'
    best_preds = svc_preds
    best_acc = svc_acc
    best_f1 = svc_f1
else:
    best_model = log_reg
    best_model_name = 'Logistic Regression'
    best_preds = log_reg_preds
    best_acc = log_reg_acc
    best_f1 = log_reg_f1

print(f"Best Model Selected: {best_model_name}")
print(f"Accuracy: {best_acc:.4f}")
print(f"Weighted F1: {best_f1:.4f}")

# ==============================
# CELL 20: Classification Report (Best Model)
# ==============================

from sklearn.metrics import classification_report

report_text = classification_report(y_test, best_preds)
print(f"Classification Report - {best_model_name}\n")
print(report_text)

# ==============================
# CELL 21: Confusion Matrix (Best Model)
# ==============================

from sklearn.metrics import confusion_matrix
import seaborn as sns

labels_order = ['Negative', 'Neutral', 'Positive']
cm = confusion_matrix(y_test, best_preds, labels=labels_order)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=labels_order, yticklabels=labels_order)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ==============================
# CELL 22: Manual Sentiment Prediction Function
# ==============================

def predict_sentiment(text):
    # Clean the input text using the same cleaning function
    cleaned = clean_text(text)

    # Transform using the fitted TF-IDF vectorizer
    vec = tfidf.transform([cleaned])

    # Predict using the best model
    prediction = best_model.predict(vec)[0]

    # Get confidence score if the model supports probabilities
    confidence = None
    if hasattr(best_model, 'predict_proba'):
        proba = best_model.predict_proba(vec)[0]
        confidence = max(proba)
    elif hasattr(best_model, 'decision_function'):
        # For LinearSVC, use decision_function margin as a pseudo-confidence
        scores = best_model.decision_function(vec)[0]
        confidence = float(max(scores) - min(scores))  # relative margin

    return prediction, confidence

# Test the function with sample reviews
test_reviews = [
    "This game is amazing, I love the graphics and gameplay!",
    "Too much lag and the game keeps crashing, very frustrating.",
    "It's okay, nothing special but not bad either."
]

for review in test_reviews:
    pred, conf = predict_sentiment(review)
    print(f"Review: {review}\nPredicted: {pred} | Confidence: {conf}\n")

# ==============================
# CELL 23: Complaint Detection (Keyword-Based)
# ==============================

# Define keyword dictionaries for each complaint category (lightweight, no ML)
complaint_keywords = {
    'Lag / Ping': ['lag', 'lagging', 'ping', 'fps drop', 'slow', 'freeze', 'freezing', 'stutter'],
    'Bugs / Glitches': ['bug', 'glitch', 'glitchy', 'error', 'broken', 'issue'],
    'Crashes': ['crash', 'crashing', 'crashes', 'force close', 'shut down', 'stopped working'],
    'Connection': ['connection', 'server', 'disconnect', 'network', 'offline', 'wifi', 'internet'],
    'Cheating': ['cheat', 'cheater', 'hacker', 'hack', 'aimbot', 'wallhack'],
    'Pay to Win': ['pay to win', 'p2w', 'money', 'expensive', 'overpriced', 'diamond', 'purchase'],
    'Updates': ['update', 'version', 'patch', 'new update', 'latest update'],
    'Gameplay': ['gameplay', 'matchmaking', 'balance', 'unfair', 'controls', 'sensitivity']
}

def detect_complaints(text):
    text_lower = str(text).lower()
    detected = []
    for category, keywords in complaint_keywords.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(category)
    return detected if detected else ['No specific complaint detected']

# Test complaint detection
test_text = "The game keeps crashing and there are so many hackers, also too much lag."
print("Detected complaints:", detect_complaints(test_text))

# Precompute complaint categories on the BALANCED dataset only (75k rows, not full 523k)
balanced_df['complaints'] = balanced_df['content'].apply(detect_complaints)
print("\nSample:\n", balanced_df[['content', 'complaints']].head())

# ==============================
# CELL 24: Save Model & Vectorizer with Joblib
# ==============================

import os
import joblib

# Create a 'models' folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save best model and vectorizer into the models folder
joblib.dump(best_model, 'models/firesense_best_model.pkl')
joblib.dump(tfidf, 'models/firesense_tfidf_vectorizer.pkl')

print("Saved best model:", best_model_name)
print("Files saved to models/ directory successfully!")

# ==============================
# CELL 25: Precompute Stats for GUI (avoid recomputation)
# ==============================

# ---- Dashboard stats (computed once from full cleaned dataset) ----
total_reviews = len(df)
average_rating = round(df['score'].mean(), 2)
positive_pct = round((df['sentiment'] == 'Positive').mean() * 100, 2)
negative_pct = round((df['sentiment'] == 'Negative').mean() * 100, 2)
neutral_pct = round((df['sentiment'] == 'Neutral').mean() * 100, 2)

rating_dist = df['score'].value_counts().sort_index()
sentiment_dist = df['sentiment'].value_counts()

# ---- Complaint stats (computed once from balanced dataset, already has 'complaints' column) ----
from collections import Counter
all_complaints = [c for sublist in balanced_df['complaints'] for c in sublist if c != 'No specific complaint detected']
complaint_counts = pd.Series(Counter(all_complaints)).sort_values(ascending=False)

# ---- Model performance stats (already computed) ----
model_perf_df = model_comparison.copy()

print("Total Reviews:", total_reviews)
print("Average Rating:", average_rating)
print("Positive %:", positive_pct)
print("Negative %:", negative_pct)
print("Neutral %:", neutral_pct)
print("\nComplaint counts:\n", complaint_counts)
