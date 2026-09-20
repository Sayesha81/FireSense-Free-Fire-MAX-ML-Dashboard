# 🎮 FireSense: Free Fire MAX Player Review Intelligence System

FireSense is an end-to-end Machine Learning and NLP pipeline featuring an interactive **Gradio dashboard** designed to analyze player feedback and sentiment for Garena Free Fire MAX using Google Play Store reviews.

---

## 🌟 Key Features
- **📊 Real-time Executive Dashboard:** Key performance metrics (Total Reviews, Average Rating, Positive/Negative % split) and distribution charts.
- **🤖 Sentiment Predictor:** Predicts single-review sentiment (Positive, Neutral, Negative) along with confidence scores.
- **🔍 Complaint Analyzer:** Categorizes technical complaints into sub-categories (*Bugs/Glitches, Lag/Ping, Connection, Pay-to-Win*, etc.).
- **🔎 Review Explorer:** Dynamic search and filtering by rating, sentiment, and custom keywords.
- **📈 Model Evaluation:** Displays accuracy, weighted F1-scores, and confusion matrices comparing algorithms (*Logistic Regression vs. LinearSVC*).

---

## 🛠 Tech Stack
- **Language:** Python
- **Machine Learning & NLP:** Scikit-learn, TF-IDF Vectorizer, Logistic Regression, LinearSVC
- **GUI & Visualization:** Gradio, Matplotlib, Seaborn
- **Data Pipeline:** Pandas, NumPy, Kagglehub API

---

## 🚀 How to Run in Google Colab

Click the button below to open and run the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sayesha81/FireSense-Free-Fire-MAX-ML-Dashboard/blob/main/Free_Fire_MAX_Player_Review_Intelligence_System.ipynb)

1. Open the notebook in Colab.
2. Run all cells sequentially.
3. Use the generated `gradio.live` link to access the live dashboard in your browser.

---

## 💻 Local Installation & Setup

1. **Clone the repository:**
  ```bash
  git clone https://github.com/Sayesha81/FireSense-Free-Fire-MAX-ML-Dashboard.git
  cd FireSense-Free-Fire-MAX-ML-Dashboard 
  ```

2. **Install dependencies:**
   ```bash
     pip install -r requirements.txt
   ```

3. **Train the model:**
   ```bash
     python train.py
   ```

4. **Launch the Gradio App:**
   ```bash
     python app.py
   ```
