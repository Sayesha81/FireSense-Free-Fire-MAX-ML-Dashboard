# ==============================
# CELL 26: FireSense GUI (Gradio) - Full Application
# ==============================

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

# ---------- Helper: Rating Chart ----------
def get_rating_chart():
    fig, ax = plt.subplots(figsize=(5,3.5))
    rating_dist.plot(kind='bar', color='orange', ax=ax)
    ax.set_title('Rating Distribution')
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    plt.tight_layout()
    return fig

# ---------- Helper: Sentiment Chart ----------
def get_sentiment_chart():
    fig, ax = plt.subplots(figsize=(5,3.5))
    colors_map = {'Positive':'green', 'Neutral':'gray', 'Negative':'red'}
    ax.pie(sentiment_dist.values, labels=sentiment_dist.index, autopct='%1.1f%%',
           colors=[colors_map[s] for s in sentiment_dist.index])
    ax.set_title('Sentiment Distribution')
    plt.tight_layout()
    return fig

# ---------- Helper: Complaint Chart ----------
def get_complaint_chart():
    fig, ax = plt.subplots(figsize=(6,4))
    complaint_counts.plot(kind='barh', color='crimson', ax=ax)
    ax.set_title('Common Complaints (Sampled 75k Reviews)')
    ax.set_xlabel('Count')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

# ---------- Helper: Confusion Matrix Chart ----------
def get_confusion_matrix_chart():
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=labels_order, yticklabels=labels_order, ax=ax)
    ax.set_title(f'Confusion Matrix - {best_model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    return fig

# ---------- Tab 2: Sentiment Predictor ----------
def gui_predict_sentiment(text):
    if not text or text.strip() == '':
        return "N/A", "N/A", "N/A", 0
    pred, conf = predict_sentiment(text)
    complaints = detect_complaints(text)
    length = len(text.split())
    conf_str = f"{conf:.3f}" if conf is not None else "N/A"
    return pred, conf_str, ", ".join(complaints), length

# ---------- Tab 3: Complaint Analyzer ----------
def gui_analyze_complaints(text):
    if not text or text.strip() == '':
        return "N/A"
    complaints = detect_complaints(text)
    return ", ".join(complaints)

# ---------- Tab 4: Review Explorer ----------
def gui_explore_reviews(sentiment_filter, score_filter, keyword):
    filtered = df
    if sentiment_filter != "All":
        filtered = filtered[filtered['sentiment'] == sentiment_filter]
    if score_filter != "All":
        filtered = filtered[filtered['score'] == int(score_filter)]
    if keyword and keyword.strip() != '':
        filtered = filtered[filtered['content'].str.contains(keyword, case=False, na=False)]

    # Limit to 50 rows max for performance
    result = filtered[['content', 'score', 'sentiment', 'at']].head(50).copy()
    result['at'] = result['at'].dt.strftime('%Y-%m-%d')
    result.columns = ['Content', 'Score', 'Sentiment', 'Date']
    return result

# ---------- Build Gradio App ----------
with gr.Blocks(title="FireSense") as demo:
    gr.Markdown("# 🔥 FireSense")
    gr.Markdown("### Free Fire MAX Player Review Intelligence System")

    with gr.Tabs():
        # ---- TAB 1: DASHBOARD ----
        with gr.Tab("Dashboard"):
            with gr.Row():
                gr.Textbox(value=str(total_reviews), label="Total Reviews", interactive=False)
                gr.Textbox(value=str(average_rating), label="Average Rating", interactive=False)
                gr.Textbox(value=f"{positive_pct}%", label="Positive %", interactive=False)
                gr.Textbox(value=f"{negative_pct}%", label="Negative %", interactive=False)
            with gr.Row():
                gr.Plot(value=get_rating_chart(), label="Rating Chart")
                gr.Plot(value=get_sentiment_chart(), label="Sentiment Chart")

        # ---- TAB 2: SENTIMENT PREDICTOR ----
        with gr.Tab("Sentiment Predictor"):
            with gr.Row():
                sp_input = gr.Textbox(lines=4, label="Enter Review Text")
            sp_btn = gr.Button("Predict Sentiment")
            with gr.Row():
                sp_sentiment = gr.Textbox(label="Predicted Sentiment")
                sp_confidence = gr.Textbox(label="Confidence")
            with gr.Row():
                sp_complaints = gr.Textbox(label="Detected Complaints")
                sp_length = gr.Textbox(label="Review Length (words)")
            sp_btn.click(gui_predict_sentiment, inputs=sp_input,
                         outputs=[sp_sentiment, sp_confidence, sp_complaints, sp_length])

        # ---- TAB 3: COMPLAINT ANALYZER ----
        with gr.Tab("Complaint Analyzer"):
            ca_input = gr.Textbox(lines=4, label="Enter Review Text")
            ca_btn = gr.Button("Analyze Complaints")
            ca_output = gr.Textbox(label="Detected Issues")
            ca_btn.click(gui_analyze_complaints, inputs=ca_input, outputs=ca_output)
            gr.Plot(value=get_complaint_chart(), label="Common Complaints Chart")

        # ---- TAB 4: REVIEW EXPLORER ----
        with gr.Tab("Review Explorer"):
            with gr.Row():
                re_sentiment = gr.Dropdown(choices=["All", "Positive", "Neutral", "Negative"],
                                            value="All", label="Sentiment Filter")
                re_score = gr.Dropdown(choices=["All", "1", "2", "3", "4", "5"],
                                        value="All", label="Score Filter")
                re_keyword = gr.Textbox(label="Keyword Search")
            re_btn = gr.Button("Search Reviews")
            re_output = gr.Dataframe(label="Matching Reviews (max 50)")
            re_btn.click(gui_explore_reviews, inputs=[re_sentiment, re_score, re_keyword], outputs=re_output)

        # ---- TAB 5: MODEL PERFORMANCE ----
        with gr.Tab("Model Performance"):
            gr.Dataframe(value=model_perf_df, label="Model Comparison")
            gr.Textbox(value=best_model_name, label="Best Model")
            gr.Textbox(value=f"{best_acc:.4f}", label="Accuracy")
            gr.Textbox(value=f"{best_f1:.4f}", label="Weighted F1")
            gr.Plot(value=get_confusion_matrix_chart(), label="Confusion Matrix")

# Launch the app
demo.launch(share=True, debug=False)
