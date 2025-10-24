import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from collections import Counter
import io

# For machine learning predictions (handle missing sklearn gracefully)
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import classification_report, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(
    page_title="Hospital Review Analysis & Prediction - India",
    page_icon="🏥",
    layout="wide"
)

# ---------- HELPER FUNCTIONS ------------

def create_indian_hospital_dataset():
    hospitals = [
        ("Tata Memorial Hospital", "Mumbai", "Maharashtra", "Multi-specialty"),
        ("Kokilaben Hospital", "Mumbai", "Maharashtra", "Multi-specialty"),
        ("Ruby Hall Clinic", "Pune", "Maharashtra", "Multi-specialty"),
        ("AIIMS Delhi", "Delhi", "Delhi", "Teaching"),
        ("Fortis Hospital", "Delhi", "Delhi", "Multi-specialty"),
        ("Max Hospital", "Gurgaon", "Haryana", "Multi-specialty"),
        ("Manipal Hospital", "Bangalore", "Karnataka", "Multi-specialty"),
        ("Apollo Hospital", "Bangalore", "Karnataka", "Multi-specialty"),
        ("Narayana Health", "Bangalore", "Karnataka", "Multi-specialty"),
        ("Apollo Hospital", "Chennai", "Tamil Nadu", "Multi-specialty"),
        ("CMC Vellore", "Vellore", "Tamil Nadu", "Teaching"),
        ("AMRI Hospital", "Kolkata", "West Bengal", "Multi-specialty"),
        ("Apollo Hospital", "Hyderabad", "Telangana", "Multi-specialty"),
        ("Care Hospital", "Hyderabad", "Telangana", "Multi-specialty"),
        ("PGIMER", "Chandigarh", "Chandigarh", "Teaching"),
    ]
    positive_templates = [
        "Excellent service by doctors and staff. Very clean facility.",
        "Great experience. Doctors were very professional and caring.",
        "Highly recommend this hospital. Treatment was world-class.",
        "Amazing staff, modern equipment, and clean environment.",
        "Best hospital in the city. Doctors are highly qualified.",
        "Very satisfied with the treatment received. Quick service.",
        "Professional staff and excellent medical care provided.",
        "State-of-the-art facilities and compassionate doctors.",
    ]
    neutral_templates = [
        "Average experience. Treatment was okay but waiting time was long.",
        "Decent hospital but could improve facilities.",
        "Okay service. Nothing exceptional but got treated properly.",
        "Standard treatment received. Room for improvement.",
    ]
    negative_templates = [
        "Poor service. Staff was rude and unprofessional.",
        "Very disappointed. Long waiting hours and careless treatment.",
        "Not recommended. Overpriced and below average care.",
        "Bad experience. Unhygienic conditions and poor management.",
        "Terrible service. Would not visit again.",
        "Unsatisfied with treatment. Doctors seemed uninterested.",
    ]
    data = []
    for hospital, city, state, h_type in hospitals:
        num_reviews = np.random.randint(15, 26)
        for _ in range(num_reviews):
            rating_prob = np.random.random()
            if rating_prob < 0.50:
                rating = 5
                review = np.random.choice(positive_templates)
            elif rating_prob < 0.75:
                rating = 4
                review = np.random.choice(positive_templates)
            elif rating_prob < 0.90:
                rating = 3
                review = np.random.choice(neutral_templates)
            elif rating_prob < 0.95:
                rating = 2
                review = np.random.choice(negative_templates)
            else:
                rating = 1
                review = np.random.choice(negative_templates)
            data.append({
                'hospital_name': hospital,
                'city': city,
                'state': state,
                'hospital_type': h_type,
                'rating': rating,
                'review_text': review,
                'review_date': (datetime.now() - pd.Timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d'),
                'reviewer_name': f"Patient_{np.random.randint(1000, 9999)}"
            })
    return data


def analyze_sentiment(text):
    # Fallback if textblob is not installed
    try:
        from textblob import TextBlob
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"
    except Exception:
        return "Neutral"

def get_sentiment_score(text):
    try:
        from textblob import TextBlob
        blob = TextBlob(str(text))
        return blob.sentiment.polarity
    except Exception:
        return 0.0

def extract_keywords(reviews, top_n=10):
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 
                  'of', 'for', 'is', 'was', 'are', 'were', 'very', 'this', 
                  'that', 'have', 'has', 'had', 'been', 'be'}
    all_words = []
    for review in reviews:
        if not review:
            continue
        words = re.findall(r"\b[a-z]{4,}\b", str(review).lower())
        all_words.extend([w for w in words if w not in stop_words])
    return Counter(all_words).most_common(top_n)

# ---------- UI CODE ------------

if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = []

if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = {}

st.title("🏥 Hospital Review Analysis & Prediction System")
st.markdown("Upload reviews or use sample data, analyze sentiment, train ML, export results.")

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Data Collection", 
    "📊 Data Analysis", 
    "🤖 ML Predictions", 
    "📈 Insights Dashboard",
    "💾 Export Data"
])

# --- TAB 1: Data Collection ---
with tab1:
    st.subheader("📥 Collect Hospital Review Data")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Upload CSV")
        uploaded_file = st.file_uploader(
            "Upload CSV file (columns: hospital_name, review_text, rating, etc.)", 
            type=['csv']
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} reviews")
                st.dataframe(df.head())
                st.session_state.scraped_data = df.to_dict('records')
            except Exception as e:
                st.error(f"Error loading file: {e}")

        st.divider()
        st.subheader("Or... Use Sample Indian Hospital Data")
        if st.button("Load Sample Dataset", type="primary"):
            sample_data = create_indian_hospital_dataset()
            st.session_state.scraped_data = sample_data
            st.success(f"✅ Loaded {len(sample_data)} sample reviews from Indian hospitals")
            st.dataframe(pd.DataFrame(sample_data).head(10))
    with col2:
        st.info("Features:\n"
                "- ✅ Upload CSV datasets\n"
                "- ✅ Sample data for 15 Indian hospitals\n"
                "- ✅ Sentiment analysis\n"
                "- ✅ Machine learning predictions\n"
                "- ✅ Export reports\n")

# --- TAB 2: Data Analysis ---
with tab2:
    st.header("Review Data Analysis")
    if not st.session_state.scraped_data:
        st.warning("No data loaded. Please upload or generate data.")
    else:
        df = pd.DataFrame(st.session_state.scraped_data)
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", len(df))
        avg_rating = df['rating'].mean() if 'rating' in df.columns else 0
        col2.metric("Average Rating", f"{avg_rating:.2f} ⭐")
        hospitals = df['hospital_name'].nunique() if 'hospital_name' in df.columns else 0
        col3.metric("Unique Hospitals", hospitals)
        states = df['state'].nunique() if 'state' in df.columns else 0
        col4.metric("States Covered", states)

        st.divider()
        st.subheader("😊 Sentiment Analysis")
        if 'review_text' in df.columns:
            df['sentiment'] = df['review_text'].apply(analyze_sentiment)
            df['sentiment_score'] = df['review_text'].apply(get_sentiment_score)
            sentiment_counts = df['sentiment'].value_counts()
            col1, col2 = st.columns([1,2])
            with col1:
                st.markdown("**Sentiment Distribution:**")
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / len(df))*100
                    st.metric(sentiment, f"{count} ({percentage:.1f}%)")
            with col2:
                st.markdown("**Sentiment Breakdown:**")
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / len(df))*100
                    st.progress(percentage/100, text=f"{sentiment}: {percentage:.1f}%")
        else:
            st.info("No 'review_text' column for sentiment analysis.")

        st.divider()
        st.subheader("🏥 Hospital-wise Performance")
        if 'hospital_name' in df.columns and 'rating' in df.columns:
            hospital_stats = df.groupby('hospital_name').agg({
                'rating': ['mean', 'count']
            }).round(2)
            hospital_stats.columns = ['Avg Rating', 'Review Count']
            hospital_stats = hospital_stats.sort_values('Avg Rating', ascending=False)
            st.dataframe(hospital_stats, use_container_width=True)

        st.divider()
        st.subheader("📍 State-wise Analysis")
        if 'state' in df.columns and 'rating' in df.columns and 'hospital_name' in df.columns:
            state_stats = df.groupby('state').agg({
                'rating': 'mean',
                'hospital_name': 'nunique'
            }).round(2)
            state_stats.columns = ['Avg Rating', 'Hospital Count']
            state_stats = state_stats.sort_values('Avg Rating', ascending=False)
            st.dataframe(state_stats, use_container_width=True)

        st.divider()
        st.subheader("💬 Common Review Themes")
        if 'review_text' in df.columns and 'rating' in df.columns:
            positive_reviews = df[df['rating'] >= 4]['review_text'].tolist()
            negative_reviews = df[df['rating'] <= 2]['review_text'].tolist()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Positive Keywords:**")
                pos_keywords = extract_keywords(positive_reviews, top_n=10)
                for word, freq in pos_keywords:
                    st.write(f"✅ {word}: {freq}")
            with col2:
                st.markdown("**Top Negative Keywords:**")
                neg_keywords = extract_keywords(negative_reviews, top_n=10)
                for word, freq in neg_keywords:
                    st.write(f"❌ {word}: {freq}")

# --- REMAINING TABS OMITTED FOR BREVITY, replicate as above for predictions/insights/export ---

# (Include your ML tab, insights, and export codes from your previous version here!)
