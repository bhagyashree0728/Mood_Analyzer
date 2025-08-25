import streamlit as st
import pandas as pd
from transformers import pipeline
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
        
        # Initialize sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
# Session state for posts
if 'posts' not in st.session_state:
    st.session_state['posts'] = []

st.title("Mood Detector Web App")
        
# Input form
with st.form("post_form"):
    post = st.text_area("Enter your social media post:")
    source = st.selectbox("Select source:", ['Twitter', 'Facebook', 'Instagram', 'LinkedIn', 'Other'])
    submitted = st.form_submit_button("Add Post")
    if submitted and post:
        st.session_state['posts'].append({
            'text': post,
            'source': source,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.success("Post added!")

# Show added posts
if st.session_state['posts']:
    st.subheader("Added Posts")
    df = pd.DataFrame(st.session_state['posts'])
    st.dataframe(df[['timestamp', 'source', 'text']])
            
    # Analyze button
    if st.button("Analyze Posts"):
        sentiments = []
        confidence_scores = []
        for post in df['text']:
            result = sentiment_analyzer(post)[0]
            sentiments.append(result['label'])
            confidence_scores.append(result['score'])
        df['sentiment'] = sentiments
        df['confidence'] = confidence_scores

        st.subheader("Analysis Results")
        st.dataframe(df)
        
        # Visualizations
        st.subheader("Visualizations")
        # Sentiment Distribution
        fig1, ax1 = plt.subplots()
        sentiment_counts = df['sentiment'].value_counts()
        ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        ax1.set_title('Sentiment Distribution')
        st.pyplot(fig1)
        
        # Post Length Distribution
        fig2, ax2 = plt.subplots()
        df['text_length'] = df['text'].str.len()
        sns.histplot(data=df, x='text_length', bins=20, ax=ax2)
        ax2.set_title('Post Length Distribution')
        st.pyplot(fig2)
        
        # Source Distribution
        fig3, ax3 = plt.subplots()
        source_counts = df['source'].value_counts()
        source_counts.plot(kind='bar', ax=ax3)
        ax3.set_title('Posts by Source')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        st.pyplot(fig3)
        
        # Confidence Analysis
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=df, x='sentiment', y='confidence', ax=ax4)
        ax4.set_title('Confidence by Sentiment')
        st.pyplot(fig4) 