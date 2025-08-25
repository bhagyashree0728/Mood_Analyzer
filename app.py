from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import pandas as pd
from datetime import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Initialize sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis", 
                            model="distilbert-base-uncased-finetuned-sst-2-english")

# Store posts in memory (in production, you'd want to use a database)
posts = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    post = data.get('post')
    source = data.get('source')
    
    if not post:
        return jsonify({'error': 'Please enter a post!'}), 400
        
    # Analyze sentiment
    result = sentiment_analyzer(post)[0]
    
    # Store post
    post_data = {
        'text': post,
        'source': source,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sentiment': result['label'],
        'confidence': result['score']
    }
    posts.append(post_data)
    
    # Generate visualizations
    if len(posts) > 0:
        df = pd.DataFrame(posts)
        
        # Create sentiment distribution plot
        plt.figure(figsize=(6, 4))
        sentiment_counts = df['sentiment'].value_counts()
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'])
        plt.title('Sentiment Distribution')
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        sentiment_plot = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'sentiment': result['label'],
            'confidence': result['score'],
            'sentiment_plot': sentiment_plot,
            'total_posts': len(posts)
        })
    
    return jsonify({
        'success': True,
        'sentiment': result['label'],
        'confidence': result['score'],
        'total_posts': len(posts)
    })

@app.route('/posts', methods=['GET'])
def get_posts():
    return jsonify(posts)

if __name__ == '__main__':
    app.run(debug=True) 