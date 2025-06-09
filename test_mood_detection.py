from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def test_mood_detection():
    # Initialize the sentiment analyzer
    print("Initializing sentiment analyzer...")
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Test cases
    test_posts = [
        "Just got promoted at work! Feeling amazing! 🎉",
        "Lost my phone today, feeling terrible.",
        "The weather is beautiful today! Perfect for a walk.",
        "Failed my exam, feeling disappointed.",
        "Got tickets to my favorite band's concert! So excited! 🎵"
    ]
    
    print("\nAnalyzing test posts...")
    results = []
    for post in test_posts:
        # Get sentiment analysis
        sentiment = sentiment_analyzer(post)[0]
        results.append({
            'Post': post,
            'Sentiment': sentiment['label'],
            'Confidence': f"{sentiment['score']*100:.2f}%"
        })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    print("\nResults:")
    print(df_results)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    df_results['Confidence'] = df_results['Confidence'].str.rstrip('%').astype(float)
    sns.barplot(x='Sentiment', y='Confidence', data=df_results)
    plt.title('Confidence Levels by Sentiment')
    plt.ylabel('Confidence (%)')
    plt.show()

if __name__ == "__main__":
    test_mood_detection() 