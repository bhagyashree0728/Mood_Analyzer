import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from transformers import pipeline
from collect_data import DataCollector
from generate_report import ReportGenerator

class InteractiveAnalyzer:
    def __init__(self):
        self.collector = DataCollector()
        self.report_generator = ReportGenerator()
        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="distilbert-base-uncased-finetuned-sst-2-english")
        
    def get_user_input(self):
        """Get social media posts from user input"""
        print("\n=== Social Media Post Analysis ===")
        print("Enter your social media posts (type 'done' when finished):")
        
        while True:
            post = input("\nEnter post (or 'done' to finish): ")
            if post.lower() == 'done':
                break
                
            source = input("Enter source (e.g., Twitter, Facebook, Instagram): ")
            self.collector.add_post(post, source)
            print("Post added successfully!")
    
    def analyze_data(self):
        """Analyze the collected data"""
        if not self.collector.posts:
            print("No posts to analyze. Please add some posts first.")
            return
            
        print("\nAnalyzing posts...")
        
        # Analyze sentiments
        df = pd.DataFrame(self.collector.posts)
        sentiments = []
        confidence_scores = []
        
        for post in df['text']:
            result = self.sentiment_analyzer(post)[0]
            sentiments.append(result['label'])
            confidence_scores.append(result['score'])
        
        df['sentiment'] = sentiments
        df['confidence'] = confidence_scores
        
        # Save analyzed data
        df.to_csv('user_posts.csv', index=False)
        
        # Generate report
        report_file = self.report_generator.generate_analysis_report('user_posts.csv')
        
        # Show quick analysis
        self._show_quick_analysis(df)
        
        return report_file
    
    def _show_quick_analysis(self, df):
        """Show quick analysis of the data"""
        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # 1. Sentiment Distribution
        plt.subplot(2, 2, 1)
        sentiment_counts = df['sentiment'].value_counts()
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'])
        plt.title('Sentiment Distribution')
        
        # 2. Post Length Distribution
        plt.subplot(2, 2, 2)
        df['text_length'] = df['text'].str.len()
        sns.histplot(data=df, x='text_length', bins=20)
        plt.title('Post Length Distribution')
        
        # 3. Source Distribution
        plt.subplot(2, 2, 3)
        source_counts = df['source'].value_counts()
        source_counts.plot(kind='bar')
        plt.title('Posts by Source')
        plt.xticks(rotation=45)
        
        # 4. Confidence Distribution
        plt.subplot(2, 2, 4)
        sns.boxplot(data=df, x='sentiment', y='confidence')
        plt.title('Confidence by Sentiment')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed statistics
        print("\nDetailed Analysis:")
        print("\n1. Sentiment Analysis:")
        print(f"Total Posts: {len(df)}")
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment}: {count} posts ({count/len(df)*100:.1f}%)")
        
        print("\n2. Source Analysis:")
        for source, count in source_counts.items():
            print(f"{source}: {count} posts")
        
        print("\n3. Confidence Analysis:")
        print(f"Average Confidence: {df['confidence'].mean():.2f}")
        print(f"Highest Confidence: {df['confidence'].max():.2f}")
        print(f"Lowest Confidence: {df['confidence'].min():.2f}")
        
        print("\n4. Post Length Analysis:")
        print(f"Average Length: {df['text_length'].mean():.1f} characters")
        print(f"Shortest Post: {df['text_length'].min()} characters")
        print(f"Longest Post: {df['text_length'].max()} characters")

def main():
    print("Welcome to the Social Media Post Analyzer!")
    
    # Create analyzer
    analyzer = InteractiveAnalyzer()
    
    while True:
        print("\nChoose an option:")
        print("1. Add new posts")
        print("2. Analyze current posts")
        print("3. View saved report")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            analyzer.get_user_input()
        elif choice == '2':
            report_file = analyzer.analyze_data()
            if report_file:
                print(f"\nAnalysis complete! Report saved to: {report_file}")
        elif choice == '3':
            if os.path.exists('reports'):
                reports = os.listdir('reports')
                if reports:
                    latest_report = max(reports, key=lambda x: os.path.getctime(os.path.join('reports', x)))
                    print(f"\nLatest report: reports/{latest_report}")
                    print("Open this file in your web browser to view the full report.")
                else:
                    print("\nNo reports found. Please run an analysis first.")
            else:
                print("\nNo reports directory found. Please run an analysis first.")
        elif choice == '4':
            print("\nThank you for using the Social Media Post Analyzer!")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main() 