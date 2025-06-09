from interactive_analysis import InteractiveAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_example_analysis():
    print("=== Example Analysis Walkthrough ===\n")
    
    # Create analyzer
    analyzer = InteractiveAnalyzer()
    
    # Example 1: Adding Posts
    print("Step 1: Adding Sample Posts")
    sample_posts = [
        ("Just got promoted at work! Feeling amazing! 🎉", "Twitter"),
        ("Lost my phone today, feeling terrible.", "Facebook"),
        ("The weather is beautiful today! Perfect for a walk.", "Instagram"),
        ("Failed my exam, feeling disappointed.", "Twitter"),
        ("Got tickets to my favorite band's concert! So excited! 🎵", "Facebook"),
        ("I'm so sad to be here! 🌸", "Twitter"),
        ("I'm so low ! 🌸", "Twitter"),


    ]
    
    for post, source in sample_posts:
        analyzer.collector.add_post(post, source)
        print(f"Added post from {source}")
    
    print("\nStep 2: Running Analysis (Option 2)")
    print("This will generate:")
    print("1. Interactive charts showing:")
    print("   - Sentiment distribution")
    print("   - Post length analysis")
    print("   - Source distribution")
    print("   - Time patterns")
    print("2. Summary statistics")
    print("3. HTML report")
    
    # Run analysis
    report_file = analyzer.analyze_data()
    
    print("\nStep 3: Viewing Report (Option 3)")
    print(f"Report saved to: {report_file}")
    print("\nThe report includes:")
    print("1. Detailed sentiment analysis")
    print("2. Post statistics")
    print("3. Source analysis")
    print("4. Time-based patterns")
    print("5. Sample posts with their analysis")
    
    print("\nTo continue analysis:")
    print("1. Add more posts using Option 1")
    print("2. Run analysis again using Option 2")
    print("3. Compare new report with previous one")
    print("4. Look for patterns in sentiment changes")

def show_analysis_examples():
    print("\n=== Analysis Examples ===\n")
    
    # Create sample data
    data = {
        'text': [
            "Feeling great today!",
            "Not my best day",
            "Amazing weather!",
            "Feeling down",
            "Excited for the weekend!"
        ],
        'source': ['Twitter', 'Facebook', 'Instagram', 'Twitter', 'Facebook'],
        'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='D')
    }
    
    df = pd.DataFrame(data)
    
    # Example 1: Sentiment Analysis
    print("Example 1: Sentiment Analysis")
    plt.figure(figsize=(10, 6))
    sentiment_counts = pd.Series(['POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE']).value_counts()
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
    plt.title('Example Sentiment Distribution')
    plt.show()
    
    # Example 2: Source Analysis
    print("\nExample 2: Source Analysis")
    plt.figure(figsize=(10, 6))
    source_counts = df['source'].value_counts()
    source_counts.plot(kind='bar')
    plt.title('Example Source Distribution')
    plt.show()
    
    print("\nTo continue with your own analysis:")
    print("1. Run 'python interactive_analysis.py'")
    print("2. Choose Option 1 to add your posts")
    print("3. Choose Option 2 to analyze")
    print("4. Choose Option 3 to view the report")
    print("5. Repeat steps 2-4 to add more data and track changes")

if __name__ == "__main__":
    run_example_analysis()
    show_analysis_examples() 