import pandas as pd
import json
from datetime import datetime

class DataCollector:
    def __init__(self):
        self.posts = []
        
    def add_post(self, text, source="manual", timestamp=None):
        """Add a social media post to the collection"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        self.posts.append({
            'text': text,
            'source': source,
            'timestamp': timestamp
        })
        
    def save_to_csv(self, filename='social_media_posts.csv'):
        """Save collected posts to a CSV file"""
        df = pd.DataFrame(self.posts)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        
    def save_to_json(self, filename='social_media_posts.json'):
        """Save collected posts to a JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.posts, f, indent=2)
        print(f"Data saved to {filename}")
        
    def load_from_csv(self, filename='social_media_posts.csv'):
        """Load posts from a CSV file"""
        df = pd.read_csv(filename)
        self.posts = df.to_dict('records')
        print(f"Loaded {len(self.posts)} posts from {filename}")
        
    def load_from_json(self, filename='social_media_posts.json'):
        """Load posts from a JSON file"""
        with open(filename, 'r', encoding='utf-8') as f:
            self.posts = json.load(f)
        print(f"Loaded {len(self.posts)} posts from {filename}")

def main():
    # Create a data collector
    collector = DataCollector()
    
    # Example: Add some sample posts
    sample_posts = [
        "Just got promoted at work! Feeling amazing! 🎉",
        "Lost my phone today, feeling terrible.",
        "The weather is beautiful today! Perfect for a walk.",
        "Failed my exam, feeling disappointed.",
        "Got tickets to my favorite band's concert! So excited! 🎵"
    ]
    
    print("Adding sample posts...")
    for post in sample_posts:
        collector.add_post(post)
    
    # Save the data
    collector.save_to_csv()
    collector.save_to_json()
    
    print("\nYou can now:")
    print("1. Add more posts using collector.add_post()")
    print("2. Save data using collector.save_to_csv() or collector.save_to_json()")
    print("3. Load data using collector.load_from_csv() or collector.load_from_json()")
    print("4. Use the data with test_mood_detection.py")

if __name__ == "__main__":
    main() 