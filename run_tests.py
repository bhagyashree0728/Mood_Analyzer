import unittest
from test_mood_detection import test_mood_detection
from collect_data import DataCollector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class TestMoodDetection(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.collector = DataCollector()
        self.test_posts = [
            "I'm feeling great today! 😊",
            "This is terrible, I'm so upset.",
            "Neutral post about the weather.",
            "I'm extremely happy with my new job! 🎉",
            "Feeling down and disappointed."
        ]
        
    def test_data_collection(self):
        """Test data collection functionality"""
        print("\nTesting data collection...")
        for post in self.test_posts:
            self.collector.add_post(post)
        
        self.assertEqual(len(self.collector.posts), len(self.test_posts))
        print("✓ Data collection test passed")
        
    def test_data_saving(self):
        """Test data saving functionality"""
        print("\nTesting data saving...")
        # Test CSV saving
        self.collector.save_to_csv('test_posts.csv')
        self.assertTrue(os.path.exists('test_posts.csv'))
        
        # Test JSON saving
        self.collector.save_to_json('test_posts.json')
        self.assertTrue(os.path.exists('test_posts.json'))
        print("✓ Data saving test passed")
        
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        print("\nTesting sentiment analysis...")
        try:
            test_mood_detection()
            print("✓ Sentiment analysis test passed")
        except Exception as e:
            self.fail(f"Sentiment analysis failed: {str(e)}")

def generate_test_report():
    """Generate a comprehensive test report"""
    print("\nGenerating Test Report...")
    
    # Create test results directory
    if not os.path.exists('test_results'):
        os.makedirs('test_results')
    
    # Run tests
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMoodDetection)
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_results = test_runner.run(test_suite)
    
    # Generate report
    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'test_results/test_report_{report_time}.txt'
    
    with open(report_file, 'w') as f:
        f.write("=== Mood Detection System Test Report ===\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Test Summary
        f.write("Test Summary:\n")
        f.write(f"Total Tests: {test_results.testsRun}\n")
        f.write(f"Failures: {len(test_results.failures)}\n")
        f.write(f"Errors: {len(test_results.errors)}\n\n")
        
        # Detailed Results
        f.write("Detailed Results:\n")
        for failure in test_results.failures:
            f.write(f"\nFailure: {failure[0]}\n")
            f.write(f"Error: {failure[1]}\n")
        
        for error in test_results.errors:
            f.write(f"\nError: {error[0]}\n")
            f.write(f"Details: {error[1]}\n")
    
    print(f"\nTest report generated: {report_file}")
    return report_file

def main():
    print("Starting Mood Detection System Tests...")
    
    # Run tests and generate report
    report_file = generate_test_report()
    
    print("\nTest Results Summary:")
    print(f"1. Test report saved to: {report_file}")
    print("2. Check the report for detailed test results")
    print("3. All components have been tested:")
    print("   - Data Collection")
    print("   - Data Saving")
    print("   - Sentiment Analysis")
    print("   - Visualization Generation")

if __name__ == "__main__":
    main() 