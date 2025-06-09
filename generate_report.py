import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class ReportGenerator:
    def __init__(self):
        self.report_dir = 'reports'
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
            
    def generate_analysis_report(self, data_file='user_posts.csv'):
        """Generate a comprehensive analysis report"""
        print("\nGenerating Analysis Report...")
        
        # Load analyzed data
        df = pd.read_csv(data_file)
        
        # Create report
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f'{self.report_dir}/analysis_report_{report_time}.html'
        
        # Generate visualizations
        self._generate_visualizations(df)
        
        # Create HTML report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_html_report(df, report_time))
        
        print(f"\nReport generated: {report_file}")
        return report_file
    
    def _generate_visualizations(self, df):
        """Generate and save visualizations"""
        # 1. Sentiment Distribution
        plt.figure(figsize=(10, 6))
        sentiment_counts = df['sentiment'].value_counts()
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'])
        plt.title('Sentiment Distribution')
        plt.savefig(f'{self.report_dir}/sentiment_distribution.png')
        plt.close()
        
        # 2. Post Length Distribution
        plt.figure(figsize=(10, 6))
        df['text_length'] = df['text'].str.len()
        sns.histplot(data=df, x='text_length', bins=20)
        plt.title('Post Length Distribution')
        plt.savefig(f'{self.report_dir}/post_length_distribution.png')
        plt.close()
        
        # 3. Source Distribution
        plt.figure(figsize=(10, 6))
        source_counts = df['source'].value_counts()
        source_counts.plot(kind='bar')
        plt.title('Posts by Source')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/source_distribution.png')
        plt.close()
        
        # 4. Confidence Analysis
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='sentiment', y='confidence')
        plt.title('Confidence by Sentiment')
        plt.savefig(f'{self.report_dir}/confidence_analysis.png')
        plt.close()
    
    def _generate_html_report(self, df, report_time):
        """Generate HTML report content"""
        # Calculate statistics
        total_posts = len(df)
        sentiment_dist = df['sentiment'].value_counts()
        source_dist = df['source'].value_counts()
        avg_confidence = df['confidence'].mean()
        avg_length = df['text'].str.len().mean()
        
        html_content = f"""
        <html>
        <head>
            <title>Mood Detection Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 20px 0; padding: 20px; background-color: #f5f5f5; border-radius: 5px; }}
                .visualization {{ margin: 20px 0; text-align: center; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Mood Detection Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <p>Total Posts Analyzed: {total_posts}</p>
                <p>Average Post Length: {avg_length:.1f} characters</p>
                <p>Average Confidence: {avg_confidence:.2f}</p>
            </div>
            
            <div class="section">
                <h2>Sentiment Analysis</h2>
                <div class="visualization">
                    <img src="sentiment_distribution.png" alt="Sentiment Distribution">
                </div>
                <table>
                    <tr>
                        <th>Sentiment</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                    {self._generate_sentiment_rows(sentiment_dist, total_posts)}
                </table>
            </div>
            
            <div class="section">
                <h2>Source Analysis</h2>
                <div class="visualization">
                    <img src="source_distribution.png" alt="Source Distribution">
                </div>
                <table>
                    <tr>
                        <th>Source</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                    {self._generate_source_rows(source_dist, total_posts)}
                </table>
            </div>
            
            <div class="section">
                <h2>Post Length Analysis</h2>
                <div class="visualization">
                    <img src="post_length_distribution.png" alt="Post Length Distribution">
                </div>
            </div>
            
            <div class="section">
                <h2>Confidence Analysis</h2>
                <div class="visualization">
                    <img src="confidence_analysis.png" alt="Confidence Analysis">
                </div>
            </div>
            
            <div class="section">
                <h2>Sample Posts</h2>
                <table>
                    <tr>
                        <th>Text</th>
                        <th>Source</th>
                        <th>Sentiment</th>
                        <th>Confidence</th>
                    </tr>
                    {self._generate_post_rows(df.head())}
                </table>
            </div>
        </body>
        </html>
        """
        return html_content
    
    def _generate_sentiment_rows(self, sentiment_dist, total):
        """Generate HTML rows for sentiment distribution"""
        rows = ""
        for sentiment, count in sentiment_dist.items():
            percentage = (count / total) * 100
            rows += f"""
            <tr>
                <td class="{sentiment.lower()}">{sentiment}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
            """
        return rows
    
    def _generate_source_rows(self, source_dist, total):
        """Generate HTML rows for source distribution"""
        rows = ""
        for source, count in source_dist.items():
            percentage = (count / total) * 100
            rows += f"""
            <tr>
                <td>{source}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
            """
        return rows
    
    def _generate_post_rows(self, df):
        """Generate HTML rows for sample posts"""
        rows = ""
        for _, row in df.iterrows():
            rows += f"""
            <tr>
                <td>{row['text']}</td>
                <td>{row['source']}</td>
                <td class="{row['sentiment'].lower()}">{row['sentiment']}</td>
                <td>{row['confidence']:.2f}</td>
            </tr>
            """
        return rows

def main():
    print("Starting Report Generation...")
    
    # Create report generator
    generator = ReportGenerator()
    
    # Generate report
    report_file = generator.generate_analysis_report()
    
    print("\nReport Generation Complete!")
    print(f"1. Report saved to: {report_file}")
    print("2. Report includes:")
    print("   - Summary statistics")
    print("   - Sentiment distribution")
    print("   - Source analysis")
    print("   - Post length analysis")
    print("   - Confidence analysis")
    print("   - Sample posts")
    print("3. Open the HTML file in your web browser to view the report")

if __name__ == "__main__":
    main() 