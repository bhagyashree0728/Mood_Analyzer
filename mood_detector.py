import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, Toplevel
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from transformers import pipeline
from datetime import datetime
import os
from fpdf import FPDF
import seaborn as sns

class MoodDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mood Detector")
        self.root.geometry("800x600")
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Store posts
        self.posts = []
        
        self._create_widgets()
        
    def _create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Post input section
        ttk.Label(main_frame, text="Enter your social media post:").grid(row=0, column=0, sticky=tk.W)
        self.post_text = scrolledtext.ScrolledText(main_frame, width=60, height=5)
        self.post_text.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Source selection
        ttk.Label(main_frame, text="Select source:").grid(row=2, column=0, sticky=tk.W)
        self.source_var = tk.StringVar()
        sources = ['Twitter', 'Facebook', 'Instagram', 'LinkedIn', 'Other']
        self.source_combo = ttk.Combobox(main_frame, textvariable=self.source_var, values=sources)
        self.source_combo.grid(row=2, column=1, sticky=tk.W, pady=5)
        self.source_combo.set(sources[0])
        
        # Add post button
        ttk.Button(main_frame, text="Add Post", command=self._add_post).grid(row=3, column=0, pady=10)
        
        # Analyze button
        ttk.Button(main_frame, text="Analyze Posts", command=self._analyze_posts).grid(row=3, column=1, pady=10)
        
        # Posts list
        ttk.Label(main_frame, text="Added Posts:").grid(row=4, column=0, sticky=tk.W)
        self.posts_list = scrolledtext.ScrolledText(main_frame, width=60, height=10)
        self.posts_list.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=6, column=0, columnspan=2, sticky=tk.W)
        
    def _add_post(self):
        post = self.post_text.get("1.0", tk.END).strip()
        source = self.source_var.get()
        
        if not post:
            messagebox.showwarning("Warning", "Please enter a post!")
            return
            
        self.posts.append({
            'text': post,
            'source': source,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Update posts list
        self.posts_list.insert(tk.END, f"Source: {source}\nPost: {post}\n{'='*50}\n")
        self.post_text.delete("1.0", tk.END)
        self.status_var.set(f"Post added! Total posts: {len(self.posts)}")
        
    def _analyze_posts(self):
        if not self.posts:
            messagebox.showwarning("Warning", "Please add some posts first!")
            return
            
        try:
            self.status_var.set("Analyzing posts...")
            self.root.update()
            
            # Analyze sentiments
            df = pd.DataFrame(self.posts)
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
            
            # Generate PDF report
            self._generate_pdf_report(df)
            
            self.status_var.set("Analysis complete! Report generated.")
            messagebox.showinfo("Success", "Analysis complete! Report generated.\nCharts and report will be shown in a new window.")
            
            # Show charts and report in GUI
            self._show_charts_and_report(df)
            
        except Exception as e:
            self.status_var.set("Error during analysis!")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def _generate_pdf_report(self, df):
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Mood Detection Analysis Report', 0, 1, 'C')
        pdf.ln(10)
        
        # Summary Statistics
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Summary Statistics', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Total Posts Analyzed: {len(df)}', 0, 1)
        pdf.cell(0, 10, f'Average Post Length: {df["text"].str.len().mean():.1f} characters', 0, 1)
        pdf.cell(0, 10, f'Average Confidence: {df["confidence"].mean():.2f}', 0, 1)
        pdf.ln(10)
        
        # Sentiment Analysis
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Sentiment Analysis', 0, 1)
        pdf.set_font('Arial', '', 12)
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            pdf.cell(0, 10, f'{sentiment}: {count} posts ({percentage:.1f}%)', 0, 1)
        pdf.ln(10)
        
        # Source Analysis
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Source Analysis', 0, 1)
        pdf.set_font('Arial', '', 12)
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            percentage = (count / len(df)) * 100
            pdf.cell(0, 10, f'{source}: {count} posts ({percentage:.1f}%)', 0, 1)
        pdf.ln(10)
        
        # Sample Posts
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Sample Posts', 0, 1)
        pdf.set_font('Arial', '', 12)
        for _, row in df.head().iterrows():
            pdf.multi_cell(0, 10, f"Source: {row['source']}\nPost: {row['text']}\nSentiment: {row['sentiment']} (Confidence: {row['confidence']:.2f})\n")
            pdf.ln(5)
        
        # Save PDF
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f'mood_analysis_report_{report_time}.pdf'
        pdf.output(report_file)
        
        # Generate and save visualizations
        self._save_visualizations(df)
        
    def _save_visualizations(self, df):
        # Create reports directory if it doesn't exist
        if not os.path.exists('reports'):
            os.makedirs('reports')
            
        # 1. Sentiment Distribution
        plt.figure(figsize=(6, 4))
        sentiment_counts = df['sentiment'].value_counts()
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'])
        plt.title('Sentiment Distribution')
        plt.savefig('reports/sentiment_distribution.png')
        plt.close()
        
        # 2. Post Length Distribution
        plt.figure(figsize=(6, 4))
        df['text_length'] = df['text'].str.len()
        sns.histplot(data=df, x='text_length', bins=20)
        plt.title('Post Length Distribution')
        plt.savefig('reports/post_length_distribution.png')
        plt.close()
        
        # 3. Source Distribution
        plt.figure(figsize=(6, 4))
        source_counts = df['source'].value_counts()
        source_counts.plot(kind='bar')
        plt.title('Posts by Source')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/source_distribution.png')
        plt.close()
        
        # 4. Confidence Analysis
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x='sentiment', y='confidence')
        plt.title('Confidence by Sentiment')
        plt.savefig('reports/confidence_analysis.png')
        plt.close()

    def _show_charts_and_report(self, df):
        # Create a new window
        win = Toplevel(self.root)
        win.title("Analysis Charts and Report")
        win.geometry("1200x800")
        
        # Notebook for tabs
        notebook = ttk.Notebook(win)
        notebook.pack(fill='both', expand=True)
        
        # --- Charts Tab ---
        charts_frame = ttk.Frame(notebook)
        notebook.add(charts_frame, text="Charts")
        
        # Sentiment Distribution
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sentiment_counts = df['sentiment'].value_counts()
        ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        ax1.set_title('Sentiment Distribution')
        canvas1 = FigureCanvasTkAgg(fig1, master=charts_frame)
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)
        
        # Post Length Distribution
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        df['text_length'] = df['text'].str.len()
        sns.histplot(data=df, x='text_length', bins=20, ax=ax2)
        ax2.set_title('Post Length Distribution')
        canvas2 = FigureCanvasTkAgg(fig2, master=charts_frame)
        canvas2.draw()
        canvas2.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)
        
        # Source Distribution
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        source_counts = df['source'].value_counts()
        source_counts.plot(kind='bar', ax=ax3)
        ax3.set_title('Posts by Source')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        canvas3 = FigureCanvasTkAgg(fig3, master=charts_frame)
        canvas3.draw()
        canvas3.get_tk_widget().grid(row=1, column=0, padx=10, pady=10)
        
        # Confidence Analysis
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df, x='sentiment', y='confidence', ax=ax4)
        ax4.set_title('Confidence by Sentiment')
        canvas4 = FigureCanvasTkAgg(fig4, master=charts_frame)
        canvas4.draw()
        canvas4.get_tk_widget().grid(row=1, column=1, padx=10, pady=10)
        
        # --- Report Tab ---
        report_frame = ttk.Frame(notebook)
        notebook.add(report_frame, text="Report Summary")
        
        report_text = scrolledtext.ScrolledText(report_frame, width=100, height=40)
        report_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Fill report summary
        total_posts = len(df)
        avg_length = df['text'].str.len().mean()
        avg_confidence = df['confidence'].mean()
        sentiment_dist = df['sentiment'].value_counts()
        source_dist = df['source'].value_counts()
        
        report_text.insert(tk.END, f"Mood Detection Analysis Report\n")
        report_text.insert(tk.END, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report_text.insert(tk.END, f"Summary Statistics\n-------------------\n")
        report_text.insert(tk.END, f"Total Posts Analyzed: {total_posts}\n")
        report_text.insert(tk.END, f"Average Post Length: {avg_length:.1f} characters\n")
        report_text.insert(tk.END, f"Average Confidence: {avg_confidence:.2f}\n\n")
        report_text.insert(tk.END, f"Sentiment Analysis\n------------------\n")
        for sentiment, count in sentiment_dist.items():
            percentage = (count / total_posts) * 100
            report_text.insert(tk.END, f"{sentiment}: {count} posts ({percentage:.1f}%)\n")
        report_text.insert(tk.END, "\nSource Analysis\n---------------\n")
        for source, count in source_dist.items():
            percentage = (count / total_posts) * 100
            report_text.insert(tk.END, f"{source}: {count} posts ({percentage:.1f}%)\n")
        report_text.insert(tk.END, "\nSample Posts\n------------\n")
        for _, row in df.head().iterrows():
            report_text.insert(tk.END, f"Source: {row['source']}\nPost: {row['text']}\nSentiment: {row['sentiment']} (Confidence: {row['confidence']:.2f})\n{'-'*40}\n")
        report_text.configure(state='disabled')

def main():
    root = tk.Tk()
    app = MoodDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 