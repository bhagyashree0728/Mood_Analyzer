import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download required NLTK data
print("Downloading required NLTK data...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Create sample data
print("\nCreating sample dataset...")
sample_data = {
    'text': [
        'I am so happy today! Everything is going great!',
        'Feeling sad and lonely, wish I had someone to talk to.',
        'This movie was amazing, I loved every moment of it!',
        'I am really angry about what happened yesterday.',
        'Feeling anxious about the upcoming exam.',
        'The sunset is beautiful, feeling peaceful.',
        'I am excited about my new job!',
        'Feeling disappointed with the results.',
        'This food is delicious, I am so satisfied!',
        'Feeling scared about the future.'
    ],
    'mood': [
        'happy', 'sad', 'happy', 'angry', 'anxious',
        'peaceful', 'excited', 'disappointed', 'satisfied', 'scared'
    ]
}

# Create DataFrame
df = pd.DataFrame(sample_data)

# Apply text cleaning and preprocessing
print("Preprocessing text data...")
df['cleaned_text'] = df['text'].apply(clean_text)
df['processed_text'] = df['cleaned_text'].apply(preprocess_text)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['mood'], test_size=0.2, random_state=42
)

# Create TF-IDF features
print("Training the model...")
tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Print model performance
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def predict_mood(text):
    # Clean and preprocess the input text
    cleaned_text = clean_text(text)
    processed_text = preprocess_text(cleaned_text)
    
    # Transform the text using TF-IDF
    text_tfidf = tfidf.transform([processed_text])
    
    # Predict the mood
    predicted_mood = model.predict(text_tfidf)[0]
    
    return predicted_mood

# Test the model with some examples
print("\nTesting the model with example texts:")
test_texts = [
    "I am feeling wonderful today!",
    "This situation is making me very angry.",
    "I am worried about tomorrow's presentation."
]

for text in test_texts:
    mood = predict_mood(text)
    print(f"\nText: {text}")
    print(f"Predicted Mood: {mood}")

# Interactive mode
print("\nEnter your own text to predict mood (type 'quit' to exit):")
while True:
    user_input = input("\nEnter text: ")
    if user_input.lower() == 'quit':
        break
    mood = predict_mood(user_input)
    print(f"Predicted Mood: {mood}") 