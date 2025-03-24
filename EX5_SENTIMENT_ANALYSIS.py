import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
# Download the required resources
nltk.download('vader_lexicon')
# Create a sentiment analyzer
sia = SentimentIntensityAnalyzer()
# Function to determine sentiment
def get_sentiment(text):
 sentiment_scores = sia.polarity_scores(text) 
 if sentiment_scores['compound'] >= 0.05:
  sentiment = 'positive'
 elif sentiment_scores['compound'] <= -0.05:
  sentiment = 'negative'
 else:
  sentiment = 'neutral'
 return sentiment
# Collect user input
user_input = input("Enter a sentence: ")
# Analyze sentiment
sentiment = get_sentiment(user_input)
print(f"Sentiment: {sentiment}")
