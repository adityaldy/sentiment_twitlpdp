import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import streamlit as st

# Load the dataset
file_path = 'tokenized.csv'
data = pd.read_csv(file_path)

# Simulate sentiment analysis
def random_sentiment(text):
    return np.random.choice(['positive', 'neutral', 'negative'])

# Apply the simulated sentiment analysis to the dataset
data['sentiment_textblob'] = data['tokenized'].apply(random_sentiment)

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['tokenized'])
y = data['sentiment_textblob']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict the sentiments on the test set
y_pred = nb_classifier.predict(X_test)

# Display the classification report
classification_report_nb = classification_report(y_test, y_pred, output_dict=True)
classification_report_nb_df = pd.DataFrame(classification_report_nb).transpose()

# Function to generate word cloud
def generate_wordcloud(data, title):
    text = ' '.join(data)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=15)
    plt.show()

# Generate word clouds for each sentiment
for sentiment in ['positive', 'neutral', 'negative']:
    generate_wordcloud(data[data['sentiment_textblob'] == sentiment]['tokenized'], f'WordCloud for {sentiment} sentiment')

# Create bar chart for sentiment distribution
plt.figure(figsize=(10, 5))
sns.countplot(x='sentiment_textblob', data=data, order=['positive', 'neutral', 'negative'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Streamlit app
def app():
    st.title('Sentiment Analysis of Indonesian Tweets')
    
    st.header('WordClouds')
    for sentiment in ['positive', 'neutral', 'negative']:
        st.subheader(f'WordCloud for {sentiment} sentiment')
        text = ' '.join(data[data['sentiment_textblob'] == sentiment]['tokenized'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.image(wordcloud.to_array())
    
    st.header('Sentiment Distribution')
    sns.countplot(x='sentiment_textblob', data=data, order=['positive', 'neutral', 'negative'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot(plt)

    st.header('Naive Bayes Classification Report')
    st.table(classification_report_nb_df)

if __name__ == '__main__':
    app()
