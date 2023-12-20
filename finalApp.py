import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
nltk.download('stopwords')
import pickle
import base64
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from google_play_scraper import Sort, reviews
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from wordcloud import WordCloud  # Import WordCloud here

# Define df_reviews and df_preprocessed as global variables
df_reviews = pd.DataFrame()
df_preprocessed = pd.DataFrame()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to scrape reviews
def scrape_reviews(app_id, count=5000):
    try:
        result, _ = reviews(
            app_id,
            lang='id',
            country='id',
            sort=Sort.MOST_RELEVANT,
            count=count,
            filter_score_with=None
        )
        df = pd.DataFrame(result)
        st.success("Scraping reviews successful.")
        st.dataframe(df.head())
        return df
    except Exception as e:
        st.error(f"Error scraping reviews: {e}")
        return pd.DataFrame()

# Function for data preprocessing
def preprocess_data(df):
    try:
        st.text("Preprocessing data...")
        if df.empty:
            st.warning("No data to preprocess.")
            return df
        df['Label'] = df['score'].apply(lambda x: 'Positif' if x > 3 else 'Negatif')
        df['text_clean'] = df['content'].str.lower()
        df['text_clean'] = df['text_clean'].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?|\d+", "", elem))
        stop_words = stopwords.words('indonesian')
        df['text_StopWord'] = df['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
        df['text_tokens'] = df['text_StopWord'].apply(lambda x: word_tokenize(x))

        term_dict = {}
        hitung = 0
        for document in df['text_tokens']:
            for term in document:
                if term not in term_dict:
                    term_dict[term] = ' '

        for term in term_dict:
            term_dict[term] = term
            hitung += 1

        df['text_steamindo'] = df['text_tokens'].apply(lambda x: ' '.join([term_dict[term] for term in x]))
        st.success("Data preprocessing successful.")
        st.dataframe(df.head())
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return pd.DataFrame()

# Function for Naive Bayes model training with customizable parameters
def train_model(X_train, y_train, vectorizer_type='tfidf', classifier_type='naive_bayes'):
    try:
        st.text("Training model...")

        # Vectorization
        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer()
        else:
            # Add more vectorizer options if needed
            vectorizer = None

        X_train_tfidf = vectorizer.fit_transform(X_train)

        # Training model
        if classifier_type == 'naive_bayes':
            model = MultinomialNB()
        else:
            # Add more classifier options if needed
            model = None

        model.fit(X_train_tfidf, y_train)

        st.success("Model trained successfully.")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None

# Function for model evaluation
def evaluate_model(model, X_test, y_test, vectorizer):
    try:
        st.text("Evaluating the model...")
        if X_test.empty or y_test.empty or vectorizer is None or model is None:
            st.warning("No data for model evaluation.")
            return None, None, None, None
        # Vectorization of test data
        X_test_tfidf = vectorizer.transform(X_test)

        # Predictions
        y_pred = model.predict(X_test_tfidf)

        # Metrics calculation
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', pos_label='Negatif', zero_division=1)
        recall = recall_score(y_test, y_pred, average='binary', pos_label='Negatif', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='binary', pos_label='Negatif', zero_division=1)
        confusion_mat = confusion_matrix(y_test, y_pred)

        st.success("Model evaluation successful.")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        st.write(confusion_mat)

        # Classification Report
        st.subheader("Classification Report")
        from sklearn.metrics import classification_report
        class_report = classification_report(y_test, y_pred, target_names=['Positif', 'Negatif'], zero_division=1)
        st.text(class_report)

        return accuracy, precision, recall, f1
    except Exception as e:
        st.error(f"Error evaluating model: {e}")
        return None, None, None, None
    
# Function to save model and vectorizer
def save_model_and_vectorizer(model, vectorizer):
    with open('sentiment_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('sentiment_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    st.success("Model and Vectorizer saved successfully.")
    
# Function to save DataFrame to CSV
def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    st.success(f"{filename} saved successfully.")

def get_table_download_link(df, filename, button_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{button_text}</a>'
    return href

def download_model_link(obj, filename, button_text):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)
    b64 = base64.b64encode(open(filename, 'rb').read()).decode()
    href = f'<a href="data:file/pkl;base64,{b64}" download="{filename}">{button_text}</a>'
    return href

# Main Streamlit app
def main():
    global df_reviews, df_preprocessed, model, vectorizer  # Declare df_reviews, df_preprocessed, model, vectorizer as global variables
    st.title("Google Play Store Reviews Sentiment Analysis")

    # Input for app ID
    app_id = st.text_input("Enter Google Play Store App ID (e.g., id.flip):")
    count = st.number_input("Enter the number of reviews to scrape:", min_value=1, step=1, value=5000)

    # Options for customization
    vectorizer_option = st.selectbox("Select Vectorizer", ['tfidf'])  # Add more options if needed
    classifier_option = st.selectbox("Select Classifier", ['naive_bayes'])  # Add more options if needed

    if st.button("Analyze"):
        if app_id:
            st.text("Scraping reviews...")
            df_reviews = scrape_reviews(app_id, count)
            if not df_reviews.empty:
                df_preprocessed = preprocess_data(df_reviews)
                if not df_preprocessed.empty:
                    model, vectorizer = train_model(df_preprocessed['text_steamindo'], df_preprocessed['Label'],
                                                    vectorizer_type=vectorizer_option,
                                                    classifier_type=classifier_option)
                    if model and vectorizer:
                        evaluate_model(model, df_preprocessed['text_steamindo'], df_preprocessed['Label'], vectorizer)

                        # Download Data
                        st.subheader("Download Data:")
                        
                        raw_data_button = st.markdown(get_table_download_link(df_reviews, 'scraped_data.csv', 'Download Scraped Data'), unsafe_allow_html=True)
                        preprocessed_data_button = st.markdown(get_table_download_link(df_preprocessed, 'preprocessed_data.csv', 'Download Preprocessed Data'), unsafe_allow_html=True)

                        # Model and Vectorizer
                        model_button = st.markdown(download_model_link(model, 'sentiment_model.pkl', 'Download Model'), unsafe_allow_html=True)
                        vectorizer_button = st.markdown(download_model_link(vectorizer, 'sentiment_vectorizer.pkl', 'Download Vectorizer'), unsafe_allow_html=True)

                        # Visualizations
                        st.header("Data Visualization")
                        st.subheader("Distribution of Ratings")
                        if 'score' in df_reviews.columns:
                            sns.countplot(x='score', data=df_reviews)
                            st.pyplot()
                            
                            # Distribution of Positive and Negative Labels
                            st.subheader("Distribution of Positive and Negative Labels")
                            sns.countplot(x='Label', data=df_preprocessed)
                            st.pyplot()

                            # WordCloud for All Reviews
                            st.subheader("Word Cloud for All Reviews")
                            all_reviews_wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(' '.join(df_preprocessed['text_steamindo']))
                            plt.figure(figsize=(10, 5))
                            plt.imshow(all_reviews_wordcloud, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot()

                            # WordCloud for Positive Reviews
                            st.subheader("Word Cloud for Positive Reviews")
                            positive_reviews_wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(' '.join(df_preprocessed[df_preprocessed['Label'] == 'Positif']['text_steamindo']))
                            plt.figure(figsize=(10, 5))
                            plt.imshow(positive_reviews_wordcloud, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot()

                            # WordCloud for Negative Reviews
                            st.subheader("Word Cloud for Negative Reviews")
                            negative_reviews_wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(' '.join(df_preprocessed[df_preprocessed['Label'] == 'Negatif']['text_steamindo']))
                            plt.figure(figsize=(10, 5))
                            plt.imshow(negative_reviews_wordcloud, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot()
                        else:
                            st.warning("Word Cloud visualization requires preprocessed data.")
                    else:
                        st.warning("Model training failed.")
            else:
                st.warning("Please enter a valid Google Play Store App ID.")

if __name__ == "__main__":
    main()
