import csv
import re
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

# preprocessing function
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'https?://\S+', '', text)  
    text = re.sub(r'<.*?>', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    text = re.sub(r'\d+', '', text)  
    text = text.strip()
    return text

# loading dataset
def load_data(file_path):
    reviews, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            if row[0] and row[1]:
                reviews.append(clean_text(row[0]))
                labels.append(row[1])
    return reviews, labels

def train_and_evaluate(X, y, model_choice='svm'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC() if model_choice == 'svm' else LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n--- Evaluation Report ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return pipeline

# predict a single user input
def predict_input(model):
    user_input = input("\nEnter a movie review: ")
    cleaned = clean_text(user_input)
    prediction = model.predict([cleaned])
    print("\nPredicted Sentiment:", prediction[0])

def main():
    file_path = 'movie_reviews.csv'
    X, y = load_data(file_path)

    print("\nTraining with Support Vector Machine...")
    model_svm = train_and_evaluate(X, y, model_choice='svm')

    print("\nTraining with Logistic Regression...")
    model_lr = train_and_evaluate(X, y, model_choice='lr')

    predict_input(model_svm)

if __name__ == "__main__":
    main()
