import pandas as pd
import joblib

# Load model and vectorizer
model = joblib.load("model/xgb_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

def clean_text(text):
    import re, string
    from nltk.corpus import stopwords
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

def preprocess(df):
    df['clean_title'] = df['title'].astype(str).apply(clean_text)
    df['clean_description'] = df['description'].astype(str).apply(clean_text)
    df['text'] = df['clean_title'] + ' ' + df['clean_description']
    X_text = vectorizer.transform(df['text'])
    X = pd.DataFrame(X_text.toarray())
    df['employment_type'] = df['employment_type'].astype(str)
    X['employment_type'] = df['employment_type'].astype('category').cat.codes
    return X

def predict_from_csv(uploaded_df):
    X = preprocess(uploaded_df)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]
    uploaded_df['fraud_probability'] = probs
    uploaded_df['predicted_label'] = preds
    return uploaded_df
