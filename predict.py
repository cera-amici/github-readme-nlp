# TAKES INPUT AND RETURNS LANGUAGE
from model import get_model, get_vectorizer
from prepare import basic_clean, remove_stopwords, lemmatize
from sklearn.feature_extraction.text import TfidfVectorizer


def predict_language(text):
    text = lemmatize(remove_stopwords(basic_clean(text)))
    tfidf = get_vectorizer()
    x = tfidf.transform([text])
    model = get_model()
    language = model.predict(x)
    return language[0]