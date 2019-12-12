import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import pickle
import os

from prepare import prep

df = prep(model=True)

def idf(word):
    n_occurrences = sum([1 for doc in df.lemmatized if word in doc])
    n_docs = len(df.lemmatized)
    idf = np.log(n_docs / n_occurrences)
    return idf

def get_words(df, fresh=False):
    if fresh or not os.path.exists('words.pickle'):
        raw_count = pd.Series(' '.join(df.lemmatized).split()).value_counts()

        words = pd.DataFrame({'raw_count': raw_count})
        words['frequency'] = words.raw_count / words.raw_count.sum()
        words['augmented_frequency'] = words.frequency / words.frequency.max()
        words['idf'] = words.index.to_series().apply(idf)

        with open('words.pickle', 'wb') as fp:
            pickle.dump(words, fp)  

    else:
        with open('words.pickle', 'rb') as fp:
            words = pickle.load(fp)

    return words

def store_model(model_object):
    with open('model.pickle', 'wb') as fp:
        pickle.dump(model_object, fp)

def get_model():
    with open('model.pickle', 'rb') as fp:
        model = pickle.load(fp)
    return model

def store_vectorizer(tfidf_object):
    with open('vectorizer.pickle', 'wb') as fp:
        pickle.dump(tfidf_object, fp)
    
def get_vectorizer():
    with open('vectorizer.pickle', 'rb') as fp:
        vectorizer = pickle.load(fp)
    return vectorizer

def prevent_overfitting(tree_function, df, store=False):
    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(df.lemmatized)
    y = df.language
    if store:
        store_vectorizer(tfidf)

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=.2, random_state=42)

    train = pd.DataFrame(dict(actual=y_train))
    test = pd.DataFrame(dict(actual=y_test))

    score_diff = 1
    leaf = 0
    
    while score_diff > .06:
        leaf += 1
        model = tree_function(criterion='entropy', min_samples_leaf=leaf, random_state=42)
        model.fit(x_train, y_train)
        train['predicted'] = model.predict(x_train)
        test['predicted'] = model.predict(x_test)
        train_acc = score(train.actual, train.predicted)[2].mean()
        test_acc = score(test.actual, test.predicted)[2].mean()
        score_diff = train_acc - test_acc

    print(f'leaf = {leaf}')
    print(f'train acc = {train_acc}')
    print(f'test acc = {test_acc}')
    if store:
        store_model(model)
    return model