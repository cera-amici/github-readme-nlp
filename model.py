import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from prepare import prep

def n_words(string):
    return len(string.split())

def n_unique_words(string):
    return len(set(string.split()))

def prep_for_model():
    df = prep()
    df = df.rename(columns={'lemmatized': 'text'})

    df['unique_words'] = df.text.apply(n_unique_words)
    df['word_count'] = df.text.apply(n_words)
    return df

df = prep_for_model()

def idf(word):
    n_occurrences = sum([1 for doc in df.text if word in doc])
    n_docs = len(df.text)
    idf = np.log(n_docs / n_occurrences)
    return idf

def get_words(df, language=None):
    if language is not None:
        df = df[df.language == language]
    
    raw_count = pd.Series(' '.join(df.text).split()).value_counts()

    words = pd.DataFrame({'raw_count': raw_count})
    words['frequency'] = words.raw_count / words.raw_count.sum()
    words['augmented_frequency'] = words.frequency / words.frequency.max()
    words['idf'] = words.index.to_series().apply(idf)
    return words

words = get_words(df)
python_words = get_words(df, 'Python')
js_words = get_words(df, 'JavaScript')
php_words = get_words(df, 'PHP')
shell_words = get_words(df, 'Shell')

tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df.text)
y = df.language

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=.2)

train = pd.DataFrame(dict(actual=y_train))
test = pd.DataFrame(dict(actual=y_test))

criteria = ['gini', 'entropy']

dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=15, min_samples_leaf=5)
dt.fit(x_train, y_train)

train['predicted'] = dt.predict(x_train)
test['predicted'] = dt.predict(x_test)

print(classification_report(train.actual, train.predicted))

print(classification_report(test.actual, test.predicted))