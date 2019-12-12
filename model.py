import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import pickle

from prepare import prep

df = prep(model=True)

def idf(word):
    n_occurrences = sum([1 for doc in df.lemmatized if word in doc])
    n_docs = len(df.lemmatized)
    idf = np.log(n_docs / n_occurrences)
    return idf

def get_words(df, language=None):
    if language is not None:
        df = df[df.language == language]
    
    raw_count = pd.Series(' '.join(df.lemmatized).split()).value_counts()

    words = pd.DataFrame({'raw_count': raw_count})
    words['frequency'] = words.raw_count / words.raw_count.sum()
    words['augmented_frequency'] = words.frequency / words.frequency.max()
    words['idf'] = words.index.to_series().apply(idf)
    return words

def store_model(model_object):
    with open('model.pickle', 'wb') as fp:
        pickle.dump(model_object, fp)

def get_model():
    with open ('model.pickle', 'rb') as fp:
        model = pickle.load(fp)
    return model

if __name__ == '__main__':
    words = get_words(df)

    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(df.lemmatized)
    y = df.language

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=.2)

    train = pd.DataFrame(dict(actual=y_train))
    test = pd.DataFrame(dict(actual=y_test))

    score_diff = 1
    leaf = 0

    while score_diff > .06:
        leaf += 1
        dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=leaf)
        dt.fit(x_train, y_train)
        train['predicted'] = dt.predict(x_train)
        test['predicted'] = dt.predict(x_test)
        train_acc = score(train.actual, train.predicted)[2].mean()
        test_acc = score(test.actual, test.predicted)[2].mean()
        score_diff = train_acc - test_acc

    print(f'leaf = {leaf}')
    print(f'train acc = {train_acc}')
    print(f'test acc = {test_acc}')

    store_model(dt)

# score_diff = 1
# leaf = 1

# while score_diff > .06:
#     rf = RandomForestClassifier(criterion='entropy', min_samples_leaf=leaf, random_state=123)
#     rf.fit(x_train, y_train)
#     train['predicted'] = rf.predict(x_train)
#     test['predicted'] = rf.predict(x_test)
#     train_acc = score(train.actual, train.predicted)[2].mean()
#     test_acc = score(test.actual, test.predicted)[2].mean()
#     score_diff = train_acc - test_acc
#     leaf += 1

# rf = RandomForestClassifier(criterion='entropy', min_samples_leaf=19, random_state=123)
# rf.fit(x_train, y_train)
# train['predicted'] = rf.predict(x_train)
# test['predicted'] = rf.predict(x_test)
# train_acc = score(train.actual, train.predicted)[2].mean()
# test_acc = score(test.actual, test.predicted)[2].mean()
# score_diff = train_acc - test_acc

# print(f'leaf = {leaf}')
# print(f'train acc = {train_acc}')
# print(f'test acc = {test_acc}')

# scaled_df = m
