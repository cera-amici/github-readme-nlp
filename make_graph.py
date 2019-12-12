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
import matplotlib.pyplot as plt

from prepare import prep
import model

df = prep(model=True)

tfidf = TfidfVectorizer()

x = tfidf.fit_transform(df.lemmatized)

y = df.language

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=.2, random_state=42)

test = pd.DataFrame(dict(actual=y_test)) 

dt = model.get_model()

test['predicted'] = dt.predict(x_test)
test['correct'] = test.predicted == test.actual

test = test.drop(columns='predicted')

test_graph = pd.DataFrame()

test_actuals = test.groupby('actual').correct.sum()

test_graph['correct'] = test_actuals

test_graph['incorrect'] = 20 - test_graph.correct

plt.title('Accuracy of Predictions on Test Data')
plt.ylabel('Count')
plt.xlabel('Programming Language')
plt.xticks(rotation=45)
test_graph.plot.bar(stacked=True)
plt.show()