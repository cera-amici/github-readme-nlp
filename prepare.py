import pandas as pd

import unicodedata
import re
import json
import os

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

CSV = 'github_readmes.csv'

ADDITIONAL_STOPWORDS = ['http', 'https', 'com', 'github', 'git', 'org', 'www', 'code', 'file']

def normalize(string):
    return unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def basic_clean(string):
    """
    Lowercase the string
    Normalize unicode characters
    Replace anything that is not a letter, number, whitespace or a single quote.
    """
    string = str(string).lower()
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # remove anything not a space character, an apostrophe, letter, or number
    string = re.sub(r"[^a-z\s]", ' ', string)

    # drop weird words <=2 characters
    # string = re.sub(r'\b[a-z]{,2}\b', '', string)

    # convert newlines and tabs to a single space
    string = re.sub(r'[\r|\n|\r\n|\t]+', ' ', string)
    string = string.strip()
    return string

def tokenize(string):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(string, return_str=True)

def stem(text):
    stemmer = nltk.porter.PorterStemmer()
    words = text.split()
    stems = [stemmer.stem(word) for word in words]
    stem_string = ' '.join(stems)
    return stem_string

def lemmatize(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in text.split()]
    lemma_string = ' '.join(lemmas)
    return lemma_string

def remove_stopwords(text, extra_words=ADDITIONAL_STOPWORDS, exclude_words=[]):
    text = tokenize(text)

    text = text.split()
    stopword_list = stopwords.words('english')

    stopword_list = set(stopword_list).difference(set(exclude_words))
    stopword_list = stopword_list.union(set(extra_words))

    filtered = []
    for word in text:
        if word not in stopword_list:
            filtered.append(word)
    text = " ".join(filtered)
    return text

def prepare_article_data(df, content='readme_contents'):
    df['original'] = df[content]
    df['cleaned'] = df.original.apply(basic_clean).apply(remove_stopwords)
    df['stemmed'] = df.cleaned.apply(stem)
    df['lemmatized'] = df.cleaned.apply(lemmatize)
    df.drop(columns=content, inplace=True)
    return df

def get_prepped(csv=CSV, fresh=False):
    if os.path.exists('prepped_' + csv) and not fresh:
        df = pd.read_csv('prepped_' + csv, index_col = 0)
    else:
        df = pd.read_csv(csv, index_col = 0)
        df = prepare_article_data(df)
        df.to_csv('prepped_' + csv)
    return df

def prep(json='prepped_data.json', fresh=False):
    if os.path.exists(json) and not fresh:
        df = pd.read_json(json)
    else:
        df = pd.read_json('data.json')
        df = prepare_article_data(df)
        df = df.rename(columns={'repo': 'title'})
        df.to_json(json)
    return df