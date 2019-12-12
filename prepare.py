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

def drop_outliers(df):
    df = df.drop(df[df.length > 140000].index)

    languages = df.language.unique()
    for language in languages:
        n_drop = len(df[df.language == language]) - 100
        to_drop = df[df.language == language].word_count.sort_values().head(n_drop)
        df = df.drop(to_drop.index)
    return df
    
def n_words(string):
    return len(string.split())

def n_unique_words(string):
    return len(set(string.split()))

def prepare_article_data(df, model, content='readme_contents'):
    df['original'] = df[content]
    df['cleaned'] = df.original.apply(basic_clean).apply(remove_stopwords)
    df['stemmed'] = df.cleaned.apply(stem)
    df['lemmatized'] = df.cleaned.apply(lemmatize)
    df['length'] = df.cleaned.apply(len)
    df.drop(columns=content, inplace=True)
    if model:
        df['word_count'] = df.lemmatized.apply(n_words)
        df['unique_words'] = df.cleaned.apply(n_unique_words)
        df = drop_outliers(df)
    return df

def prep(json='prepped_data.json', fresh=False, model=False):
    if os.path.exists(json) and not fresh and not model:
        df = pd.read_json(json)
    else:
        df = pd.read_json('data.json')
        df = prepare_article_data(df, model=model)
        df = df.rename(columns={'repo': 'title'})
        if not model:
            df.to_json(json)
    return df