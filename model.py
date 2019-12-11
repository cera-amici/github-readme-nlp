import pandas as pd
import numpy as np

from prepare import prep

def prep_for_model():
    df = prep()[['lemmatized', 'language']]
    df = df.rename(columns={'lemmatized': 'text'})

    def n_words(string):
        return len(string.split())

    def n_unique_words(string):
        return len(set(string.split()))

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