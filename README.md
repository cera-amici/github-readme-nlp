# GitHub README Language Identification with Natural Language Processing

***Goal:*** Build a model to predict the programming language of a GitHub repository based on its README file.

## Table of Contents

- [Installation](#installation)
- [Organization](#organization)
- [Dictionary](#dictionary)

## Installation

Instructions on setting up the GitHub README project and necessary steps to successfully run on any computer. 

- Make a copy of this repository.
- Ensure you've installed the following: Python 3.7, pandas, NLTK, scikit-learn, wordcloud, and numpy.
- Also download wordnet and stopwords from the NLTK package.
- Should you hope to update the READMEs on which the model is trained, you will also need to install bs4, the BeautifulSoup package, and configure an env.py file with two environment variables (github_token, github_username). Click [here](https://github.com/settings/tokens) to generate a personal access token for the github API.
- To run the model, 

## Organization

`githug_readme_nlp.ipynb` pipeline:

**Acquisition**
- Acquire from included json file (data.json) if it exists.
- If not, scrape a list of the top 100 most-starred GitHub repositories for the following languages: Python, Shell, JavaScript, and PHP.
- Then, use the GitHub API to create a json file that includes each repository's name, raw README contents, and listed language.

**Preparation**
- Perform a basic clean of the README text.
- Create a stemmed version of the cleaned text.
- Create a lemmatized version of the cleaned text.

**Exploration**
- Vizualize distributions within the data

**Modeling**
- Split data
- Create multiple models with training data

**Evaluation**
- Analyize evaluation metrics with test data

## Dictionary

### Uncommon Words and Phrases

**natural language processing:** using programming and machine learning techniques to understand large amounts of text

**stem:** reducing words to their root (or stem). simply removes ends of words and preserves the beginning that matches other words (e.g. stemming -> stem, stems -> stem, stemmata -> stem)

**lemmatize:** reduces words to their grammatical base. Notably different from stemming with irregular words (e.g. better -> good). 

**corpus:** full set of documents or other text materials

### Data Dictionary

**language:** programming language of the repository

**title:** title of the repository

**original:** raw text of the README

**cleaned:** original with the following transformations applied: removed non-UTF-8 characters, lowercased, removed non-alphabetic characters, and replaced all whitespace characters with a space

**stemmed:** the stemmed version of cleaned using the PorterStemmer from NLTK

**lemmatized:** the lemmatized version of cleaned using the WordNetLemmatizer from NLTK