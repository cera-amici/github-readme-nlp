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
- Should you hope to update the READMEs on which the model is trained, you will also need to install bs4, the BeautifulSoup package, and configure an env.py file with two environment variables (github_token, github_username). Go here to generate a personal access token https://github.com/settings/tokens for the github API.
- To run the model, 

## Organization

`githug_readme_nlp.ipynb` pipeline:

_**Acquisition**_
- Acquire from included json file (data.json) if it exists.
- If not, scrape a list of the top 100 most-starred GitHub repositories for the following languages: Python, Shell, JavaScript, and PHP.
- Then, use the GitHub API to create a json file that includes each repository's name, raw README contents, and listed language.

_**Preparation**_
- Perform a basic clean of the README text.
- Create a stemmed version of the cleaned text.
- Create a lemmatized version of the cleaned text.

_**Exploration**_
- Vizualize distributions and correlations within the data

_**Modeling**_
- Split data
- Create multiple models with training data

_**Evaluation**_
- Analyize evaluation metrics with test data

## Dictionary

### Data Dictionary

**language:** log error equal to the log(zestimate) - log(home_value), values range from -4.65 to 5.26

**title:** created variable for number of years the property is old up to the year 2017

**original:** created variable of the value per square foot from the land_value divided by the lot_square_feet

**cleaned:** created variable of the value per square foot from the structure_value divided by the home_square_feet

**stemmed:** the number of bathrooms the unit contains, can include half baths as a .5 value

**lemmatized:** number of bedrooms assigned to the unit