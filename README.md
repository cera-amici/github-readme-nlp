# README: logerror_zillow Clustering Project

***Goals Overview:*** Identify any features that are driving the difference in Zestimate and sales price which is creating a log error as well as to construct a model to predict the targeted log error.

## Table of Contents

- [Installation](#installation)
- [Organization](#organization)
- [Planning](#planning)
- [Dictionary](#dictionary)

## Installation

Instructions on setting up the Zillow project and necessary steps to successfully run on any laptop. 

- Create a new repository on GitHub
- Install VS Code
- Install Sequel Pro
- Install Anaconda
- Use Jupyter Notebook
- Reference get_function.py files with functions to run model

You must use Python version 3.7 or later and if you do not have PIP installed, you can download and install it from this page: [open](https://pypi.org/project/pip/).

You will be using common data science package libraries throughout.

## Organization

`logerror_zillow.ipynb` pipeline:

_**Acquisition**_
- Acquire from SQL the zillow database

_**Preperation**_
- Handle Nulls, outliers, drop variables, avoid overcomplications

_**Exploration**_
- Vizualize distributions, clustering for early discoveries

_**Modeling**_
- Create multiple models with fit/predict with train data

_**Evaluation**_
- Analyize evaluation metrics and run test data

**Important:** 
To reporduce this project please follow along with the logerror_zillow clusterting project notebook [open](https://github.com/P-F-M/logerror_zillow/blob/master/logerror_zillow.ipynb) and set: n_init = 1 and Random State equal to 123.

Replace the following content within your project's env.py file: (host, user, password) 

## Planning

### Goals:

1. Predict the log error
2. Discover highlights from findings, vizualizations, and real estate domain knowledge
3. Capture lessons learned within data science, Python, SQL, etc.

### Zilow data:

* Include properties with a transaction in 2017.
* Include only the last transaction for each properity.
* Include Zestimate error and date of transaction.
* Include properties that include a latitude and longitude value.
* Remove all properties that are not single unit properties (Reference: SQL query in get_function.py).

## Dictionary

### Data Dictionary

**target:** log error equal to the log(zestimate) - log(home_value), values range from -4.65 to 5.26

**age:** created variable for number of years the property is old up to the year 2017

**land_value_square_footage:** created variable of the value per square foot from the land_value divided by the lot_square_feet

**home_value_square_footage:** created variable of the value per square foot from the structure_value divided by the home_square_feet

**bathroom:** the number of bathrooms the unit contains, can include half baths as a .5 value

**bedroom:** number of bedrooms assigned to the unit

**home_square_feet:** amount of square footage size assigned to the unit

**lot_square_feet:** amount of square footage size assigned to the land

**structure_value:** dollar value of unit only

**total_value:** total dollar value of unit and land added together

**land_value:** dollar value of land only

**tax_amount:** value of taxes assessed to the property from previous year

**cluster:** created variable from the K-Means clustering method, varies by number of K

**avg_home_value_square_footage:** created variable from the cluster groups where the clusters were grouped and the mean was assigned to the variable

**avg_land_value_square_footage:** created variable from the cluster groups where the clusters were grouped and the mean of each cluster was assigned to the variable

**`* Note:`** all variables are assigned a data type of: Float64 except for the int32 "cluster" variable