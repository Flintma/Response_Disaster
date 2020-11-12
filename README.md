# Disaster Response Pipelines


### Table of Contents

1. [Installation](#installation)
2. [Introduction](#introduction)
3. [Files Descriptions](#files)
4. [Instructions](#instructions)

## Installation <a name="installation"></a>

The necessary packages are:
- pandas
- warnings
- sys
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3
- re


## Introduction<a name = "introduction"></a>
The goal of the project is to categorize the message.
The raw data is processed on ETL first to clean and than saved in the database. Then the data is delivered to the modeling pipeline to build the model. After the model is built, the result is displayed on website using the Flask app.
 
 
 
## Files Descriptions <a name="files"></a>

The files structure is arranged as below:

	- README.md: read me file
	- ETL Pipeline Preparation.ipynb: contains ETL pipeline preparation code
	- ML Pipeline Preparation.ipynb: contains ML pipeline preparation code
	- workspace
		- \app
			- run.py: flask file to run the app
		- \templates
			- master.html: main page of the web application 
			- go.html: result web page
		- \data
			- disaster_categories.csv: categories dataset
			- disaster_messages.csv: messages dataset
			- DisasterResponse.db: disaster response database
			- process_data.py: ETL process
		- \models
			- train_classifier.py: classification code

## Instructions <a name="instructions"></a>

To execute the app follow the instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

