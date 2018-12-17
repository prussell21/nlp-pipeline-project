# NLP Pipeline For Disaster Relief Messages

![Web Page Banner](https://raw.githubusercontent.com/prussell21/nlp-pipeline-project/master/docs/web-page-banner.png)

An NLP pipeline and model that allows a user to upload new messages to clean, tokenize and train a multi-output-classifier for determining whether message from multiple sources are appropriate for disaster relief.

### Usage

1. Run the following commands in the project's root directory to set up your database and model.

   ETL
    - To run ETL pipeline that cleans data and stores in database.
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - This generates DisasterResponse.db with merged and cleaned data from original message and category csv files
    
   Machine Learning
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - Trains MultiOutputClassifer and saves new model to models/classifier.pkl

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

### Output

Using a Multi-Output-Classifier, relevant categories are highlighted


![Classification Image](https://raw.githubusercontent.com/prussell21/nlp-pipeline-project/master/docs/message-classificaton.png)

### Scripts and Data

process_data.py: merges, cleans, and tokenizes message and category data for machine learning

train_classifer.py: trains MultiOutputClassifier using RandomForestClassifier as it's estimator

disaster_messages.csv: Message data provided by FigureEight

disaster_categories.csv: Labeled categories for corresponding Message data provided by FigureEight

https://www.figure-eight.com/datasets/


