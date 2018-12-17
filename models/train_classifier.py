import sys
import pandas as pd
import re
import pickle

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def load_data(database_filepath):
    
    '''
    Loads data from database table
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(database_filepath, engine)
    
    X = df['message']
    Y = df.iloc[:,4:]
    
#     genre_table = pd.get_dummies(df['genre'])
#     genre_table = genre_table.astype(int)
#     Y = pd.concat([Y, genre_table], axis=1)
#     Y = Y.drop('genre', axis=1)
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    
    '''
    Tokenizes text input
    INPUT
    text: messages
    OUPUT:
    words: normalized, lemmatized and tokenized text data
    '''
    #Normalizes text data
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #Tokenizes text data into words
    words = word_tokenize(text)
    
    #Removed stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    #Word lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    tokenized_data = lemmed
    
    return tokenized_data

def build_model():
    
    '''
    Builds nlp pipeline MultiOutputClassifier model and GridSearch parameters
    
    OUTPUT:
    GridSearch object
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'clf__estimator__n_estimators':[8, 10, 12]}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Evaluates model using sklearn classificaton report
    INPUT
    model: Gridsearch object
    X_test: Split test data
    Y_test: Split target data
    category_names: Column list of categories
    
    OUPUT
    Classificaiton report containing precision, recall, and f1 score for each classification
    '''
    
    y_pred = model.predict(X_test)
 
    for column in category_names:
        pred_index = 0
        column_prediction = [x[pred_index] for x in y_pred]
        print(classification_report(Y_test[column], column_prediction, target_names=[column]))

        pred_index = pred_index + 1

def save_model(model, model_filepath):
    '''
    Saves model as pickle file to desired filepath locaiton
    
    INPUT
    model: MultiOutputClassificaton model
    model_filepath: desired filepath for pickle file
    '''
    joblib.dump(model, model_filepath) 

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()