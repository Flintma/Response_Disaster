import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion
import warnings
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
warnings.simplefilter('ignore')

def load_data(database_filepath):
    """
    Load and merge datasets
    input:
         database name
    outputs:
        X: messages 
        y: response variable
        category names. Y column value
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.iloc[:,5:]
    category_names=Y.columns.values
    return X,Y,category_names

def tokenize(text):
    """ Normalize and tokenize
    """
    # Converting everything to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # normalization word tokens and remove stop words
    normlizer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    normlized = [normlizer.stem(word) for word in tokens if word not in stop_words]
    
    return normlized


def build_model(X_train,Y_train):
    """
    pipe line construction
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring='f1_micro', n_jobs=-1, verbose = 10)
    cv.fit(X_train, Y_train)
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    inputs
        model
        X_test
        y_test
        category_names
    output:
        scores
    """
    y_prediction = model.predict(X_test)
    print(classification_report(Y_test.values,y_prediction, target_names=category_names))

def save_model(model, model_filepath):
    """
    Save model to a pickle file
    """
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()


def main():
    '''Main program to run the modeling process
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train,Y_train)
        
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