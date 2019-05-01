# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords 
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
import pickle

def load_data(database_filepath):
    """
    Input:
        database_filepath -> path to SQLite db
    Output:
        X -> features DataFrame
        Y -> label DataFrame
        cat -> cagtegories used for the app
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df',engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    cat = list(df.columns[4:])
    return X,Y, cat


def tokenize(text):
    """
     Input:
        text -> text messages
     Output:
        clean_tokens -> clean tokenized text
    """
    lemmatizer =WordNetLemmatizer()
    
    #Cache the stop words for speed 
    cachedStopWords = stopwords.words("english")
    
    text = re.sub(r"[^a-zA-Z0-9]", " ",text.lower())
    tokens = word_tokenize(text)
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
    This function creates a ML Pipeline that process text messages
    according to NLP and apply Random Forest classifier from Scikit Learn
    
    Output:
        ML Model
    """
    model = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = { 
    'features__text_pipeline__vect__ngram_range': ((1,1),(1,2)),
    'features__text_pipeline__vect__max_features': (None,5000),
    'features__text_pipeline__tfidf__use_idf': (True, False),          
    'clf__estimator__n_estimators': [10],
    'clf__estimator__min_samples_split': [2] 
    }
    cv = GridSearchCV(model, param_grid= parameters, verbose=2)
    model = cv
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Input:
        model ->  ML Pipeline
        X_test -> test features
        Y_test -> test labels
        cat -> categories 
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    


def save_model(model, model_filepath):
    """
    Input:
        model -> Scikit Pipelin object
        model_filepath -> destination path 
    
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


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