##import the libraries
import sys
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import numpy as np
import re
import sklearn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
import pickle
import sqlite3
nltk.download(['punkt','wordnet','stopwords'])

def load_data(database_filepath):
    """loads the data and returns X and y variables for the analysis"""
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('Select * from "Message_Categories"', conn)
    df =df[df['related']!=2]
    df =df.drop('child_alone', axis=1)
    X = df.message.values
    y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns
    
    return X, y, category_names


def tokenize(text):
    """Cleans the text for the analysis by normalizing, tokenizing and lemmatizing the text"""
    
    for word in text:
        word = word.lower(
        word = re.sub(r"don't", 'do not', word)
        word = re.sub(r"aren't", 'are not', word)
        word = re.sub(r"doesn't", 'does not', word)
        word = re.sub(r"'re", ' are', word)
        word = re.sub(r"i'm", 'i am', word)
        word = re.sub(r"i'd", 'i would', word)
        word = re.sub(r"i've", 'i have', word)
        word = re.sub(r"we'd", 'we would', word)
        word = re.sub(r"it's", 'it is', word)
        word = re.sub(r"what's", 'what is', word)
        word = re.sub(r"who's", 'who is', word)
        word = re.sub(r"where's", 'where is', word)
        word = re.sub(r"we'll", 'we will', word)
        word = re.sub(r"he'll", 'he will', word)
        word = re.sub(r"can't", 'can not', word)
        word = re.sub(r"there's", 'there is', word)
        word = re.sub(r"they've", 'they have', word)
        word = re.sub(r"that's", 'that is', word)
        word = re.sub(r"didn't", 'did not', word)
        word = re.sub(r"i'll", 'i will', word)
        word = re.sub(r"haven't", 'have not', word)
        word = re.sub(r"let's", 'let us', word)
        word = re.sub(r"wasn't", 'was not', word)
        word = re.sub(r"how's", 'how is', word)
        word = re.sub(r"isn't", 'is not', word)
        word = re.sub(r"you'll", 'you will', word)
        word = re.sub(r"won't", 'will not', word)
        word = re.sub(r'[^a-zA-Z0-9]', ' ', word)
                  
        tokens = word_tokenize(word)
        lemmatizer = WordNetLemmatizer()
        clean_token = []
        for word in tokens:
            tok = lemmatizer.lemmatize(word).strip()
            clean_token.append(tok)
        clean_token = [word for word in clean_token if word not in stopwords.words("english")]
        return clean_token


def build_model():
    """Builds a model to be applied on the clean data.
    A pipeline is built for the model and through GridSearch, returns the model with best parameters"""
            
    estimator=RandomForestClassifier()
    pipeline = Pipeline([
        ('transformer', Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer())
            ])),
         ('clf', MultiOutputClassifier(estimator))
        ])
    
    parameters = { 
    'transformer__vect__max_features': [5000, 3000, 2000],
    'transformer__vect__ngram_range': ((1,1),(1,2)),
    'transformer__tfidf__use_idf': (True, False)
            }

    model = GridSearchCV(pipeline, param_grid = parameters)

    return model

def evaluate_model(model, X_test, y_test, category_names):
    """Evaluates the model with the f1 score, recall score, accuracy and precision"""
    y_pred = model.predict_proba(X_test)
    y_pred = [(a>.5).astype(int) for a in y_pred]
    score_df = {}
    precision_list = []
    recall_list = []
    f1score_list = []
    accuracy_list = []

    #labels = df1.iloc[:,4:].columns
    for i in range(y_test.shape[1]):
        class_report = classification_report(y_test[:,i], y_pred[i][:,1], output_dict=True)
        acc_score =  accuracy_score(y_test[:,i], y_pred[i][:,1])
        precision_list.append(class_report['weighted avg']['precision'])
        recall_list.append(class_report['weighted avg']['recall'])
        f1score_list.append(class_report['weighted avg']['f1-score'])
        accuracy_list.append(acc_score)
    score_df = pd.DataFrame({'Variable':category_names, 'Precision':precision_list,'Recall':recall_list,'F-1_score':f1score_list, 'Accuracy':accuracy_list})
    score_df.set_index('Variable', inplace = True)
    return score_df


def save_model(model, model_filepath):
            """saves the model in a pickle file"""
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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