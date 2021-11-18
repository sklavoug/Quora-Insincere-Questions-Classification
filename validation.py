# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:28:53 2021

@author: SKLAVOUG
"""

# validation.py
# Runs cross-validation on three models (MNB, logistic regression, SVM) with
# multiple hyperparameters.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Read and split the data
df = pd.read_csv('train.csv', 
                 low_memory=False,
                 index_col='qid')

x = df['question_text']
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.1, 
                                                    random_state=1,
                                                    shuffle=True,
                                                    stratify=y)

# nested_cross_val
# Function which actually performs nested cross-validation, with 5 folds on the
# outer CV (with cross_val_score) and 3 folds on the inner CV (with RandomizedSearchCV)
def nested_cross_val(clf, parameters):
    inner = KFold(n_splits=5, shuffle=True, random_state=1)
    outer = KFold(n_splits=3, shuffle=True, random_state=1)
    
    # Nested cross-validation
    # Randomised search for the outer CV, and cross_val_score for the inner CV
    clf = RandomizedSearchCV(test_clf_mnb, parameters_mnb, cv=inner, verbose=3, scoring='f1')
    clf.fit(x_train, y_train)
    
    nested_score = cross_val_score(clf, X=x_train, y=y_train, cv=outer, scoring='f1')
    print(nested_score)
    print(nested_score.mean())

### MNB ###
test_clf_mnb = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())])

parameters_mnb = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'vect__lowercase': (True, False),
    'clf__alpha': (0.0001, 0.001, 0.01, 0.1)
}

print('MNB')
nested_cross_val(test_clf_mnb, parameters_mnb)

### SVM ###
test_clf_svm = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LinearSVC())])

parameters_svm = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__lowercase': (True, False),
    'clf__C': (0.0001, 0.001, 0.01, 0.1),
    'clf__class_weight': ('balanced', None),
    'clf__max_iter': (10, 50, 80),
}

print('SVM')
nested_cross_val(test_clf_svm, parameters_svm)

### Logistic Regression ###
test_clf_log = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression())])

parameters_log = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__lowercase': (True, False),
    'clf__C': (0.0001, 0.001, 0.01, 0.1),
    'clf__class_weight': ('balanced', None),
    'clf__penalty': ('l1','l2'),
}

print('LOG')
nested_cross_val(test_clf_log, parameters_log)


