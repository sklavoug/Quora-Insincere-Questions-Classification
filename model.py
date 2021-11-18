# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:28:53 2021

@author: SKLAVOUG
"""

# model.py
# Code to create and run the final model (MNB) with the best hyperparameters
# for CountVectorizer and MNB model based on cross-validation.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score

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

# Create CountVectorizer and MultinomialNB model with best hyperparameters
# chosen by nested cross-validation.    
count = CountVectorizer(lowercase=False,
                        max_features=50000,
                        max_df=0.75)

x_train_bag = count.fit_transform(x_train)
x_test_bag = count.transform(x_test)

clf = MultinomialNB(alpha=0.1)
model = clf.fit(x_train_bag, y_train)
pred_y = model.predict(x_test_bag)

# Output F1 score as well as precision and recall
print(f'F1: {f1_score(y_test, pred_y)}')
print(f'Precision: {precision_score(y_test, pred_y)}')
print(f'Recall: {recall_score(y_test, pred_y)}')

# Store results as csv
final = pd.DataFrame(y_test)
final['predicted'] = pred_y
final = final.join(x_test, on='qid', how='left')
final.to_csv('results.csv')
print('### Results stored in results.csv ###')

# Extract features and convert log probabilities to use in more detailed analysis
features = pd.DataFrame(index=range(0,50000))
features['Features'] = count.get_feature_names()
features['Sincere'] = clf.feature_log_prob_[0]
features['Insincere'] = clf.feature_log_prob_[1]
features['Sincere'] = np.exp(1) ** features['Sincere']
features['Insincere'] = np.exp(1) ** features['Insincere']

features.to_csv('features.csv', index=False)
print('### Features stored in features.csv ###')