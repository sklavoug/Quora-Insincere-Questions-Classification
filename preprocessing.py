# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:28:53 2021

@author: SKLAVOUG
"""

# preprocessing.py
# Results of attempts at preprocessing using both sklearn's CountVectorizer
# and manual implementation including stemming. Note that none of the options
# contained within were actually used in the final model, as the best
# F1 score came from no preprocessing but using some of the CountVectorizer
# attributes such as max_df and max_features.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from nltk.stem import PorterStemmer


# Cleaning function, sets text to lower-case, removes any non-alphabetical
# characters (including punctuation) and stems all the resulting words.
# First two steps are achieved by CountVectorizer but stemming required
# each component to be done manually as there didn't seem to be a way to 
# add stemming to CountVectorizer.
def clean(df):
        
    
    def ascii_only(row):
        text = ''
        for letter in row:
            if (ord(letter) < 97 or ord(letter) > 122) and ord(letter) != 32:
                pass
            else:
                text += letter
        return text
    
    porter = PorterStemmer()
    
    def stemmer(row):
        text = ''
        row = row.split()
        for word in row:
            text += porter.stem(word) + ' '
        return text
    
    df['question_text'] = df['question_text'].str.lower()
    print('*** TEXT LOWERED ***')
    df['question_text'] = df['question_text'].apply(ascii_only)
    print('*** ASCII-fied ***')
    df['question_text'] = df['question_text'].apply(stemmer)
    print('*** Stemmed ***')
    return df

# Read data
df = pd.read_csv('train.csv', 
                 low_memory=False,
                 index_col='qid')

# Stopwords, from stopwords-json
stop = pd.read_csv('stopwords.csv')
stop = list(stop['0'])
additions = ['ain', 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 
              'isn', 'll', 'mon', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']

for i in additions:
    stop.append(i)

x = df['question_text']
y = df['target']

# Split with a test size of 10% stratified
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.1, 
                                                    random_state=1,
                                                    shuffle=True,
                                                    stratify=y)

# 'All Removed', no additional steps in CountVectorizer and lowercase=False
print('ALL REMOVED')
count = CountVectorizer(lowercase=False)

x_train_bag_of_words = count.fit_transform(x_train)
x_test_bag_of_words = count.transform(x_test)

clf = MultinomialNB()
model = clf.fit(x_train_bag_of_words, y_train)
pred_y = model.predict(x_test_bag_of_words)

print(f'F1: {f1_score(y_test, pred_y)}')
print(f'Precision: {precision_score(y_test, pred_y)}')
print(f'Recall: {recall_score(y_test, pred_y)}')

###

# 'Lower', lowercase=True only
print('LOWER')
count = CountVectorizer(lowercase=True)

x_train_bag_of_words = count.fit_transform(x_train)
x_test_bag_of_words = count.transform(x_test)

clf = MultinomialNB()
model = clf.fit(x_train_bag_of_words, y_train)
pred_y = model.predict(x_test_bag_of_words)

print(f'F1: {f1_score(y_test, pred_y)}')
print(f'Precision: {precision_score(y_test, pred_y)}')
print(f'Recall: {recall_score(y_test, pred_y)}')

###

# 'Stop', stopwords removed only
print('STOP')
count = CountVectorizer(stop_words=stop)

x_train_bag_of_words = count.fit_transform(x_train)
x_test_bag_of_words = count.transform(x_test)

clf = MultinomialNB()
model = clf.fit(x_train_bag_of_words, y_train)
pred_y = model.predict(x_test_bag_of_words)

print(f'F1: {f1_score(y_test, pred_y)}')
print(f'Precision: {precision_score(y_test, pred_y)}')
print(f'Recall: {recall_score(y_test, pred_y)}')

###
# 'Lower + Stop', lowercase and stopwords removed
print('LOWER + STOP')
count = CountVectorizer(stop_words=stop,
                        lowercase=True)

x_train_bag_of_words = count.fit_transform(x_train)
x_test_bag_of_words = count.transform(x_test)

clf = MultinomialNB()
model = clf.fit(x_train_bag_of_words, y_train)
pred_y = model.predict(x_test_bag_of_words)

print(f'F1: {f1_score(y_test, pred_y)}')
print(f'Precision: {precision_score(y_test, pred_y)}')
print(f'Recall: {recall_score(y_test, pred_y)}')

###
# 'Lower + Stop + Stem', lowercase, stopwords removed and text stemmed
# Note that this is the only preprocessing iteration that uses the clean() function.
print('LOWER + STOP + STEM')

df = clean(df)
count = CountVectorizer()

x_train_bag_of_words = count.fit_transform(x_train)
x_test_bag_of_words = count.transform(x_test)

clf = MultinomialNB()
model = clf.fit(x_train_bag_of_words, y_train)
pred_y = model.predict(x_test_bag_of_words)

print(f'F1: {f1_score(y_test, pred_y)}')
print(f'Precision: {precision_score(y_test, pred_y)}')
print(f'Recall: {recall_score(y_test, pred_y)}')