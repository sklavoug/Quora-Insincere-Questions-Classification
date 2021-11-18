# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:28:53 2021

@author: SKLAVOUG
"""

# detailed_output.py
# Analysis of model features (in features.csv). Strictly for my own knowledge
# and improvement in the future, as analysing the results and applying
# the knowledge gained could lead to overfitting on the test set.

import pandas as pd

def impactful_insincere(df):
    df = df.sort_values(by='Insincere', ascending=False)

    df = df.head(40)
    
    ax = df.plot.bar(x='Features', 
                        y=['Sincere','Insincere'],
                        figsize=(20,8),
                        fontsize=20)
    

# Generate graph of most impactful words for the insincere class
df = pd.read_csv('features.csv')

impactful_insincere(df.copy())

def difference(df):
    df = df.sort_values(by='Difference', ascending=False)

    df = df.head(40)
    
    ax = df.plot.bar(x='Features', 
                        y=['Difference'],
                        figsize=(20,8),
                        fontsize=20,
                        color='#00682f')

# Generate graph of largest differences between sincere and insincere probabilities
df['Difference'] = ((df['Sincere'] - df['Insincere'])**2)**(1/2)

difference(df)