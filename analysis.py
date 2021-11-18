# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:28:53 2021

@author: SKLAVOUG
"""

# analysis.py
# Initial analysis of the dataset. Starting hypothesis was that an 'insincere'
# question would be harder to identify than, say, a 'toxic' question because
# toxic questions would share a vocabulary more frequently than insincere
# questions. This was not the case, however, as the analysis showed that
# words which might appear in a toxic comment or question still appeared
# in the insincere dataset.

import pandas as pd
from nltk.corpus import stopwords

# insincere_sincere_word_comparison
# Function to take most frequently-occurring words in 'insincere' dataset and
# check how often they appear (i.e., in how many documents).
def insincere_sincere_word_comparison():
    def insincere_sample(df):
        
        stop = stopwords.words('english')
        
        stop.append('')
        stop.append('dont')
        stop.append('would')
        
        # Convert to lower-case, expand the column of lists into rows and remove
        # duplicates
        text = df['question_text'].str.lower().str.split()
        
        text = text.explode('question_text')
    
        # Remove any characters that aren't lower-case alphabetical (including punctuation)    
        text = text.apply(lambda x: ''.join(["" if ord(i) < 97 or ord(i) > 122 else i for i in x]))
        
        text = text.groupby(text).count().sort_values(ascending=False)
        
        text = pd.DataFrame(text)
        
        text = text.loc[~text.index.isin(stop)]
        
        return text.head(40)
    
    # Read csv and get insincere sample
    df = pd.read_csv('train.csv')
    insic_common = insincere_sample(df.loc[df['target'] == 1])
    
    # Standardise sincere and insincere samples and then compare
    insincere = df['question_text'].loc[df['target'] == 1].str.lower()
    sincere = df['question_text'].loc[df['target'] == 0].str.lower()
    
    proportions = pd.DataFrame(data={'Insincere': [0],
                                     'Sincere': [0]},
                                     index=insic_common.index)
    
    for i in insic_common.index:
        temp_ins = insincere.str.count(i)
        temp_sin = sincere.str.count(i)
        prop_ins = len(temp_ins.loc[temp_ins != 0]) / len(insincere)
        prop_sin = len(temp_sin.loc[temp_sin != 0]) / len(sincere)
        proportions.loc[i] = [prop_ins,prop_sin]
    
    print(proportions)
    
    return proportions

comp = insincere_sincere_word_comparison()
comp = comp.reset_index()

# Convert to percentages and plot
comp['Insincere'] = comp['Insincere'] * 100
comp['Sincere'] = comp['Sincere'] * 100

ax = comp.plot.bar(x='question_text', 
                   y=['Insincere','Sincere'],
                   figsize=(20,8),
                   fontsize=20)