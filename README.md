# Quora-Insincere-Questions-Classification

## 0. Executive Summary
Based on the ![Kaggle competition](https://www.kaggle.com/c/quora-insincere-questions-classification), classifies the sincerity of Quora questions using F1 score and a Multinomial Naive Bayes model, with a final F1 score around 0.536.

## 1. Introduction
‘Sincerity’ is an interesting measure. Unlike ‘toxic’ comments or questions – which can largely be identified through the use of particular words, often in reference to minority groups – ‘sincerity’ seems to be a more difficult concept to measure. Much like a rhetorical question, it’s less about the use of particular words and more about context and a human-like cognitive understanding of when a question is asking for information and when it is simply being asked for the sake of division or inflammation. With this in mind, I was drawn to the Kaggle Quora Insincere Questions Classification competition, whose goal is to determine whether a question is ‘sincere’ or not using a dataset of roughly 1.3 million questions from Quora. Despite the fact that this project involves simple binary classification (insincere or sincere), the top leaderboard F1 score was only around 0.7, meaning the task itself was quite challenging.

To solve the problem I utilised nested cross-validation to determine both the most appropriate model and the best hyperparameters for each model. I also included a number of preprocessing techniques and steps in both sklearn’s native library and manual application, however, interestingly the highest-scoring model utilised no preprocessing whatsoever although this did involve a trade-off between precision and recall (for the total F1 score). In addition, I was curious about the nature of an ‘insincere’ question as opposed to a ‘toxic’ question and did perform some rudimentary analysis to better understand the dataset, with the results of this analysis being available in the
‘Implementation’ section of this report. I also analysed the final output of the model (outside the scope of testing) to better understand why it generated false positives and false negatives to improve performance of similar models in the future.

## 2. Implementation
There were a number of factors to consider when approaching this problem, including high-level analysis of the dataset to better understand it, the scoring metric used, and algorithm selection and dataset split.

### 2.1 - Analysis
At a high level, the dataset contains 1,302,122 questions, with sincere questions making up 1,225,312 million (93.8%) and insincere making up the remaining 80,810 (6.2%). These are fairly imbalanced classes, however, as they’re both quite large I believed there would be sufficient examples of both for the model to make a useful decision regarding whether an example was sincere or insincere (as opposed to if there were only, say, 100 examples of insincere questions, in which case I would have had to find a solution to balance the datasets).

At the outset, my hypothesis was that unlike a toxic question, an 'insincere' question is harder to spot likely because it contains fewer of the same terms. Interestingly this seemed to be incorrect: traditional toxic dogwhistles including racist and sexist language appear relatively frequently in this dataset, with, for example, the word 'trump' appearing in 7.6% of all insincere samples but only 0.6% of all sincere samples. Figure 1 (next page) shows that the majority of the 40 most common words in the insincere dataset (with some stopwords removed) appear far more frequently in the insincere than the sincere dataset, with notable exceptions being for words that don’t necessarily have negative connotations in isolation (e.g., one, cant, feel, know, get). This made me very curious as to how the model would perform, as there would likely be some issues when some terms appear much more frequently in one class than another.

<img src="https://github.com/sklavoug/Quora-Insincere-Questions-Classification/blob/main/2.1.png" alt="Figure 1: 40 most frequent words in insincere dataset as a proportion of insincere and sincere datasets" width="1000"/>
<font size='6'> *italics* Figure 1: 40 most frequent words in insincere dataset as a proportion of insincere and sincere datasets</font size>
