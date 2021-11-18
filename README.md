# Quora-Insincere-Questions-Classification

## 0. Executive Summary and How to Run
Based on the ![Kaggle competition](https://www.kaggle.com/c/quora-insincere-questions-classification), classifies the sincerity of Quora questions using F1 score and a Multinomial Naive Bayes model, with a final F1 score around 0.536.

### PYTHON FILES
- analysis.py contains the analysis for 2. Implementation, particularly the generation of the graph showing most common words in the insincere dataset and their frequency in the sincere dataset.

- preprocessing.py includes the preprocessing steps which were experimented with but ultimately not used, as the model's performance actually improved with no preprocessing whatsoever.

- validation.py contains the nested cross-validations for the three models. The cross-validation's verbosity is set to the highest value, and the output in the console was converted to data and scores averaged/interrogated for the tables in part 3.

- Model is stored in the 'model.py' file. As it doesn't take very long to train and test, no outputs have been cached. Results are stored in 'results.csv' with 'target' being the target value and 'predicted' being the model's output on the test set. Resulting features in the model along with their probabilities for each class are stored in 'features.csv'.

- detailed_output.py uses the output of the model to analyse the results and determine factors influencing the model's classification.

### CSV FILES (not included)
- train.csv should contain the dataset from Kaggle.

- stopwords.csv should contain expanded stopword list from stopwords-json, used in preprocessing.py

- features.csv should contain features used by the model and probabilities associated with each feature for each class.

- results.csv should contain model's predictions and true values for the test set.

## 1. Introduction
‘Sincerity’ is an interesting measure. Unlike ‘toxic’ comments or questions – which can largely be identified through the use of particular words, often in reference to minority groups – ‘sincerity’ seems to be a more difficult concept to measure. Much like a rhetorical question, it’s less about the use of particular words and more about context and a human-like cognitive understanding of when a question is asking for information and when it is simply being asked for the sake of division or inflammation. With this in mind, I was drawn to the Kaggle Quora Insincere Questions Classification competition, whose goal is to determine whether a question is ‘sincere’ or not using a dataset of roughly 1.3 million questions from Quora. Despite the fact that this project involves simple binary classification (insincere or sincere), the top leaderboard F1 score was only around 0.7, meaning the task itself was quite challenging.

To solve the problem I utilised nested cross-validation to determine both the most appropriate model and the best hyperparameters for each model. I also included a number of preprocessing techniques and steps in both sklearn’s native library and manual application, however, interestingly the highest-scoring model utilised no preprocessing whatsoever although this did involve a trade-off between precision and recall (for the total F1 score). In addition, I was curious about the nature of an ‘insincere’ question as opposed to a ‘toxic’ question and did perform some rudimentary analysis to better understand the dataset, with the results of this analysis being available in the
‘Implementation’ section of this report. I also analysed the final output of the model (outside the scope of testing) to better understand why it generated false positives and false negatives to improve performance of similar models in the future.

## 2. Implementation
There were a number of factors to consider when approaching this problem, including high-level analysis of the dataset to better understand it, the scoring metric used, and algorithm selection and dataset split.

### 2.1 - Analysis
At a high level, the dataset contains 1,302,122 questions, with sincere questions making up 1,225,312 million (93.8%) and insincere making up the remaining 80,810 (6.2%). These are fairly imbalanced classes, however, as they’re both quite large I believed there would be sufficient examples of both for the model to make a useful decision regarding whether an example was sincere or insincere (as opposed to if there were only, say, 100 examples of insincere questions, in which case I would have had to find a solution to balance the datasets).

At the outset, my hypothesis was that unlike a toxic question, an 'insincere' question is harder to spot likely because it contains fewer of the same terms. Interestingly this seemed to be incorrect: traditional toxic dogwhistles including racist and sexist language appear relatively frequently in this dataset, with, for example, the word 'trump' appearing in 7.6% of all insincere samples but only 0.6% of all sincere samples. Figure 1 (next page) shows that the majority of the 40 most common words in the insincere dataset (with some stopwords removed) appear far more frequently in the insincere than the sincere dataset, with notable exceptions being for words that don’t necessarily have negative connotations in isolation (e.g., one, cant, feel, know, get). This made me very curious as to how the model would perform, as there would likely be some issues when some terms appear much more frequently in one class than another.

#### Figure 1: 40 most frequent words in insincere dataset as a proportion of insincere and sincere datasets
<img src="https://github.com/sklavoug/Quora-Insincere-Questions-Classification/blob/main/2.1.png" alt="Figure 1: 40 most frequent words in insincere dataset as a proportion of insincere and sincere datasets" width="1000"/>

### 2.2 - Scoring
The scoring metric for this challenge was F1 score, which is the harmonic mean of precision (TP / (TP + FP)) and recall (TP / (TP + FN)). This is an appropriate measure for an NLP task relating to toxicity, as the precision will punish a model which is too assertive and uses a particular batch of words which appear commonly in toxic comments to make its decision, while the recall will punish a model which is too conservative in its judgements, incorrectly labelling true values as false. Given how many words appeared frequently in the insincere class but infrequently in the sincere class, I suspected any classification model would perform better on recall than precision, as it would be much more likely to incorrectly label a sincere example as insincere because it contained a certain word (e.g., ‘trump’) than to incorrectly label an insincere example as sincere for not containing words.

### 2.3 - Algorithm Selection and Dataset Split
The main three algorithms I compared are logistic regression, support vector machines, and multinomial Naïve-Bayes, as they were noted as being ![good baseline models](https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf) for text classification and all have existing implementations in sklearn (see Appendix Table 4 for a description).

As this Kaggle competition’s test set did not include the true values for each document, I decided instead to split the training set into a train and test set so I could benchmark my model’s performance. The main consideration in this case would be ensuring that the split favoured the training data (so the model could learn as much as possible) without having too few examples in the test set, however, as the dataset was quite large I thought it would be appropriate to split it with 90% in the training set and 10% in the test set, and notably ensuring that the sets were stratified so the training and test set would have the same proportion of sincere and insincere examples by using sklearn’s train_test_split function with its ‘stratify’ attribute set to true.

## 3. Experimentation
### 3.1 - Preprocessing
Most Natural Language Processing (NLP) is based on ‘vectorisation’, which essentially involves converting each unique word in the text into a column and having each document as a row, with the row x column value representing either the appearance of that word in the text (binary true/false), or the number of times that word appeared in the text (similar to the difference between Bernoulli NB and Multinomial NB), both of which are options in CountVectorizer’s ‘binary’ parameter. This generally leads to sparse matrices with a large number of columns, however, for the purpose of NLP this much information is unnecessary and can be simplified to improve model performance.

There are four key aspects of vectorisation which can affect model performance: word standardisation, stemming, stopwords, and n-grams. In terms of standardisation, vectorisation requires the data to be relatively uniform – the model will still function on non-uniform data, but its performance will be hampered. A good example of this is punctuation, as when vectorisation occurs the words ‘really’, ‘Really’, ‘REALLY’, ‘REALLY?’ and ‘really?’ will be considered completely different and treated as such, when realistically they are the same word. Fortunately sklearn’s CountVectorizer function splits data on symbols such as question marks, and also has a ‘lowercase’ attribute to set all text to lowercase by default (meaning that the five ‘really’s above will all be treated as the same word).

The second aspect is stemming, which relates closely to standardisation. Similarly to how the five ‘really’s are treated separately, there are also different versions of words which essentially mean the same thing. For example, the words ‘mean’, ‘meant’ and ‘meaning’ all share a common root (or stem) in the word ‘mean’, but much like the ‘really’s, they will be treated as different words. Stemming is a way of removing this excess and unnecessary complexity by stemming each word to its root, so all three variations would just be ‘mean’.

The final two aspects are stopwords and n-grams, both of which occur after the words have been standardised. As noted prior, NLP models use a word’s appearance and/or frequency to determine how to classify a document. There are, however, a number of words which will likely appear in all documents (e.g., ‘the’, ‘a’, ‘for’). These words would appear so frequently in all classes that they add very little to the model’s decision-making and are essentially noise, so can be removed (through the ‘stop_words’ parameter). In line with the recommendation from Kaggle’s ![Basic NLP with NLTK guide](https://www.kaggle.com/alvations/basic-nlp-with-nltk), I used the ![stopwords-json](https://github.com/6/stopwords-json/blob/master/dist/en.json) dataset which is more expansive than either sklearn’s or NLTK’s in-built stopword function.

Finally, n-grams are essentially phrases of length n extracted from the text, e.g., 1-grams are unigrams and represent one word, 2-grams are bigrams and represent two words, etc. These words are extracted from the documents and can be important in determining intent or toxicity, e.g., the words ‘trump’ and ‘sucks’ in isolation could lead the classifier to classify the comment as toxic, however, the combined ‘trump sucks’ may be far less common in toxic comments but more common in non-toxic comments, which would change the classification.

My hypothesis, based on the small amount of NLP work I’ve done in the past, was that each of these would improve the model in some way by reducing the overall pool of words it would have to recognise and enriching the data for each document. Interestingly the opposite seemed to be true – removing standardisation, stemming, and stopword removal actually improved the model’s performance, in some cases significantly.

#### Table 1: Preprocessing Experimentation with MNB Algorithm
|           | All Removed  | Lower + Stop  | Lower      | Stop   | Lower + Stop + Stem  |
| --------- | -----------: | ------------: | ---------: | ------:| -------------------: |
| F1        | **0.5629**   | 0.5472        | 0.5574     | 0.5441 | 0.5531               |
| Precision | 0.4776       | **0.491**     | 0.4665     | 0.4887 | 0.4727               |
| Recall    | 0.6854       | 0.618         | **0.6925** | 0.6138 | 0.6664               |

Table 1 shows the outcome of initial testing with a MNB model, and demonstrates that removing all preprocessing steps and simply vectorising the text with the in-built CountVectorizer (and setting lowercase to false) resulted in a higher F1 score than any of the other preprocessing steps. Interestingly this approach did not generate the highest precision or recall, but was instead a case of having a relatively high recall with a middling precision score, thereby achieving the best harmonic mean between the two.

Also interestingly, the best-performing preprocessing combinations for precision and recall were setting the text to lowercase and removing stopwords, and just removing stopwords respectively, however, both of these approaches yielded very low values of the other score among the combinations (i.e., high precision and low recall, or low precision and high recall). For this reason, preprocessing was experimented with heavily (including reproducing some of CountVectorizer’s attributes so NLTK’s stemming could be applied to the text prior to the models being fit), but ultimately setting lowercase to false yielded the best results. N-grams were the final aspect of preprocessing but in order to better test the effect, n-grams were included as part of the cross-validation.

### 3.2 - Nested Cross-Validation
![Nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) is a technique used to prevent information ‘leakage’. This essentially means that having access to both a training and test set can lead to overfitting through repeated attempts and iterations, i.e., I would train my model on the training set and tune its parameters based on performance on the test set, thereby removing the strict barrier between the two and letting information gained during training ‘leak’ into the test set.

Nested cross-validation reduces the risk of information leakage by repeatedly splitting the training data into train, validate, and test sets and then performing iterations with each option of the hyperparameters provided, thereby tuning the hyperparameters at the same time as the models’ scores are compared.

This is an important step as a particular model may work well on both the training and test sets with all their features, but very poorly with a subset of those features. A good example of this actually occurred during cross-validation, where the SVM model had the highest F1 score of all the models tested, however, its score was far less consistent than the MNB model (which was ultimately used) and its standard deviation was the highest of all (see Table 2). This indicates that the model performed well in particular tests but not others, and lacks the consistency one would want from a well-performing model, as it is important that a model be generalizable (i.e., can get a similar or consistent score regardless of the data it receives).

#### Table 2: Results of Nested Cross-Validation
|     | Highest F1 | Lowest F1 | Avg F1    | Stdev     | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4 |
| :-- | ---------: | --------: | --------: | --------: | ----------: | ----------: | ----------: | ----------: |
| SVM | **0.588**  | 0.047     | 0.4       | 0.16      | 0.473       | 0.403       | 0.336       | 0.382       |
| MNB | 0.556      | **0.47**  | **0.534** | **0.014** | **0.533**   | **0.534**   | **0.533**   | **0.537**   |
| LOG | 0.499      | 0.001     | 0.336     | 0.157     | 0.296       | 0.291       | 0.222       | 0.353       |

The nested cross-validation I implemented used a 5-fold split for the inner validation and a 3-fold split for the outer validation, using randomised search to determine parameters and then feeding the resulting classifier into sklearn’s cross_val_score function to determine the best score. Notably, grid search was initially used to tune parameters but proved to be extremely costly with such a large dataset, so randomised search was implemented instead to cross-validate with reduced computational expense.

Notably, a term-frequency times inverse document-frequency (tfidf) transformer was considered and experimented with, however, it seemed to reduce the performance of all models significantly (to as low as 0.001 F1 score) so was not implemented as part of nested cross-validation.

Within the MNB cross-validation, some interesting trends emerged. Table 3 shows that the highest- scoring iterations used a max_df near the max (either 0.75 or 1), consistently did not use lowercase, and had a relatively high alpha value (0.1 or 0.01), while the lowest-performing models had 0.5 max_df, used lowercase, and used the lowest alpha value (0.0001).

#### Table 3: Hyperparameters and Resulting F1 Score for MNB Model
| Ngram | Max_feat | Max_df | Lowercase | Alpha | F1     |
| :---- | -------: | -----: | --------: | ----: | -----: |
| (1,1) | 50,000   | 0.75   | FALSE     | 0.1   | 0.556  |
| (1,1) | None     | 1      | FALSE     | 0.1   | 0.554  |
| (1,1) | 50,000   | 0.75   | FALSE     | 0.1   | 0.556  |
| (1,1) | None     | 1      | FALSE     | 0.1   | 0.553  |
| (1,2) | None     | 1      | FALSE     | 0.01  | 0.553  |
| ...   | ...      | ...    | ...       | ...   | ...    |
| (1,2) | None     | 0.5    | TRUE      | 0.0001| 0.47   |
| (1,2) | None     | 0.5    | TRUE      | 0.0001| 0.471  |
| (1,2) | None     | 0.5    | TRUE      | 0.0001| 0.476  |
| (1,2) | None     | 0.5    | TRUE      | 0.0001| 0.48   |
| (1,2) | None     | 0.5    | TRUE      | 0.0001| 0.481  |

## 4. Results
### 4.1 - Output
The final output of the model with the top parameters from Table 3 was 0.536. Notably the model’s recall value was higher than any of the preprocessing test cases (Table 1) at 0.765, however, its precision was also much lower than any of the test cases, at 0.413. This is not a particularly good result, as it essentially means the model performed correctly about half the time, however, as noted in the introduction, the top score on the Kaggle leaderboards was around 70, indicating this is a challenging exercise to begin with. What follows is an analysis of the test set’s output and true values to better understand why the model performed the way it did, noting that any insights gleaned from this exercise would mostly be for personal education rather than further tuning of the model’s parameters or hyperparameters (as doing so would be the exact kind of information ‘leakage’ prevented by nested cross-validation, and would realistically most likely result in overfitting on the test set thus hindering the model’s generalisability).

### 4.2 - Detailed Output
#### Figure 2: Probability values for most impactful features in determining insincerity
<img src="https://github.com/sklavoug/Quora-Insincere-Questions-Classification/blob/main/4.2-1.png" alt="Figure showing high probability values for words like 'the', 'what', 'How', and 'in' for the Sincere subset, and higher values for 'Why', 'people', 'they', and 'and' in the Insincere subset" width="1000"/>

A detailed analysis of the results shows that different features have definitely impacted the model’s classification, however, these effects may have been small given how little the probability differences for some features is. Figure 2 shows the most impactful words in deciding that a document is insincere from the MNB classifier (i.e., the words with the highest probability of appearing in an insincere document). As noted in Table 1, a number of preprocessing steps were considered including stopword removal, however, removing the stopwords actually led to a slight increase in recall and a considerable decrease in precision, leading to a lower F1 score overall. This detailed analysis may suggest reasons why, as common stopwords (such as ‘the’, ‘to’, ‘and’, ‘of’) seem to have very little impact since their probabilities of appearing in insincere documents is roughly the same as them appearing in sincere documents.

#### Figure 3: Absolute difference in probability for sincere vs insincere class
<img src="https://github.com/sklavoug/Quora-Insincere-Questions-Classification/blob/main/4.2-2.png" alt="Figure showing absolute difference for 'What', 'Why', 'How', 'in', and 'is'" width="1000"/>

The detailed analysis also provides a potential explanation for why converting the text to lowercase reduced its F1 score. Figure 3 shows the two biggest differences between sincere and insincere probabilities in the dataset above are for ‘What’ (favours sincere, +0.01664) and ‘Why’ (favours insincere, +0.02481). The fact that these words are capitalised, and that their non-capitalised counterparts do not appear on the top list, suggests that they are appearing commonly at the start of a sentence or, more likely, as the first word in the question. Interestingly, this suggests that ‘Why’ questions are more likely to be insincere and ‘What’ questions are more likely to be sincere.

### 4.3 - Potential Improvement
There are several potential improvements and expansions to the model which could be implemented. The first is to use ![BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers), a language representation model developed by Google whose notable features include being pre- trained on an existing corpus of text and determining probability not just for features themselves, but features with multiple meanings (e.g., ‘mean’ as in intended, and ‘mean’ as in average). As well, I’d like to try and implement a neural network to compare deep learning techniques against the more linear models applied in this project.

Another potential improvement would be to consider work already conducted on unintended bias in toxic comment classification, most notably the jigsaw ![Kaggle competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) and ![Google paper](https://arxiv.org/pdf/1903.04561.pdf) relating to Area Under the Receiver Operating Characteristic Curve (ROC-AUC), which may significantly improve the model’s recall as it would offset the probability of words which appear frequently in toxic comments but could influence classification of non-toxic comments (e.g., the phrase ‘I am trans’ may not be toxic but will likely be labelled as such due to the appearance of the word ‘trans’, which would presumably appear often in toxic comments and lead to unintended bias).

## 5. References
Borkan et al., ‘Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification’, https://arxiv.org/pdf/1903.04561.pdf

Devlin et al., (2018) ‘BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding’, https://arxiv.org/abs/1810.04805

Graham, P., ‘stopwords-json’, Github repo, https://github.com/6/stopwords-json/blob/master/dist/en.json

Kaggle, ‘Jigsaw Unintended Bias in Toxicity Classification’, https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview

Kaggle, ‘Quora Insincere Questions Classification’, https://www.kaggle.com/c/quora-insincere-questions-classification/overview

Sklearn, ‘Nested versus non-nested cross-validation’, https://scikit- learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

Tan, L., ‘Basic NLP with NLTK’, Python notebook, https://www.kaggle.com/alvations/basic-nlp-with-nltk

Wang, S., and Manning, C.D., ‘Baselines and Bigrams: Simple, Good Sentiment and Topic Classification’, https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf

## 6. Appendix
<table>
    <thead>
        <tr>
            <th>Function / Algorithm</th>
            <th>Description</th>
            <th>Hyperparameter</th>
            <th>Description</th>
            <th>Values Cross-Validated</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>CountVectorizer</td>
            <td rowspan=4>Performs vectorisation on text data, i.e., converts words (or n-grams) in text to vectors with each word represented by a column and each document a row.</td>
            <td>max_df</td>
            <td>A kind of stopword removal, removes words which appear in the corpus more often than the decimal value.</td>
            <td>0.5, 0.75, 1.0</td>
        </tr>
        <tr>
            <td>max_features</td>
            <td>Only considers the top x words based on frequency.</td>
            <td>None, 5000, 10,000, 50,000</td>
        </tr>
        <tr>
          <td>ngram_range</td>
          <td>Number of words to tokenise together.</td>
          <td>Unigram and bigram.</td>
        </tr>
        <tr>
          <td>lowercase</td>
          <td>Convert characters to lowercase.</td>
          <td>True, False</td>
        </tr>
        <tr>
            <td>Multinomial Naive Bayes</td>
            <td>Utilises the Naive Bayes formula to determine class probabilities based on frequency of appearance of words in a document.</td>
            <td>alpha</td>
            <td>Smoothing parameter, used to prevent situations where a word not appearing will generate a probability of zero.</td>
            <td>0.0001, 0.001, 0.01, 0.1</td>
        </tr>
        <tr>
            <td rowspan=3>Support-Vector Machine</td>
            <td rowspan=3>Performs Linear classification by maximising the margin (i.e., the difference) between the two classes.</td>
            <td>C</td>
            <td>Regularisation parameter, with strength of regularisation being inversely proportional to C.</td>
            <td>0.0001, 0.001, 0.01, 0.1</td>
        </tr>
        <tr>
          <td>class_weight</td>
          <td>Used for imbalanced classes to scale weightings.</td>
          <td>balanced, None</td>
        </tr>
        <tr>
          <td>max_iter</td>
          <td>Maximum number of iterations.</td>
          <td>10, 50, 80</td>
        </tr>
        <tr>
            <td rowspan=3>Logistic Regression</td>
            <td rowspan=3>Performs regression to determine the probability that a document is of a particular class (i.e., the range from 0 to 1 are the real values it plots).</td>
            <td>C</td>
            <td>Regularisation parameter, with strength of regularisation being inversely proportional to C.</td>
            <td>0.0001, 0.001, 0.01, 0.1</td>
        </tr>
        <tr>
          <td>class_weight</td>
          <td>Used for imbalanced classes to scale weightings.</td>
          <td>balanced, None</td>
        </tr>
        <tr>
          <td>penalty</td>
          <td>Norm used in penalisation.</td>
          <td>l1, l2</td>
        </tr>
    </tbody>
</table>
