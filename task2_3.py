#!/usr/bin/env python
# coding: utf-8

# description

# In[1]:


import numpy as np
from nltk.probability import *
from itertools import chain

def stats_print(tk_description):
    words = list(chain.from_iterable(tk_description)) # we put all the tokens in the corpus in a single list
    vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
    lexical_diversity = len(vocab)/len(words)
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of description:", len(tk_description))
    lens = [len(article) for article in tk_description]
    print("Average description length:", np.mean(lens))
    print("Maximun description length:", np.max(lens))
    print("Minimun description length:", np.min(lens))
    print("Standard deviation of description length:", np.std(lens))


# ## Importing libraries 

# In[2]:


from itertools import chain
import pandas as pd

# Code to import libraries as you need in this assessment, e.g.,
# Read job_ad.csv
job_ad = pd.read_csv('job_ad.csv')
# print first 3 rows
job_ad.head(3)
# get the description of the job ad
description = job_ad['Description']
# get the tokenized description of the job ad
tk_description = job_ad['Tokenized Description']
webindex = job_ad['Webindex']
vocab = sorted(list(chain.from_iterable(tk_description)))
print(tk_description)
len(vocab)


# In[3]:


descriptionFile = './description.txt'
with open(descriptionFile) as f:
    tk_description = f.read().splitlines() # read all the descriptions into a list


# In[4]:


print(len(tk_description))
emp = 10
tk_description[emp]


# #### Converting each description text string into list of tokens

# In[5]:


tk_description = [description.split(" ") for description in tk_description] # note that we have to revert the join string into list of tokens
tk_description[emp]


# #### Explore the current statistics

# In[6]:


stats_print(tk_description)


# #### Reading the corresponding category labels

# In[7]:


categoryFile = './category.txt'
with open(categoryFile) as f:
    category = f.read().splitlines() # read all the category into a list


# #### Making sure we done it right
# Take an example, e.g., the 10th element

# In[8]:


emp = 10
print(len(category))
category[emp]


# Convert the loaded category labels to integers:

# In[9]:


category = [int(s) for s in category]


# In[10]:


sum(category) # seeing the total number of


# ## Task 2. Processing the Most and Less Frequent Words¶
# 
# In task 2, you are required to write codes to explore and handle the most and less frequent words.

# ### Task 2.1 Most Frequent Words
# 
# In this subtask, you will write code to explore the most frequent words in the pre-processed tokenized review text corpus. You will need to:
# * explore the most frequent words (top 25) based on term frequency and document frequency, respectively
# * compare the results using different frequency measurements, which words are extracted based on both frequency measurements?
# * think and decide on whether or not you would remove some of the most frequent words

# In[11]:


from nltk.probability import *
from itertools import chain

words = list(chain.from_iterable(tk_description)) # we put all the tokens in the corpus in a single list


# ### Most frequent words w.r.t. Term Frequency
# We first explore the most frequent words in terms of term frequency:

# In[12]:


term_fd = FreqDist(words) # compute term frequency for each unique word/type


# In[13]:


term_fd.most_common(25)


# ### Most frequent words w.r.t. Document Frequency
# We then explore the most frequent words in terms of document frequency:

# In[14]:


words_2 = list(chain.from_iterable([set(review) for review in tk_description]))
doc_fd = FreqDist(words_2)  # compute document frequency for each unique word/type
doc_fd.most_common(25)


# The list seems quite similar, let's what are in common and what are different based on the two frequency measurements.

# In[15]:


tf_words = set(w[0] for w in term_fd.most_common(25))
df_words = set(w[0] for w in doc_fd.most_common(25))

tf_words.union(df_words) # frequent words in both measurements


# In[16]:


# words are most frequent based on term frequence, but not document frequence
tf_words.difference(df_words)


# In[17]:


# words are most frequent based on document frequence, but not term frequence
df_words.difference(tf_words)


# Indeed, most of thes words seems to have a bit of taste (indication on sentiment).
# We decided not to remove them.

# ### Task 2.2 Less Frequent Words
# 
# Now, let's move on to the less frequent words. In this subtask, you are required to:
# * find out the list of words that appear only once in the **entire corpus**
# * remove these less frequent words from each tokenized review text
# 

# We first need to find out the set of less frequent words by using the `hapaxes` function applied on the **term frequency** dictionary.

# In[18]:


lessFreqWords = set(term_fd.hapaxes())
lessFreqWords


# In[19]:


len(lessFreqWords)


# Oh, a lot!!! Many of them appear to be quite ad hoc.
# Let's remove them:

# In[20]:


def removeLessFreqWords(review):
    return [w for w in review if w not in lessFreqWords]

tk_description = [removeLessFreqWords(review) for review in tk_description]


# In[21]:


stats_print(tk_description)


# ## Task 3. Finding Bigrams
# 
# In this task, you are required to explore the bigrams (top 25) in the pre-processed review text. You are also required to write code to include the bigrams that you think make sense in to the vocabulary.

# Finding the list of top 25 bigrams:

# In[22]:


from nltk.util import ngrams
bigrams = ngrams(words, n = 2)
fdbigram = FreqDist(bigrams)


# In[23]:


bigrams = fdbigram.most_common(25) # top 25 bigrams
bigrams


# In[24]:


rep_patterns = [" ".join(bg[0]) for bg in bigrams]
rep_patterns


# Most of them make sense and constructed meaningful phase, except `film like`,`film not`,`movie like`,`movie not`.
# Therefore, we will include all the bigrams in the vocabulary, except the above mentioned ones.

# In[25]:


filtered = ['film like','film not','movie like','movie not'] # define a list of bigrams that we won't include
rep_patterns = [bg for bg in rep_patterns if bg not in filtered] # create a list of bigrams that we want to include
rep_patterns


# In[26]:


replacements = [bg.replace(" ","_") for bg in rep_patterns] # convert the format of bigram into word1_word2
replacements


# In the following, we join each tokenized review text, and replace the bigrams with the format 'word1_word2', and then we re-tokenized them again into list of tokens. As such, each bigram that we want to include in the vocabulary will become a single token.

# In[27]:


import re
tk_description = [" ".join(review) for review in tk_description] # construct the review string

for i in range(0, len(tk_description)):
    for j in  range(0,len(rep_patterns)):
        tk_description[i] = re.sub(rep_patterns[j], replacements[j], tk_description[i]) # replace with bigram representation

tk_description = [review.split(" ") for review in tk_description] # convert back to tokenised review


# Have a look at the stats again :)

# In[28]:


stats_print(tk_description)


# ## Task 4. Constructing the Vocabulary
# 
# Now, we complete all the basic pre-process step and we are ready to move to feature generation! &#129321;
# Before we start, in this task, you are required to construct the final vocabulary, e.g., `vocab`:

# In[29]:


# generating the vocabulary

words = list(chain.from_iterable(tk_description)) # we put all the tokens in the corpus in a single list
vocab = sorted(list(set(words))) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words

len(vocab)


# ## Task 5. Generating Feature Vectors
# 
# In this task, we are going to generate feature vectors from tokenized review text. We are going to explore different feature vectors, including binary, count, and tf-idf vectors.

# ### Task 5.1 Generating Binary Vectors
# In this subtask, let's start with generating the binary vector representation for each review.

# We need to first import the `CountVectorizer` and initialise it.

# In[30]:


# binding the words together for each review
joined_description = [' '.join(review) for review in tk_description]


# In[31]:


from sklearn.feature_extraction.text import CountVectorizer
bVectorizer = CountVectorizer(analyzer = "word",binary = True,vocabulary = vocab) # initialise the CountVectorizer


# In[32]:


binary_features = bVectorizer.fit_transform(joined_description)
binary_features.shape


# ### Task 5.2 Generating Count Vectors
# 
# In this subtasks, you are required to generate the count vector features of review texts.

# In[33]:


cVectorizer = CountVectorizer(analyzer = "word",vocabulary = vocab) # initialised the CountVectorizer
count_features = cVectorizer.fit_transform(joined_description)
count_features.shape


# ### Task 5.3 Generating TF-IDF Vectors
# 
# In this subtasks, you are required to generate the count vector features of review texts.

# In[34]:


from sklearn.feature_extraction.text import TfidfVectorizer
tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab) # initialised the TfidfVectorizer
tfidf_features = tVectorizer.fit_transform(joined_description) # generate the tfidf vector representation for all articles
tfidf_features.shape


# In[35]:


joined_description


# In[36]:


tfidf_features


# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>Task 2. Generating Feature Representations</strong></h3>
# 
# So let's say we do binary feature representation but with 3 types of data, the title, the description, and title+description.

# In[37]:


from collections import Counter


"""
Bag-of-words model:
Generate the Count vector representation for each job advertisement description, and save
them into a file (please refer to the required output). Note, the generated Count vector
representation must be based on the generated vocabulary in Task 1 (as saved in vocab.txt).
"""
# bag of words model
def bag_of_words(description, vocab):
    # create a list of 0s with the same length as the vocab
    bow = [0] * len(vocab)
    # count the number of times each word appears in the description
    word_counts = Counter(description)
    # update the bow list with the word counts
    for word, count in word_counts.items():
        bow[vocab.index(word)] = count
    return bow

# Generate the Count vector representation for each job advertisement description
bow = [bag_of_words(description, vocab) for description in tk_description]


# In[38]:


bow


# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>2.1 Saving outputs</strong></h3>
# 
# Save the count vector representation as per spectification.
# - `count_vectors.txt`
# 
# `count_vectors.txt` stores the sparse count vector representation of job advertisement descriptions in the following format. Each line of this file corresponds to one advertisement. It starts with a ‘#’ key followed by the webindex of the job advertisement, and a comma ‘,’. The rest of the line is the sparse representation of the corresponding description in the form of word_integer_index:word_freq separated by comma. Following is an example of the file format.

# In[39]:


# save count vector representation of job advertisement descriptions
with open('count_vectors.txt', 'w') as f:
    for i, description in enumerate(tk_description):
        f.write('#' + str(webindex[i]) + ',')
        for word in description:
            f.write(str(vocab.index(word)) + ':' + str(bow[i][vocab.index(word)]) + ',')
        f.write('\n')
    print("Successfully write count vector representation of job advertisement descriptions into count_vectors.txt file")


# ## Task 6. Training Logistic Regression Models for Sentiment Analysis
# 
# In this final task, you are required to bulid logistic regression models based on different feature vectors (binary, count and tfidf vectors), explore the cases where the model produced false predictions, and evaluate the performance of the model using a 10-fold cross-validation.

# In the following, we first uses count vector features as an example to bulid a logistic regression model and  explore the preformance of the model:

# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

seed = 15
X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(count_features, category, list(range(0,len(category))),test_size=0.2, random_state=seed)

max_iter = 1000 # this is a relative harder problem and we have to increase the maximum iteration parameter of the logistic regression model.

model = LogisticRegression(random_state=seed,max_iter=max_iter) # initial a logistic regression model
model.fit(X_train, y_train) # fit the model
model.score(X_test, y_test) # calculated the accuracy score on the test data


# Looking at the confusion matrix

# In[43]:


from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)


# In[44]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

categories = ['Neg','Pos'] # this gives sorted set of unique label names

sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=categories, yticklabels=categories) # creates a heatmap from the confusion matrix
plt.ylabel('Actual')
plt.xlabel('Predicted')


# Let's explore some mis-classified examples.

# In[45]:


import random

y_pred_prob = model.predict_proba(X_test) # instead of getting the direct prediction, i.e., a label, we can also get the probability distributions of labels

for p_ind in range(0, 2):
    for a_ind in range(0, 2):
        if p_ind !=  a_ind: # when it mis-classifies
            print("'{}' predicted as '{}' : {} examples.".format(categories[a_ind], categories[p_ind],                                                                 conf_mat[a_ind, p_ind]))
            print("====================================================")

            # retrieve the indices for the mis-classification
            mis_inds = [test_indices[i] for i in range(0,len(y_test)) if                         y_pred[i] == p_ind and y_test[i] == a_ind]
            #print out the article ID and the tokenised text content of the mis-classified examples
            for ind in random.sample(mis_inds,2): # explore 2 examples
                print("------------------------------------------------")
                print(joined_reviews[ind])
                print("-----------------------------------------------\n")
            print()


# ### 10-Fold Cross Validation

# In[46]:


from sklearn.model_selection import KFold
num_folds = 10
kf = KFold(n_splits= num_folds, random_state=seed, shuffle = True) # initialise a 5 fold validation
print(kf)


# In[47]:


def evaluate(X_train,X_test,y_train, y_test,seed):
    model = LogisticRegression(random_state=seed,max_iter = 1000)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression

num_models = 3
cv_df = pd.DataFrame(columns = ['binary','count','tfidf'],index=range(num_folds)) # creates a dataframe to store the accuracy scores in all the folds

fold = 0
for train_index, test_index in kf.split(list(range(0,len(category)))):
    y_train = [str(category[i]) for i in train_index]
    y_test = [str(category[i]) for i in test_index]

    X_train_binary, X_test_binary = binary_features[train_index], binary_features[test_index]
    cv_df.loc[fold,'binary'] = evaluate(binary_features[train_index],binary_features[test_index],y_train,y_test,seed)

    X_train_count, X_test_count = count_features[train_index], count_features[test_index]
    cv_df.loc[fold,'count'] = evaluate(count_features[train_index],count_features[test_index],y_train,y_test,seed)

    X_train_tfidf, X_test_tfidf = tfidf_features[train_index], tfidf_features[test_index]
    cv_df.loc[fold,'tfidf'] = evaluate(tfidf_features[train_index],tfidf_features[test_index],y_train,y_test,seed)

    fold +=1


# Printing the result of each fold for each vector representation:

# In[ ]:


cv_df


# In[ ]:


cv_df.mean()


# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>Task 3. Job Advertisement Classification</strong></h3>

# ...... Sections and code blocks on buidling classification models based on different document feature represetations. 
# Detailed comparsions and evaluations on different models to answer each question as per specification. 
# 
# <span style="color: red"> You might have complex notebook structure in this section, please feel free to create your own notebook structure. </span>

# In[ ]:


# Code to perform the task...


# In[41]:


import os

# The .py format of the jupyter notebook
for fname in os.listdir():
    if fname.endswith('ipynb'):
        os.system(f'jupyter nbconvert {fname} --to python')


# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>Summary</strong></h3>
# Give a short summary and anything you would like to talk about the assessment tasks here.

# ## Couple of notes for all code blocks in this notebook
# - please provide proper comment on your code
# - Please re-start and run all cells to make sure codes are runable and include your output in the submission.   
# <span style="color: red"> This markdown block can be removed once the task is completed. </span>
