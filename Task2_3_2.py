#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# 
# <h3 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Task 2&3.<br>Feature Representation & Classification</strong></h3>
# 
# #### Student Name: Tran Ngoc Anh Thu
# #### Student ID: s3879312
# 
# Date: "October 2, 2022"
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used (please go to `requirements.txt` file for further details)
# * sklearn
# * collections
# * re
# * numpy
# * nltk
# * itertools
# * pandas
# * os
# * pylab
# * collections
# 
# ## Introduction
# 
# Machine and Algorithm can not understand lossing categorical data. Therefore, we should encode those text into numerical values using feature representation
# We have pre-processed data in Task1. Once the text data is cleaned and tokenized it is ready for NLP analysis. Vectorization of the tokens allows us to mathematically represent text as vectors. There are numerous ways to create these vectors.
# 
# + Compare between ML models with different feature representation (count vector, weigthed and unweighted embedding models)
# 
# + Single model (e.g., logistic regression with count vector representation), and compare the performance with different amount of info. 
# 
# + Unweighted, you just sum the word embeddings of an job ad as the representation of an job ad. 
# 
# + Weighted sum, ou can do a weighted sum of word embeddings as representation instead of just sum.
# 
# 
# ### Multi-class classification 
# + Model: we will use Linear Model (Logistic Regression) combine with different attributes and feature representations. 
# + Model evaluation:K-fold cross-validation (5 folds here) to avoid overfitting the data
# + Metric: we will chose Accuracy to compare our models
# > * `accuracy = correct_predictions / total_predictions`
# 
# ## Steps
# + 2.1. Examining and loading data
# 
# + 1.2. Basic Text Pre-processing
#     * Feature Representation: Binary, Count, TF-IDF
#     * Classification with Feature Representation
#     * FastText embedding model with unweighted and weighted vector
#     * Comparision
#     
# + 1.3. Summary
# > * Discussion
#       
# + 1.4. References
# 
# 
# ## Dataset
# + A small collection of job advertisement documents (around 776 jobs) inside the `data` folder.
# + Inside the data folder, there are four different sub-folders: Accounting_Finance, Engineering, Healthcare_Nursing, and Sales, representing a job category.
# + The job advertisement text documents of a particular category are in the corresponding sub-folder.
# + Each job advertisement document is a txt file named `Job_<ID>.txt`. It contains the title, the webindex (some will also have information on the company name, some might not), and the full description of the job advertisement.

# In[1]:


from itertools import chain
from nltk.probability import *
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# import code as a function
from src.utils import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# ignore warning
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# logging for event tracking
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Let's read in the labels and article data, and construct a single dataframe to store them correspondingly. 
# Note that the order of the articles are the same as the labels data, as they are all in sorted order of document names. 

# In[3]:


# Read job_ad.csv
job_ad = pd.read_csv('job_ad.csv')
webindex = job_ad['Webindex']


# print first 3 rows
job_ad.head(3)


# Read in the article text, and creates another list to store the tokenized version of the article text accordingly.

# In[4]:


descriptionFile = './description.txt'
with open(descriptionFile) as txtf:
    description_txts = txtf.read().splitlines() # reading a list of strings, each for a document/article
tk_description = [a.split(' ') for a in description_txts]


# Store the article text (as well as the tokenized version) to the dataframe.

# In[5]:


job_ad['Tokenized Description'] = tk_description


# In[6]:


job_ad.sample(n = 5) # look at a few examples


# In[7]:


# read the category of the job ad
categoryFile = './category.txt'
with open(categoryFile) as f:
    category = f.read().splitlines() # read all the category into a list
    
print(len(category))
type(category)


# <h3 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Task 3. Classification with FastText Word Embedding Models</strong></h3>
# 
# * generate document embeddings based on the load FastText word embeddings;
# * explore the reprensentiveness of the features through tSNE;
# * bulid the logistic regression model based on the generated document embeddings for news classfication.

# In[9]:


# loading the trained Fasttext model based on Job Ad data
from gensim.models.fasttext import FastText

# Defining values for parameters
embedding_size = 200
window_size = 5
min_word = 5
down_sampling = 1e-2
 
get_ipython().run_line_magic('time', '')
bbcFT = FastText(tk_description,
                      vector_size=embedding_size,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                      workers = 5,
                      sg=1,
                      epochs=100)


# In[10]:


# save model 
from gensim.models import Word2Vec

fast_text_model_file = 'fast_text_model'

# # Save fastText gensim model
# bbcFT.save("models/FastText/fast_Text_model")
# print(f'Successfully loaded {fast_text_model_file}')

# Load saved gensim fastText model
bbcFT = Word2Vec.load("models/FastText/fast_Text_model")


# <h3 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Classification with Unweighted document vectors</strong></h3>

# In[11]:


# NOTE this can take some time to finish running
# generate document embeddings
bbcFT_wv = bbcFT.wv
bbcFT_dvs = gen_docVecs(bbcFT_wv,job_ad['Tokenized Description'])
bbcFT_dvs.isna().any().sum()


# In[12]:


# explore feature space
features = bbcFT_dvs.to_numpy()
plotTSNE(job_ad['Category'],features)


# ### --------> OBSERVATION:
# 
# The unweighted embeding model seems does not do a good job of classifying features into their appropriate categories with only 0.3 Accuracy score

# In[13]:


# build the classification model and report results
seed = 3879312
X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(bbcFT_dvs, job_ad['Category'], list(range(0,len(job_ad))),test_size=0.33, random_state=seed)

model = LogisticRegression(max_iter = 1000,random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# <h3 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Classification with TF-IDF weighted document vectors</strong></h3>
# 
# # Map TF-IDF from scratch

# In[14]:


def read_vocab(vocab_file):
    vocab = {}
    with open(vocab_file) as f:
        for line in f:
            (word, index) = line.split(':')
            vocab[word.strip()] = int(index)
    return {v: k for k, v in vocab.items()}

# Generates the w_index:word dictionary
voc_fname = 'vocab.txt'
voc_dict = read_vocab(voc_fname)
voc_dict


# - the `doc_wordweights` function takes the tfijob_ad document vector file, as well as the w_index:word dictionary, creates the mapping between w_index and the actual word, and creates a dictionary of word:weight or each unique word appear in the document.

# In[15]:


def doc_wordweights(fName_tVectors, voc_dict):
    tfidf_weights = [] # a list to store the  word:weight dictionaries of documents
    
    with open(fName_tVectors) as tVecf: 
        tVectors = tVecf.read().splitlines() # each line is a tfidf vector representation of a document in string format 'word_index:weight word_index:weight .......'
    for tv in tVectors: # for each tfidf document vector
        tv = tv.strip()
        weights = tv.split(' ') # list of 'word_index:weight' entries
        weights = [w.split(':') for w in weights] # change the format of weight to a list of '[word_index,weight]' entries
        wordweight_dict = {voc_dict[int(w[0])]:w[1] for w in weights} # construct the weight dictionary, where each entry is 'word:weight'
        tfidf_weights.append(wordweight_dict) 
    return tfidf_weights

fName_tVectors = 'jobAd_tVector.txt'
tfijob_ad_weights = doc_wordweights(fName_tVectors, voc_dict)

# take a look at the tfijob_ad word weights dictionary of the first document
tfijob_ad_weights[0]


# Ok, once we have the word:weight dictionary of each document, now we can construct the tf-idf weighted document embeddings. 
# * the following `gen_docVecs` function is an revision/extension of the previous written function, that takes the word embeddings dictionary, the tokenized text of articles, and the tfijob_ad weights (list of word:weight dictionaries, one for each article) as arguments, and generates the document embeddings:
#  1. creates an empty dataframe `docs_vectors` to store the document embeddings of articles
#   2. it loop through every tokenized text:
#     - creates an empty dataframe `temp` to store all the word embeddings of the article
#     - for each word that exists in the word embeddings dictionary/keyedvectors, 
#         - if the argument `tfijob_ad` weights are empty `[]`, it sets the weight of the word as 1
#         - otherwise, retrieve the weight of the word from the corresponding word:weight dictionary of the article from  `tfijob_ad`
#     - row bind the weighted word embedding to `temp`
#     - takes the sum of each column to create the document vector, i.e., the embedding of an article
#     - append the created document vector to the list of document vectors

# In[16]:


# extended version of the `gen_docVecs` function
def gen_docVecs(wv,tk_txts,tfijob_ad = []): # generate vector representation for documents
    docs_vectors = pd.DataFrame() # creating empty final dataframe
    #stopwords = nltk.corpus.stopwords.words('english') # removing stop words

    for i in range(0,len(tk_txts)):
        tokens = list(set(tk_txts[i])) # get the list of distinct words of the document

        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for w_ind in range(0, len(tokens)): # looping through each word of a single document and spliting through space
            try:
                word = tokens[w_ind]
                word_vec = wv[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
                
                if tfijob_ad != []:
                    word_weight = float(tfijob_ad[i][word])
                else:
                    word_weight = 1
                temp = temp.append(pd.Series(word_vec*word_weight), ignore_index = True) # if word is present then append it to temporary dataframe
            except:
                pass
        doc_vector = temp.sum() # take the sum of each column(w0, w1, w2,........w300)
        docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) # append each document value to the final dataframe
    return docs_vectors


# Ok we now have everything ready to generate the weight document embeddings. 
# We can do this on any of our previous explored models, including the pretrained Word2Vec GoogleNews300, Glove, our in-house trained Word2Vec and FastText.
# 
# Let's generated the weighted version of the document embedding vectors first

# In[17]:


weighted_bbcFT_company = gen_docVecs_weight(bbcFT_wv,job_ad['Tokenized Company'],tfijob_ad_weights)
weighted_bbcFT_title = gen_docVecs_weight(bbcFT_wv,job_ad['Tokenized Title'],tfijob_ad_weights)
weighted_bbcFT_description = gen_docVecs_weight(bbcFT_wv,job_ad['Tokenized Description'],tfijob_ad_weights)


# And we can do very much the same thing as what we do before for other models. 
# Here, we will do this as loops, for each model:
# - we plot out the feature vectors  projected in a 2-dimensional space,then 
# - we build the logistic regression model for document classfication and report the model performance.

# In[18]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
def plotTSNE(labels,features): # features as a numpy array, each element of the array is the document embedding of an article
    categories = sorted(labels.unique())
    # Sampling a subset of our dataset because t-SNE is computationally expensive
    SAMPLE_SIZE = int(len(features) * 0.3)
    np.random.seed(0)
    indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
    projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices].astype(int))
    colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey']
    for i in range(0,len(categories)):
        points = projected_features[(labels[indices] == categories[i])]
        plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[i], label=categories[i])
    plt.title("Feature vector for each article, projected on 2 dimensions.",
              fontdict=dict(fontsize=15))
    plt.legend()
    plt.show()


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
seed = 0
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

models = weighted_bbcFT_description
model_names = "Weighted Pretrained FastText with Description"

dv = models
name = model_names
features = dv.to_numpy() # convert the dataframe stored features to an numpy array
print(len(features))
print(name + ": tSNE 2 dimensional projected Feature space")
plotTSNE(job_ad['Category'],features)
    
# creating training and test split
X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(dv, job_ad['Category'], list(range(0,len(job_ad))),test_size=0.33, random_state=seed)

model = LogisticRegression(max_iter = 2000,random_state=seed)
# X_train = np.array(X_train).reshape(-1, 1)
# y_train = np.array(y_train).reshape(-1, 1)
# y_test = np.array(y_test).reshape(-1, 1)
# X_test = np.array(X_test).reshape(-1, 1)
model.fit(X_train, y_train)
print("Accuracy: ", model.score(X_test, y_test))
print('\n\n')


# In[22]:


type(features)


# In[23]:


job_ad.info()


# #### Note: Creating tf-ifd weighted document embeddings using Gensim

# In[24]:


from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel

# we have two vocabularies here, one from the in-house bulit Word2Vec, the other from the articles 
# note that althought the Word2Vec is built on the same dataset, but they might have done further 
# pre-processing during model build (e.g., setting min_count), and thus, might create mismatch in the two vocabularis. 
# therefore, we remove tokenized words that doesn't exist in the keyedvectors in the Word2Vec keyedvectors
processed_text = [[w for w in t if w in bbcFT_wv.index_to_key] for t in job_ad['Tokenized Description']] 

# use the Gensim package to create a dictionary that encapsulates the mapping between normalized words and their integer ids.
docs_dict = Dictionary(processed_text) # creates a dictionary from the text
docs_dict.filter_extremes(no_below=5) # filtering words that appear less than 5 times
docs_dict.compactify() # assign new word ids to all words, shrinking any gaps.


# In[25]:


# see what are the words that been get rid off when we do the fliter
[w for w in bbcFT_wv.index_to_key if w not in docs_dict.values()]


# In[26]:


len(docs_dict.keys())


# This process yields a vocabulary with 8647 words. 
# Then we use Gensim again to create a bag-of-words representation of each document, i.e., the tf-idfvector for each document.
# 

# In[27]:


import numpy as np
from gensim.matutils import sparse2full

docs_corpus = [docs_dict.doc2bow(doc) for doc in job_ad['Tokenized Description']] # convert corpus to Bag of Word format
model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict) # fit the tfijob_ad model
# apply model to the list of corpus document, 
# so each document is a list of tuples, (word_index, weight) for each word appears in the document
docs_tfidf  = model_tfidf[docs_corpus]


# In[28]:


# see for example, the tfijob_ad weights of the words in the 2nd document
docs_tfidf[1]


# We can then transfer `docs_tfidf` to matrix form. `vstack` function from numpy can stack arrays in sequence vertically (row wise), and `sparse2full` function convert a document in sparse document format (in size of the number of words in the document) into a dense numpy array (of size of the vocabulary)

# In[29]:


docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])


# In[30]:


docs_vecs.shape


# In[31]:


docs_vecs[0]


# The result, `docs_vecs`, is a matrix with 2225 rows (docs) and 8647 columns (tf-idfterms). 

# Let's see the performance of this tf-idfvector:

# In[32]:


# creating training and test split
X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(docs_vecs, job_ad['Category'], list(range(0,len(job_ad))),test_size=0.33, random_state=seed)

model = LogisticRegression(max_iter = 1000,random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# ### --------> OBSERVATION
# 
# Best Accuracy !

# In[33]:


# see how many words are encoded in our in-house Word2Vec model?
len(bbcFT_wv.index_to_key)


# In[34]:


# how about the tfijob_ad vector?
len(docs_dict)


# We only care about words that are in both vocabulary. 
# In the following, we creates the word embeddings arrays for words exists in docs_dict

# In[35]:


word_emb_vecs = np.vstack([bbcFT_wv[docs_dict[i]] for i in range(len(docs_dict)) if docs_dict[i] in bbcFT_wv.index_to_key])


# In[36]:


word_emb_vecs.shape


# In[37]:


word_emb_vecs[0]


# Now we can do the dot product of the two vectors and get our tfijob_ad weighted document embeddings:

# In[38]:


tfijob_ad_docs_emb = np.dot(docs_vecs, word_emb_vecs) 


# In[39]:


# creating training and test split
X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(tfijob_ad_docs_emb, job_ad['Category'], list(range(0,len(job_ad))),test_size=0.33, random_state=seed)

model = LogisticRegression(max_iter = 1000,random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>References</strong></h3>
# 
# + [1] [Multinomial Logistic Regression With Python](https://machinelearningmastery.com/multinomial-logistic-regression-with-python/)
# + [2] [Quick Introduction to Bag-of-Words (BoW) and TF-IDF for Creating Features from Text](https://www.analyticsvidhya.com/blog/2020/02/quick-introduction-bag-of-words-bow-tf-idf/)
# + [3] [FastText paper from Facebook](https://arxiv.org/pdf/1607.04606.pdf)
# + [4] [Gensim’s fastText](https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html#sphx-glr-auto-examples-tutorials-run-fasttext-py)
# + [5] https://thinkinfi.com/fasttext-word-embeddings-python-implementation/
# + [6] [TFIDF with Word Embeddings](https://github.com/ytnvj2/DocumentEmbedding/blob/master/TFIDFwithEmbeddings.ipynb)  
# + [7] [Problem-solving with ML: automatic document classification](https://cloud.google.com/blog/products/ai-machine-learning/problem-solving-with-ml-automatic-document-classification)     
# + [8] [Creating TF-IDF Weighted Word Embeddings](http://dsgeek.com/2018/02/19/tfidf_vectors.html)    
# + [9] [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
