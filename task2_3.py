#!/usr/bin/env python
# coding: utf-8

# ## Intro
# 
# Once the text data is cleaned and tokenized it is ready for NLP analysis. Vectorization of the tokens allows data scientist to mathematically represent text as vectors. There are numerous ways to create these vectors.
# 
# 
# + compare between ML models with different feature representation (count vector, weigthed and unweighted embedding models)
# 
# + single model (e.g., logistic regression with count vector representation), and compare the performance with different amount of info. 
# 
# + Unweighted, you just sum the word embeddings of an job ad as the representation of an job ad. 
# 
# + Weighted sum, ou can do a weighted sum of word embeddings as representation instead of just sum. 
# 
# You can do that for any embedding models and generate two version of doc representation
# 
# ## bulid:
# 
# + count vector representation
# + weigthed embedding representation (use TF-IDF as the weight when you do weighted sum of word embeddings)
# + unweigted embedding representation. 
# 
# ## Question 
# + load pretrained model or train it from stractch

# In[10]:


from itertools import chain
from nltk.probability import *

# import code as a function
from src.utils import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
# set desired matplotlib gloabal figure size
plt.rcParams["figure.figsize"] = (20,10)


# ## Importing libraries

# In[11]:


# Read job_ad.csv
job_ad = pd.read_csv('job_ad.csv')

# # get the description of the job ad
# description = job_ad['Description']
# # get the tokenized description of the job ad
# tk_description = job_ad['Tokenized Description']
webindex = job_ad['Webindex']


# print first 3 rows
job_ad.head(3)


# In[12]:


descriptionFile = './description.txt'
with open(descriptionFile) as f:
    tk_description = f.read().splitlines() # read all the descriptions into a list
    
print(len(tk_description))
type(tk_description)


# #### Converting each description text string into list of tokens

# In[13]:


tk_description = [description.split(" ") for description in tk_description] # note that we have to revert the join string into

# Explore the current statistics
stats_print(tk_description)


# #### Reading the corresponding category labels

# In[14]:


# read the category of the job ad
categoryFile = './category.txt'
with open(categoryFile) as f:
    category = f.read().splitlines() # read all the category into a list
    
print(len(category))
type(category)


# #### Making sure we done it right
# Take an example, e.g., the 10th element

# In[15]:


print(f'The number of the category: {len(category)}')
print(f'The number of the description: {len(tk_description)}')
if len(category) == len(tk_description):
    print(f'The number of category of category and description are the same and corresponding to each other')


# In[16]:


test_index = 20


# ### ----------------> OBSERVATION
# 
# We can see the length of the description text and corresponding labels are equal.

# Convert the loaded category labels to integers:

# ## Constructing the Vocabulary
# 
# Now, we complete all the basic pre-process step and we are ready to move to feature generation! &#129321;
# Before we start, in this task, you are required to construct the final vocabulary, e.g., `vocab`:

# In[17]:


# generating the vocabulary

words = list(chain.from_iterable(tk_description)) # we put all the tokens in the corpus in a single list
vocab = sorted(list(set(words))) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words

len(vocab)
print(f'The number of the category: {len(category)}')
print(f'The number of the description: {len(tk_description)}')
print(f'The number of the unique vocabulary: {len(category)}')

# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>Task 2. Generating Feature Representations</strong></h3>
# 
# So let's say we do binary feature representation but with 3 types of data, the title, the description, and title+description.

# In[18]:


from collections import Counter

"""
Bag-of-words model using CountVectorizer
Generate the Count vector representation for each job advertisement description, and save
them into a file (please refer to the required output). Note, the generated Count vector
representation must be based on the generated vocabulary in Task 1 (as saved in vocab.txt).
"""
cVectorizer = CountVectorizer(analyzer = "word",vocabulary = vocab) # initialised the CountVectorizer
count_features = cVectorizer.fit_transform(joined_description).toarray()



# ## 2.1 Saving outputs
# Save the count vector representation as per spectification.
# - `count_vectors.txt`
# 
# `count_vectors.txt` stores the sparse count vector representation of job advertisement descriptions in the following format. Each line of this file corresponds to one advertisement. It starts with a ‘#’ key followed by the webindex of the job advertisement, and a comma ‘,’. The rest of the line is the sparse representation of the corresponding description in the form of word_integer_index:word_freq separated by comma. Following is an example of the file format.
def save_count_vector(count_features, webindex, filename):
    with open(filename, 'w') as f:
        for i in range(len(count_features)):
            f.write('#' + str(webindex[i]) + ',')
            for j in range(len(count_features[i])):
                if count_features[i][j] != 0:
                    f.write(str(j) + ':' + str(count_features[i][j]) + ',')
            f.write('\n')
    f.close()
    print('Count vector representation saved to ' + filename)


# In[19]:



# save count vector representation of job advertisement descriptions
with open('count_vectors.txt', 'w') as f:
    for i, description in enumerate(tk_description):
        f.write('#' + str(webindex[i]) + ',')
        for word in description:
            f.write(str(vocab.index(word)) + ':' + str(bow[i][vocab.index(word)]) + ',')
        f.write('\n')
    print("Successfully write count vector representation of job advertisement descriptions into count_vectors.txt file")


# ## Building Vector Representation
# 
# After text pre-processing has been completed, each individual document needs to be transformed into 
# some kind of numeric representation that can be input into most NLP and text mining algorithms.
# For example, classification algorithms, such as Support Vector Machine, can only take data in a 
# structured and numerical form. They do not accept free language text.
# A popular structured representation of text is the vector-space model, which represents each text/article
# as a vector where the elements of the vector indicate the occurence of words within the text.
# 
# The vector-space model makes an implicit assumption that 
# the order of words in a text document are not as
# important as words themselves, and thus disregarded.
# This assumpiton is called [**Bag-of-words**](https://en.wikipedia.org/wiki/Bag-of-words_model).
# 
# Given a set of documents and a pre-defined list of words appearing 
# in those documents (i.e. a vocabulary), we can compute a vector representation for each document.
# This vector representation can take one of the following three forms:
# * a binary representation, each entry is either `word:0` (the word does not appear in the document; or `word:1` (the word appears in the document). We call this **binary vector representation**. 
# * an integer count, each entry is `word:count`, telling how many times a word appear in a document. We call this **count vector representation**. 
# * and a float-valued weighted vector, each entry is `word:weight`, telling a **weighted representative importance** of a word to a document. One of the most common weighted vectors used in natural language processing is called the *tfidf* vector. 
# 
# Given the cleaned up BBC News articles, how can we generate those vectors for each document? 
# 
# Unfortunately, NLTK does not implement methods that directly produce those vectors.
# Therefore, we will either write our own code to compute them or appeal to other data analysis libraries.
# 
# Here we are going to use [scikit-learn](http://scikit-learn.org/stable/index.html), an open source machine 
# learning library for Python.
# If you use Anaconda, you should already have scikit-learn installed, otherwise you will need to 
# [install it](http://scikit-learn.org/stable/install.html) by following the instruction on its official website.
# 
# Although scikit-learn features various classification, regression and clustering algorithms
# we are particularly interested in its feature extraction module, [sklearn.feature_extraction](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction).
# This module is often used to "extract features in a format supported by machine learning algorithms from datasets consisting of formats such as text and image." Please refer to its documentation on text feature extraction,
# Section 6.2.3 of [Feature Extraction](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction). 
# 
# In the following, we will demonstrate the usage of the following two classes:
# * [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer): It converts a collection of text documents to a matrix of token counts. 
# * [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer):
# It converts a collection of raw documents to a matrix of TF-IDF features.
# 
# 
# ### 5.3 Generating TF-IDF Vectors
# 
# Finally, we will generate the TF-IDF Vector to represent each of the document.
# 
# Similar to the use of `CountVector`, we first initialise a `TfidfVectorizer` object by only specifying the value of "analyzer" and the vocabulary, and then covert the BBC News articles into a list of strings, each of which corresponds
# to a BBC News article.

# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer
tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab) # initialised the TfidfVectorizer
tfidf_features = tVectorizer.fit_transform([' '.join(article) for article in tk_description]) # generate the tfidf vector representation for all articles
tfidf_features.shape


# In[21]:


# # print out the weighted vector for the example document.
# validator(tfidf_features,vocab,test_ind, article_ids,article_txts,tk_description)


# ### 7.3 Saving the Vector Representation
# 
# Given the vocabulary, each document can be represented as a sequence of entries that correspond to the tokens, or in the following sparse form:
# ```
#     word_index:word_count
# ```
# where the `word_count` is either 0 or 1 or binary vector, and an integer value for count vector; or 
# ```
#     word_index:weight
# ```
# for the TF-IDF vector. 
# In the following, we save the binary, count and tfidf vector representation of the documents, respectively. 
# Each of the following block of codes loops through each document, and each token appears in the document (retrived by the index `f_ind`. 
# As the data features is of the dimension of the size of the vocab, but there are only a limited number of words appear in a document, we retrieve the index of the non-zero entry of the data features by calling the `nonzero()` function:
# * the [`nonzero()`](https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html) function return the indices of the elements that are non-zero. These indices are returned as a tuple of arrays, one for each dimension of the matrix, containing the indices of the non-zero elements in that dimension. 
# 
# Note that here `data_features[a_ind]` returns a (1 x vocabSzie) sparse matrix of type '<class 'numpy.int64'>'. 
# Therefore, the return from 
# ```python
# binary_features[a_ind].nonzero()
# ```
# is a tuple of two arrays, the first array (indexed 0) is the indices of non-zero elements in the row dimension, and the second array (indexed 1) is the indicies of the non-zero elements in the column dimension. 
# Here we need to take the column dimension (indexed of a word in the vocabulary that appear in the document), therefore we take the 2nd array (retrieved by index 1). 
# Note also that the element of the 1st array is always 0, as mentioned before, the dimension of the matrix is 1 times the size of vocabulary. 
# 
# For each word index that has a non-zero entry, one could then retrieve the feature value by indexing, for example:
# ```python
# count_features[a_ind][0,f_ind]
# ```
# this retrieves the frequency count of word (indexed `f_ind`) in the document `count_features[a_ind]`.

# In[22]:


# Saving the Vector Representation
def write_vectorFile(data_features,filename):
    num = data_features.shape[0] # the number of document
    out_file = open(filename, 'w') # creates a txt file and open to save the vector representation
    for a_ind in range(0, num): # loop through each article by index
        for f_ind in data_features[a_ind].nonzero()[1]: # for each word index that has non-zero entry in the data_feature
            value = data_features[a_ind][0,f_ind] # retrieve the value of the entry from data_features
            out_file.write("{}:{} ".format(f_ind,value)) # write the entry to the file in the format of word_index:value
        out_file.write('\n') # start a new line after each article
    out_file.close() # close the file
    
tVector_file = "./jobAd_tVector.txt" # file name of the tfidf vector

write_vectorFile(tfidf_features,tVector_file) # write the tfidf vector to file
print(f'Successfully saved {tVector_file} into the directory')


# ### 7.4 Saving the Article IDs
# 
# Oh sorry, one last thing..... we should also save the article IDs accordingly, so that we can easily retrieve the labels of the article in downstream analysis. 
# 
# A very important note that the Article ID is different from the article index here. 
# The Article ID is given from the original dataset, as an identifier of an article. 
# The article index we were talking about in a couple of places in this Jupyter Notebook is related to how we read and preprocess the data. 
# Remember, at the beginning, we have read all the articles into a list `article_txts`, so the index here refers to the index of the article in the list  `article_txts`. 
# 
# Indeed, in this example, since the article IDs are integers, we should have make this more consistent (just in case...). 
# This could be simply done by setting an order when we read the articles. 
# The following code that we use before to loop through each of the article .txt file
# ```python
# for filename in os.listdir(dir_path):
# ``` 
# can be changed to: 
# ```python
# for filename in sorted(os.listdir(dir_path)):
# ``` 
# 
# as such, we visit the article txt files in a sorted order according to their filename. &#128578; We will leave this for you to try and experience the difference. 
# 
# Though, in many other context, we might not have that much luck with the file naming :) Therefore, we should always learn to save the indexing we used.

# In[23]:


# dir_path = "./articles"
# article_ids = [] # list to store the article ID
# article_txts = [] # list to store the raw text

out_file = open("./webindex.txt", 'w') # creates a txt file named 'bbcNews_articleIDs.txt' to save the tfidf vector
for a_ind in range(0, len(tk_description)):
    out_file.write("{}\n".format(webindex[a_ind])) # write the article ID of each article
out_file.close() # close the file


# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>Task 2. Generating Feature Representations</strong></h3>
# 
# So let's say we do binary feature representation but with 3 types of data, the title, the description, and title+description.

# ## Task 5. Generating Feature Vectors
# 
# In this task, we are going to generate feature vectors from tokenized review text. We are going to explore different feature vectors, including binary, count, and tf-idf vectors.

# ### Task 5.1 Generating Binary Vectors
# In this subtask, let's start with generating the binary vector representation for each review.

# We need to first import the `CountVectorizer` and initialise it.

# In[24]:


# binding the words together for each review
joined_description = [' '.join(review) for review in tk_description]


# In[25]:


from sklearn.feature_extraction.text import CountVectorizer
bVectorizer = CountVectorizer(analyzer = "word",binary = True,vocabulary = vocab) # initialise the CountVectorizer


# In[26]:


binary_features = bVectorizer.fit_transform(joined_description)
binary_features.shape


# In[27]:


tVector_file = "./jobAd_bVector.txt" # file name of the tfidf vector

write_vectorFile(binary_features,tVector_file) # write the tfidf vector to file
print(f'Successfully saved {tVector_file} into the directory')


# ### Task 5.2 Generating Count Vectors
# 
# In this subtasks, you are required to generate the count vector features of review texts.

# In[28]:


cVectorizer = CountVectorizer(analyzer = "word",vocabulary = vocab) # initialised the CountVectorizer
count_features = cVectorizer.fit_transform(joined_description)
count_features.shape


# In[29]:


tVector_file = "./jobAd_cVector.txt" # file name of the tfidf vector

write_vectorFile(cVectorizer,tVector_file) # write the tfidf vector to file
print(f'Successfully saved {tVector_file} into the directory')


# ### Task 5.3 Generating TF-IDF Vectors
# 
# ![](media/images/td-idf-graphic.png)
# Source: http://filotechnologia.blogspot.com/2014/01/a-simple-java-class-for-tfidf-scoring.html
# 
# In this subtasks, you are required to generate the count vector features of review texts.

# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer
tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab) # initialised the TfidfVectorizer
tfidf_features = tVectorizer.fit_transform(joined_description) # generate the tfidf vector representation for all articles
tfidf_features.shape


# In[31]:


tfidf_features


# In[32]:


tfidf_features


# ## 4. Generating TF-IDF weighted document vectors
# 
# Ok, I hope you have lots of fun building document embeddings based on varoius word embedding models. 
# Previously, when we generate the document embeddings, we just sum up the embeddings vector of each tokenized word in the article, a bit simplicity 🤔
# 
# In this section, let's make it a bit more challenging, we are going to build the tf-idf document embeddings. 
# What does that mean? 🤨
# Hmm~~ it's not magic, we just do a weigthed sum of the word embedding vectors, however, the weight here, refers to the tf-idf weight of the word. 
# 
# If you already forgot about what is `tf-idf`, please refer to Activity 3 Pre-processing Text and Generating Features. 
# Otherwise, move on!
# So we've generated the tf-idf vector representation of documents in Activity 3 and saved in a txt file called `bbcNews_tVector.txt`. The format of this file is:
# - each line represents an article;
# - each line is of the format 'w_index:weight w_index:weight ......' 
# 
# Oh, but we don't have that word index `w_index` here in this activity, what should we do? 🤔
# ah ha, we also saved the vocabulary in a file `bbcNews_voc`, in which each line is a word, and of the format `index,word`. 
# Theresore, based on these two files, we can create a word:weight mapping for each tokenized word in a document!
# 
# Ok, in the following couple block of codes, this is exactly what we are trying to do, step by step. 
# - the `gen_vocIndex` function reads the the vocabulary file, and create an w_index:word dictionary

# In[33]:


def read_vocab(vocab_file):
    vocab = {}
    with open(vocab_file) as f:
        for line in f:
            (word, index) = line.split(':')
            vocab[word.strip()] = int(index)
            # swap the key and value of the dictionary
    return {v: k for k, v in vocab.items()}



# Generates the w_index:word dictionary
voc_fname = 'vocab.txt'
voc_dict = read_vocab(voc_fname)
voc_dict


# ### --------> OBSERVATION
# 
# - the `doc_wordweights` function takes the tfidf document vector file, as well as the `word_string:word_integer_index` dictionary, creates the mapping between word_integer_index and the actual word, and creates a dictionary of word:weight or each unique word appear in the document.

# In[34]:


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
tfidf_weights = doc_wordweights(fName_tVectors, voc_dict)

# take a look at the tfidf word weights dictionary of the first document
tfidf_weights[0]


# ### ------------------> OBSERVATION:
# 
# Ok, once we have the word:weight dictionary of each document, now we can construct the tf-idf weighted document embeddings. 
# * the following `gen_docVecs` function is an revision/extension of the previous written function, that takes the word embeddings dictionary, the tokenized text of articles, and the tfidf weights (list of word:weight dictionaries, one for each article) as arguments, and generates the document embeddings:
#  1. creates an empty dataframe `docs_vectors` to store the document embeddings of articles
#   2. it loop through every tokenized text:
#     - creates an empty dataframe `temp` to store all the word embeddings of the article
#     - for each word that exists in the word embeddings dictionary/keyedvectors, 
#         - if the argument `tfidf` weights are empty `[]`, it sets the weight of the word as 1
#         - otherwise, retrieve the weight of the word from the corresponding word:weight dictionary of the article from  `tfidf`
#     - row bind the weighted word embedding to `temp`
#     - takes the sum of each column to create the document vector, i.e., the embedding of an article
#     - append the created document vector to the list of document vectors

# In[ ]:


# Generated the weighted version of the document embedding vectors first
weighted_bbcFT_dvs = gen_docVecs(bbcFT_wv,df['tk_description'],tfidf_weights)


# And we can do very much the same thing as what we do before for other models. 
# Here, we will do this as loops, for each model:
# - we plot out the feature vectors  projected in a 2-dimensional space,then 
# - we build the logistic regression model for document classfication and report the model performance.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
seed = 0
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

models = [weighted_preTW2v_dvs,weighted_preTGloVe_dvs,weighted_bbcW2v_dvs,weighted_bbcFT_dvs]
model_names = ["Weighted Pretrained Word2Vec", "Weighted Pretrained GloVe", "Weighted In-house Word2Vec","Weighted In-house FastText"]
for i in range(0,len(models)): #loop through each model
    dv = models[i]
    name = model_names[i]
    features = dv.to_numpy() # convert the dataframe stored features to an numpy array
    print(name + ": tSNE 2 dimensional projected Feature space")
    plotTSNE(df['Category'],features)
    
    # creating training and test split
    X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(dv, df['Category'], list(range(0,len(df))),test_size=0.33, random_state=seed)

    model = LogisticRegression(max_iter = 2000,random_state=seed)
    model.fit(X_train, y_train)
    print("Accuracy: ", model.score(X_test, y_test))
    print('\n\n')


# # OR
# 
# #### Note: Creating tfidf weighted document embeddings using Gensim
# 
# In the previous sections, we tried very hard to create the tfidf weighted document embeddings using the generated tf-idf weights save in previous activity. 
# Indeed, we can using Genism to do this direction, and it's indeed, a bit less effor required 😑 Will show you below. 
# We will use the in-house build Word2Vec model as an example.

# In[93]:


job_ad.columns


# In[94]:


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


# In[95]:


# see what are the words that been get rid off when we do the fliter
[w for w in bbcFT_wv.index_to_key if w not in docs_dict.values()]


# In[96]:


len(docs_dict.keys())


# In[97]:


import numpy as np
from gensim.matutils import sparse2full

docs_corpus = [docs_dict.doc2bow(doc) for doc in job_ad['Tokenized Description']] # convert corpus to Bag of Word format
model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict) # fit the tfidf model
# apply model to the list of corpus document, 
# so each document is a list of tuples, (word_index, weight) for each word appears in the document
docs_tfidf  = model_tfidf[docs_corpus]

# see for example, the tfidf weights of the words in the 2nd document
docs_tfidf[1]


# In[81]:


# Weighted sum, ou can do a weighted sum of word embeddings as representation instead of just sum.
def create_weighted_embedding_models(X):
    global weighted_features
    weighted_features = []
    for i in range(len(X)):
        weighted_features.append(np.average(embedding_matrix[X[i]], axis=0, weights=weights[i]))


# create logistic regression with unweighted embedding representation
def create_unweighted_embedding_models(X):
    global unweighted_features
    unweighted_features = []
    for i in range(len(X)):
        unweighted_features.append(np.sum(embedding_matrix[X[i]], axis=0))


# In[19]:


# extended version of the `gen_docVecs` function
def gen_docVecs(wv,tk_txts,tfidf = []): # generate vector representation for documents
    docs_vectors = pd.DataFrame() # creating empty final dataframe
    #stopwords = nltk.corpus.stopwords.words('english') # removing stop words

    for i in range(0,len(tk_txts)):
        tokens = list(set(tk_txts[i])) # get the list of distinct words of the document

        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for w_ind in range(0, len(tokens)): # looping through each word of a single document and spliting through space
            try:
                word = tokens[w_ind]
                word_vec = wv[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
                
                if tfidf != []:
                    word_weight = float(tfidf[i][word])
                else:
                    word_weight = 1
                temp = temp.concat(pd.Series(word_vec*word_weight), ignore_index = True) # if word is present then append it to temporary dataframe
            except:
                pass
        doc_vector = temp.sum() # take the sum of each column(w0, w1, w2,........w300)
        docs_vectors = docs_vectors.concat(doc_vector, ignore_index = True) # append each document value to the final dataframe
    return docs_vectors


# In[20]:


# save count vector representation of job advertisement descriptions
with open('count_vectors.txt', 'w') as f:
    for i, description in enumerate(tk_description):
        f.write('#' + str(webindex[i]) + ',')
        for word in description:
            f.write(str(vocab.index(word)) + ':' + str(bow[i][vocab.index(word)]) + ',')
        f.write('\n')
    print("Successfully write count vector representation of job advertisement descriptions into count_vectors.txt file")


# ## Task 6. Training Logistic Regression Models for label Classification
# 
# In this final task, you are required to bulid logistic regression models based on different feature vectors (binary, count and tfidf vectors), explore the cases where the model produced false predictions, and evaluate the performance of the model using a 5-fold cross-validation.

# In the following, we first uses count vector features as an example to bulid a logistic regression model and  explore the preformance of the model:

# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

seed = 3879312
X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(count_features, category, list(range(0,len(category))),test_size=0.2, random_state=seed)

max_iter = 1000 # this is a relative harder problem and we have to increase the maximum iteration parameter of the logistic regression model.

model = LogisticRegression(random_state=seed,max_iter=max_iter, multi_class='multinomial') # initial a logistic regression model
model.fit(X_train, y_train) # fit the model
model.score(X_test, y_test) # calculated the accuracy score on the test data


# Looking at the confusion matrix

# In[22]:


from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)


# In[23]:


categories = ['Accounting_Finance', 'Engineering', 'Healthcare_Nursing', 'Sales'] # this gives sorted set of unique label names

sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=categories, yticklabels=categories, cmap="PiYG") # creates a heatmap from the confusion matrix
plt.ylabel('Actual')
plt.xlabel('Predicted')


# Let's explore some mis-classified examples.

# In[24]:


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
                print(joined_description[ind])
                print("-----------------------------------------------\n")
            print()


# ### 5-Fold Cross Validation

# In[25]:


from sklearn.model_selection import KFold
num_folds = 5
kf = KFold(n_splits= num_folds, random_state=seed, shuffle = True) # initialise a 5 fold validation
print(kf)


# In[26]:


def evaluate(X_train,X_test,y_train, y_test,seed):
    model = LogisticRegression(random_state=seed,max_iter = 1000)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[27]:


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

# In[28]:


cv_df


# In[29]:


cv_df.mean()


# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>Task 3. Job Advertisement Classification</strong></h3>

# In[30]:


def gen_docVecs(wv,tk_txts): # generate vector representation for documents
    docs_vectors = pd.DataFrame() # creating empty final dataframe
    #stopwords = nltk.corpus.stopwords.words('english') # if we haven't pre-processed the articles, it's a good idea to remove stop words

    for i in range(0,len(tk_txts)):
        tokens = tk_txts[i]
        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for w_ind in range(0, len(tokens)): # looping through each word of a single document and spliting through space
            try:
                word = tokens[w_ind]
                word_vec = wv[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
                temp = pd.concat([temp, pd.Series(word_vec)], ignore_index = True)
            except:
                pass
        doc_vector = temp.sum() # take the sum of each column
        docs_vectors = pd.concat([docs_vectors, doc_vector], ignore_index = True)
    return docs_vectors


# ...... Sections and code blocks on buidling classification models based on different document feature represetations. 
# Detailed comparsions and evaluations on different models to answer each question as per specification. 
# 
# <span style="color: red"> You might have complex notebook structure in this section, please feel free to create your own notebook structure. </span>

# ## 3. FastText model trained on BBC News data 
# 
# trained our `Word2Vec` and `FastText` models using our BBC News dataset
# 
# ## 3. FastText
# 
# As mentioned before, `word2vec` model does not accommodate words that do not appear in the training corpus. 
# 
# Here, we’ll learn to work with the fastText library for training word-embedding models, and performing similarity operations & vector lookups analogous to Word2Vec. 
# 
# In the following block of code, we import the `FastText` model form Gensim library, then:
# 1. We set the path to the corpus file. Similar as above, we use the Bbc News article as the training corpus;
# 2. Initialise the `FastText` model, similar as before, we use 100 dimention vectors;
# 3. Then we build the vocabulary from the copurs;
# 4. Finally, we train the fasttext model based on the corpus.
# 
# Finally, we experiment the FastText embeddings. 
# Similar, we:
# * load the FastText model saved in our prevoius activity;
# * generate document embeddings based on the load FastText word embeddings;
# * explore the reprensentiveness of the features through tSNE;
# * bulid the logistic regression model based on the generated document embeddings for news classfication.
# 
# T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, A. Joulin. Advances in Pre-Training Distributed Word Representations
# 
# 
# There are multiple pre-trained models in Gensim, see Section **Pretrained models** in https://radimrehurek.com/gensim/models/word2vec.html.

# In[31]:


# from gensim.models.fasttext import FastText

# # 1. Set the corpus file names/path
# corpus_file = './bbcNews.txt'

# # 2. Initialise the Fast Text model
# bbcFT = FastText(vector_size=100) 

# # 3. build the vocabulary
# bbcFT.build_vocab(corpus_file=corpus_file)

# # 4. train the model
# bbcFT.train(
#     corpus_file=corpus_file, epochs=bbcFT.epochs,
#     total_examples=bbcFT.corpus_count, total_words=bbcFT.corpus_total_words,
# )

# print(bbcFT)


# In[32]:


# # We can retrieve the KeyedVectors from the model as follows,

# bbcFT_wv = bbcFT.wv
# print(bbcFT_wv)


# In[33]:


# # Save the model
# bbcFT.save("bbcFT.model")


# In[34]:


# loading the trained Fasttext model based on bbc News data
from gensim.models.fasttext import FastText
bbcFT = FastText.load("bbcFT.model")
print(bbcFT)
bbcFT_wv= bbcFT.wv


# In[35]:


job_ad.columns


# In[36]:


# NOTE this can take some time to finish running
# generate document embeddings
bbcFT_dvs = gen_docVecs(bbcFT_wv,job_ad['Tokenized Description'])
bbcFT_dvs.isna().any().sum()


# In[37]:


bbcFT_dvs


# ### ----------> OBSERVATION
# 
# 0 null record

# ### 1.1 Understand your task by tSNE
# 
# Alright! so we have the document embedding vector representation for each article now, we can proceed to the task of document classification. 
# Before, we move on, a good habbit is to explore and understand how difficult the task is, whether there are too much noise in the data, making it impossible to clearly separate each category. 
# 
# One way to confirm that the feature space we are using is representative enough for our task (classifying articles into separate labels) to be solvable is to use dimensionality-reduction techniques: These methods project a high-dimensional vector into a lower number of dimensions, with different guarantees on this projection according to the method used. 
# In this activity, we will use [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding), a popular dimensionality reduction technique used in many fields, including NLP.
# 
# Since we will do the same thing again and again when we try other embeddings, we will construct a function to do this again.
# The following function `plotTSNE` takes the following arugments:
# * labels, the lable/category of each article
# * features, a numpy array of document embeddings, each for an article.
# 
# and projects the feature/document embedding vectors in a 2 dimension space and plot them out. 
# It does the following:
# 1. get the set of classes, called `categories` (5 categories)
# 2. sample 30% of the data/document embeddings randomly, and record the indices selected
# 3. project the selected document embeddings in 2 dimensional space using tSNE, each document embedding now corresponds to a 2 dimensional vector in `projected_features`
# 4. plot them out as scatter plot and highlight different categories in different color

# In[38]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
def plotTSNE(labels,features): # features as a numpy array, each element of the array is the document embedding of an article
    categories = sorted(labels.unique())
    # Sampling a subset of our dataset because t-SNE is computationally expensive
    SAMPLE_SIZE = int(len(features) * 0.3)
    np.random.seed(0)
    indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
    projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])
    colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey']
    for i in range(0,len(categories)):
        points = projected_features[(labels[indices] == categories[i])]
        plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[i], label=categories[i])
    plt.title("Feature vector for each article, projected on 2 dimensions.",
              fontdict=dict(fontsize=15))
    plt.legend()
    plt.show()
    
# explore feature space
features = bbcFT_dvs.to_numpy() # convert the document vector dataframe to a numpy array
plotTSNE(job_ad['Category'],features) # plot the tSNE to have a look


# In[42]:


# build the classfication model and report results
X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(bbcFT_dvs, job_ad['Category'], list(range(0,len(job_ad))),test_size=0.33, random_state=seed)

model = LogisticRegression(max_iter = 1000,random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


models = [weighted_bbcFT_company,weighted_bbcFT_title,weighted_bbcFT_description]
model_names = ["Weighted Pretrained FastText with Company", "Weighted Pretrained FastText with Title", "Weighted Pretrained FastText with Description"]
# loop through the models and report the results
for i in range(0,len(models)):
    model = models[i]
    model.fit(X_train, y_train)
    print(model_names[i] + " Accuracy: " + str(model.score(X_test, y_test)))


# In[40]:


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

# # Reference
# 
# + https://machinelearningmastery.com/multinomial-logistic-regression-with-python/
# + https://towardsdatascience.com/introduction-to-natural-language-processing-for-text-df845750fb63

# In[ ]:




