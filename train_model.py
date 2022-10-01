#!/usr/bin/env python
# coding: utf-8

# # Fun with Word Embeddings
# 
# In this activity, we will explore a few pre-trained word embeddings, and also will learn to train Word2Vec and FastText word embeddings. 
# 
# We will use Gensim implementation of Word2Vec. 
# [Gensim](https://radimrehurek.com/gensim/) is an open source Python library for natural language processing, with a focus on topic modeling. 
# To prepare for this activity, you need to install Gensim. 
# You can do this by going to your terminal, and run the following command:
# ```
# pip install --upgrade gensim
# ```

# In[1]:


get_ipython().system('pip install --upgrade gensim  # lazy installation :D')


# In this notebook, there will be lots of things happening behind the scene &#128556; 
# We can track events and display information through basic [Logging](https://docs.python.org/3/howto/logging.html).
# 
# In the following, we also specify the format that we want the information to be displayed by specifying the formatting string `'%(asctime)s : %(levelname)s : %(message)s'`. 
# 
# For a full set of things that can appear in format strings, you can refer to the documentation for [LogRecord](https://docs.python.org/3/library/logging.html#logrecord-attributes) attributes, but for simple usage, you just need the `levelname` (severity), `message` (event description, including variable data) and perhaps to display when the event occurred with `asctime`. 

# In[2]:


import logging
from pprint import pprint as print

# Tracking events and display information through Logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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

# In[3]:


from gensim.models.fasttext import FastText

# 1. Set the corpus file names/path
corpus_file = './bbcNews.txt'

# 2. Initialise the Fast Text model
bbcFT = FastText(vector_size=100) 

# 3. build the vocabulary
bbcFT.build_vocab(corpus_file=corpus_file)

# 4. train the model
bbcFT.train(
    corpus_file=corpus_file, epochs=bbcFT.epochs,
    total_examples=bbcFT.corpus_count, total_words=bbcFT.corpus_total_words,
)

print(bbcFT)


# We can retrieve the KeyedVectors from the model as follows,

# In[4]:


bbcFT_wv = bbcFT.wv
print(bbcFT_wv)


# And see the vector of `king`:

# In[5]:


bbcFT_wv['king']


# ### Save the model
# 
# Similar as we do with the trained Word2Vec, we can also save our trained FastText model using the standard gensim methods. 
# 

# In[6]:


# Save the model
bbcFT.save("bbcFT.model")


# ## 4. Summary
# 
# In this activity, we have played with a few pretrained word embeddings including `Word2Vec` trained on the Google news dataset, and GloVe. 
# We have also learnt to trained our `Word2Vec` and `FastText` models using our BBC News dataset. 
# I hope you have a lot of fun in this activity. 
# 
# The semantic embeddings seem amazing, though so far, we have not yet really explore the actual usage of them. 
# In the next activity, we will try to use this word embeddings for text classification task. Get ready! 😉

# ## 5. Exercise
# * There are multiple pre-trained models in Gensim, see Section **Pretrained models** in https://radimrehurek.com/gensim/models/word2vec.html. Indeed, Gensim also includes the GloVe implementation. In this activity, we didn't use this Gensim implementation, instead, we had demonstated how to load the pre-trained GloVe word embeddings from the original source. You can explore other pretrained models in Gensim.

# ## Reference:
# [1] [Word Embeddings— Fun with Word2Vec and Game of Thrones](https://medium.com/@khulasaandh/word-embeddings-fun-with-word2vec-and-game-of-thrones-ea4c24fcf1b8)  
# [2] [Gensim Word2Vec Tutorial – Full Working Example](http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.YJzBmmYza3f)  
# [3] [Word2vec Tutorial](https://rare-technologies.com/word2vec-tutorial/)  
# [4] [Word2Vec Model -- gensim](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)   
# [5] [Using pre-trained word embeddings](https://nlp.stanford.edu/projects/glove/) 
# [6] [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
# [7] [Using pre-trained word embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings/). 
