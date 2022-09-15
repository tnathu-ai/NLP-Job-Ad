#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: XXXX XXXX
# #### Student ID: 000000
# 
# Date: XXXX
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# 
# ## Introduction
# You should give a brief information of this assessment task here.
# 
# <span style="color: red"> Note that this is a sample notebook only. You will need to fill in the proper markdown and code blocks. You might also want to make necessary changes to the structure to meet your own needs. Note also that any generic comments written in this notebook are to be removed and replace with your own words.</span>
# 
# ## Dataset
# 
# + This is a small collection of job advertisement documents (around 750 jobs).

# ## Importing libraries 

# In[42]:


# Code to import libraries as you need in this assessment, e.g.,

from sklearn.datasets import load_files
from collections import Counter


# ### 1.1 Examining and loading data
# 
# Before doing any pre-processing, we need to load the data into a proper format. 
# To load the data, you have to explore the data folder. Inside the `data` folder:
# + Inside the data folder you will see 4 different subfolders, namely: `Accounting_Finance`, `Engineering`,`Healthcare_Nursing`, and `Sales`, each folder name is a job category.
# + The job advertisement text documents of a particular category are located in the corresponding subfolder.
# + Each job advertisement document is a txt file, named as "Job_<ID>.txt". It contains the title, the webindex,(some will also have information on the company name, some might not), and the full description of the job advertisement. 
# 
# In this case, providing that the dataset is given in a very well organised way, I would use a super handy API [`load_files`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html) from `sklearn.datasets`. 
#     
# **import the function by:**
# ```python
# from sklearn.datasets import load_files  
# ```
# 
# Then you can use the function to directly load the data and labels, for example:
# ```python
# df = load_files(r"data")  
# ```
# 
# The loaded `movie_data` is then a dictionary, with the following attributes:
# 
# | **ATTRIBUTES**   | **DESCRIPTION**                                           |
# |--------------|---------------------------------------------------------------|
# | Webindex     | 8 digit Id of the job advertisement on the website            |
# | Title        | Title of the advertised job position                          |
# | Company      | Company (employer) of the advertised job position             |
# | Description  | the description of each job advertisement                     |
# 
# 
# - Examine the data folder, including the categories and job advertisment txt documents, etc. Explain your findings here, e.g., number of folders and format of txt files, etc.
# - Load the data into proper data structures and get it ready for processing.
# - Extract webIndex and description into proper data structures.
# 

# In[43]:


# load each folder and file inside the data folder
df = load_files(r"data")


# In[44]:


# type of the loaded file
type(df)


# In[45]:


# each folder name is a job category corresponding to the df filenames
df["filenames"]


# In[46]:


df['target'] # this means the value 0 is negative, the value 1 is positive.


# In[47]:


df['target_names'] 


# In[48]:


# test whether it matches, just in case
emp = 10 # an example, note we will use this example through out this exercise.
df['filenames'][emp], df['target'][emp] # from the file path we know that it's the correct class too


# In[49]:


# assign variables
description, sentiments = df.data, df.target


# In[50]:


description[emp]


# In[51]:


def decode(l):
    if isinstance(l, list):
        return [decode(x) for x in l]
    else:
        return l.decode('utf-8')

description = decode(description)


# 

# In[52]:


sentiments[emp]


# ### ---------------> OBSERVATION:
# The current `description` contains:
# 
# | **ATTRIBUTES**   | **DESCRIPTION**                                           |
# |--------------|---------------------------------------------------------------|
# | Webindex     | 8 digit Id of the job advertisement on the website            |
# | Title        | Title of the advertised job position                          |
# | Company      | Company (employer) of the advertised job position             |
# | Description  | the description of each job advertisement                     |
# 
# and I only want the description itself to perform text-preprocessing and NLP on it. Therefore, I will perform the following pre-processing steps to the description of each job advertisement;

# stop_words

# ...... Sections and code blocks on basic text pre-processing
# 
# 
# <span style="color: red"> You might have complex notebook structure in this section, please feel free to create your own notebook structure. </span>

# In[53]:


type(description)


# In[54]:


# code to perform the task...
# Extract information from each job advertisement. Perform the following pre-processing steps to the description of each job advertisement
def extract_description(description):
    import re
    description = [re.search(r'\nDescription: (.*)', str(i)).group(1) for i in description]
    return description

extract_description(description)


# In[55]:


description = extract_description(description)
description


# In[56]:


description[emp]


# In[57]:


from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain


def tokenizeDescription(raw_description):
    """
        This function first convert all words to lowercases,
        it then segment the raw description into sentences and tokenize each sentences
        and convert the description to a list of tokens.
    """
    # description = raw_description.decode('utf-8') # convert the bytes-like object to python string, need this before we apply any pattern search on it
    description = raw_description.lower() # cover all words to lowercase

    # segment into sentences
    sentences = sent_tokenize(description)

    # tokenize each sentence
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern)
    token_lists = [tokenizer.tokenize(sen) for sen in sentences]

    # merge them into a list of tokens
    tokenised_description = list(chain.from_iterable(token_lists))
    return tokenised_description


# In[58]:


import numpy as np
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


# In[59]:


tk_description = [tokenizeDescription(r) for r in description]  # list comprehension, generate a list of tokenized articles

print("Raw description:\n",description[emp],'\n')
print("Tokenized description:\n",tk_description[emp])


# In[60]:


stats_print(tk_description)


# ### Task 2.2 Remove words with length less than 2.
# 
# In this sub-task, you are required to remove any token that only contains a single character (a token that of length 1).
# You need to double-check whether it has been done properly

# In[61]:


words = list(chain.from_iterable(tk_description)) # we put all the tokens in the corpus in a single list
word_counts = Counter(words) # count the number of times each word appears in the corpus
print("Number of words that appear only once:", len([w for w in word_counts if word_counts[w] == 1]))


# In[62]:


st_list = [[w for w in description if len(w) <= 1 ]                       for description in tk_description] # create a list of single character token for each description
list(chain.from_iterable(st_list)) # merge them together in one list

# filter out single character tokens
tk_description = [[w for w in description if len(w) >=2]                       for description in tk_description]


# In[63]:


# Remove the top 50 most frequent words

words = list(chain.from_iterable(tk_description)) # we put all the tokens in the corpus in a single list
word_counts = Counter(words) # count the number of times each word appears in the corpus
top50 = word_counts.most_common(50) # get the top 50 most frequent words
print("Top 50 most frequent words:\n",top50)

tk_description = [[w for w in description if w not in top50] for description in tk_description]


# In[64]:


print("Tokenized description:\n",tk_description[emp])


# ### Task 2.3 Removing Stop words
# 
# In this sub-task, you are required to remove the stop words from the tokenized text inside `stopwords_en.txt` file

# In[65]:


# remove the stop words inside `stopwords_en.txt` from the tokenized text
with open('stopwords_en.txt', 'r') as f:
    stop_words = f.read().splitlines() # read the stop words into a list
print("Stop words:\n",stop_words)


# In[66]:


[w for w in stop_words if ("not" in w or "n't" in w or "no" in w)]


# In[67]:


# specify
ignored_words = [w for w in stop_words if not ("not" in w or "n't" in w or "no" in w)]

# filter out stop words
tk_description = [[w for w in description if w not in ignored_words]                       for description in tk_description]

stats_print(tk_description)


# Saving required outputs
# Save the vocabulary, bigrams and job advertisment txt as per spectification.
# - vocab.txt

# In[73]:


def save_description(descriptionFilename,tk_description):
    out_file = open(descriptionFilename, 'w') # creates a txt file and open to save the descriptions
    string = "\n".join([" ".join(description) for description in tk_description])
    out_file.write(string)
    out_file.close() # close the file

def save_sentiments(sentimentFilename,sentiments):
    out_file = open(sentimentFilename, 'w') # creates a txt file and open to save sentiments
    string = "\n".join([str(s) for s in sentiments])
    out_file.write(string)
    out_file.close() # close the file


# In[74]:


save_description('description.txt',tk_description)


# In[75]:


save_sentiments('sentiments.txt',sentiments)


# In[76]:


print(df.data[emp]) # an example of a sentiment txt
print(tk_description[emp]) # an example of the pre-process sentiment text
all(df.target==sentiments) # validate whether we save the sentiment properly


# In[77]:


# code to save output data...
# Save all job advertisement text and information in txt file
with open('job_ad.txt', 'w') as f:
    for i in range(len(tk_description)):
        # f.write("Title: " + title[i] + "\n")
        # f.write("Company: " + company[i] + "\n")
        # f.write("Location: " + location[i] + "\n")
        # f.write("Salary: " + salary[i] + "\n")
        f.write("Raw Description: " + description[i] + "\n")
        f.write("Tokenized Description: " + str(tk_description[i]) + "\n")
        f.write("\n")
        print("Successfully write job advertisement " + str(i) + " in txt file")


# ## Summary
# Give a short summary and anything you would like to talk about the assessment task here.

# ## Couple of notes for all code blocks in this notebook
# - please provide proper comment on your code
# - Please re-start and run all cells to make sure codes are runable and include your output in the submission.   
# <span style="color: red"> This markdown block can be removed once the task is completed. </span>

# In[69]:


# The .py format of the jupyter notebook
import os

for fname in os.listdir():
    if fname.endswith('ipynb'):
        os.system(f'jupyter nbconvert {fname} --to python')


# In[69]:




