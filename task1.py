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
# * sklearn
# * collections
# * re
# * numpy
# * nltk
# * itertools
# * pandas
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

# In[1]:


# import libraries
import numpy as np
from sklearn.datasets import load_files
from collections import Counter
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
import re


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

# In[2]:


# load each folder and file inside the data folder
df = load_files(r"data")
# type of the loaded file
type(df)


# In[3]:


# each folder name is a job category corresponding to the df filenames
df["filenames"]


# In[4]:


df['target'] # this means the value 0 is negative, the value 1 is positive.


# In[5]:


# Name of the categories
df['target_names']


# In[6]:


# test whether it matches, just in case
emp = 10 # an example, note we will use this example throughout this exercise.
df['filenames'][emp], df['target'][emp] # from the file path we know that it's the correct class too


# In[7]:


# assign variables
full_description, sentiments = df.data, df.target


# In[8]:


# the 10th job advertisement description
full_description[emp]


# ### ------> OBSERVATION:
# As we can see the current description is in the **binary** form. Therefore, we need to decode into normal string for further pre-processing

# In[9]:


def decode(l):
    if isinstance(l, list):
        return [decode(x) for x in l]
    else:
        return l.decode('utf-8')

full_description = decode(full_description)


# 

# In[10]:


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

# # Text pre-processing

# In[11]:


type(full_description)


# In[12]:


# Extract information from each job advertisement. Perform the following pre-processing steps to the description of each job advertisement
def extract_description(full_description):
    description = [re.search(r'\nDescription: (.*)', str(i)).group(1) for i in full_description]
    return description


description = extract_description(full_description)
extract_description(full_description)


# In[13]:


# Extract title
def extract_title(full_description):
    title = [re.search(r'Title: (.*)', str(i)).group(1) for i in full_description]
    return title


title = extract_title(full_description)
extract_title(full_description)


# In[14]:


# Extract Webindex
def extract_webindex(full_description):
    webindex = [re.search(r'Webindex: (.*)', str(i)).group(1) for i in full_description]
    return webindex

webindex = extract_webindex(full_description)
extract_webindex(full_description)


# In[15]:


def extract_company(company):
    company = [re.search(r'Company: (.*)', str(i)).group(1) if re.search(r'Company: (.*)', str(i)) else "NA" for i in company]
    return company
company = extract_company(full_description)
extract_company(full_description)


# In[16]:


description[emp]


# In[17]:


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

tk_description = [tokenizeDescription(r) for r in description]  # list comprehension, generate a list of tokenized articles

print("Raw description:\n",description[emp],'\n')
print("Tokenized description:\n",tk_description[emp])


# In[18]:


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

stats_print(tk_description)


# ### Task 2.2 Remove words with length less than 2.
# 
# In this sub-task, you are required to remove any token that only contains a single character (a token that of length 1).
# You need to double-check whether it has been done properly

# In[19]:


words = list(chain.from_iterable(tk_description)) # we put all the tokens in the corpus in a single list
word_counts = Counter(words) # count the number of times each word appears in the corpus
print("Number of words that appear only once:", len([w for w in word_counts if word_counts[w] == 1]))


# In[20]:


st_list = [[w for w in description if len(w) <= 1 ]                       for description in tk_description] # create a list of single character token for each description
list(chain.from_iterable(st_list)) # merge them together in one list

# filter out single character tokens
tk_description = [[w for w in description if len(w) >=2]                       for description in tk_description]


# In[21]:


# Remove the top 50 most frequent words
words = list(chain.from_iterable(tk_description)) # we put all the tokens in the corpus in a single list
word_counts = Counter(words) # count the number of times each word appears in the corpus
top50 = word_counts.most_common(50) # get the top 50 most frequent words
print("Top 50 most frequent words:\n",top50)

tk_description = [[w for w in description if w not in top50] for description in tk_description]


# ### Task 2.3 Removing Stop words
# 
# In this sub-task, you are required to remove the stop words from the tokenized text inside `stopwords_en.txt` file

# In[22]:


# remove the stop words inside `stopwords_en.txt` from the tokenized text
with open('stopwords_en.txt', 'r') as f:
    stop_words = f.read().splitlines() # read the stop words into a list
print("Stop words:\n",stop_words)


# In[23]:


[w for w in stop_words if ("not" in w or "n't" in w or "no" in w)]


# In[24]:


# specify
ignored_words = [w for w in stop_words if not ("not" in w or "n't" in w or "no" in w)]

# filter out stop words
tk_description = [[w for w in description if w not in ignored_words]                       for description in tk_description]

stats_print(tk_description)


# Saving required outputs
# Save the vocabulary, bigrams and job advertisment txt as per spectification.
# - vocab.txt

# In[25]:


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


# In[26]:


save_description('description.txt',tk_description)


# In[27]:


save_sentiments('sentiments.txt',sentiments)


# In[28]:


print(df.data[emp]) # an example of a sentiment txt
print(tk_description[emp]) # an example of the pre-process sentiment text
all(df.target==sentiments) # validate whether we save the sentiment properly


# In[29]:


# code to save output data...
# Save all job advertisement text and information in txt file
with open('job_ad.txt', 'w') as f:
    for i in range(len(tk_description)):
        f.write(full_description[i] + "\n")
        f.write("Tokenized Description: " + str(tk_description[i]) + "\n")
        f.write("Sentiment: " + str(sentiments[i]) + "\n")
        f.write("\n")
    print("Successfully write job advertisement with the tokenized description in txt file")


# In[30]:


def write_vocab(vocab, filename):
    with open(filename, 'w') as f:
        for i, word in enumerate(vocab):
            f.write(word + ':' + str(i) + '\n')
# convert tokenized description into a alphabetically sorted list
vocab = sorted(list(set(chain.from_iterable(tk_description))))
write_vocab(vocab, 'vocab.txt')
# print out the first 10 words in the vocabulary
print(vocab[:10])


# In[37]:


import pandas as pd

# convert job ad to a dataframe
job_ad = pd.DataFrame({'Title': title, 'Webindex': webindex, 'Company': company, 'Description': description,'Tokenized Description': tk_description, 'sentiment': sentiments})
# print first 3 rows
job_ad.head(3)
# save job ad to csv file
job_ad.to_csv('job_ad.csv', index=False)



# In[35]:


# The .py format of the jupyter notebook
import os

for fname in os.listdir():
    if fname.endswith('ipynb'):
        os.system(f'jupyter nbconvert {fname} --to python')


# ## Summary
# Give a short summary and anything you would like to talk about the assessment task here.

# In[ ]:
# Read job_ad.csv
job_ad = pd.read_csv('job_ad.csv')
# print first 3 rows
job_ad.head(3)
# get the description of the job ad
job_ad['Description'].describe()
# get the tokenized description of the job ad
job_ad['Tokenized Description'].describe()

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

"""
count_vectors.txt This file stores the sparse count vector representation of job advertisement
descriptions in the following format. Each line of this file corresponds to one advertisement. It starts
with a ‘#’ key followed by the webindex of the job advertisement, and a comma ‘,’. The rest of the line
is the sparse representation of the corresponding description in the form of
word_integer_index:word_freq separated by comma. Following is an example of the file format (note
that the following image is artificial and used to demonstrate the required format only, it doesn't
reflect the values of the actual expected output)
"""
# save count vector representation of job advertisement descriptions
with open('count_vectors.txt', 'w') as f:
    for i, description in enumerate(tk_description):
        f.write('#' + str(webindex[i]) + ',')
        for word in description:
            f.write(str(vocab.index(word)) + ':' + str(bow[i][vocab.index(word)]) + ',')
        f.write('\n')
    print("Successfully write count vector representation of job advertisement descriptions in txt file")
