#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# <h3 style="color:#ffc0cb;font-size:40px;font-family:Georgia;text-align:center;"><strong>Task 2: Generate Feature Representation <br> Task 3: Classification</strong></h3>
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
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * sklearn
# * collections
# * re
# * numpy
# * nltk
# * itertools
# * pandas
# * os
# 
# ## Steps
# 1. Load data
# 2. Text Pre-processing
#     * Sentence Segmentation
#     * Word Tokenization
#     * Removing Single Character Tokens
#     * Removing Stop words
# 3. Saving the Pre-processing Reviews
# 
# ## Introduction
# You should give a brief information of this assessment task here.
# 
# <span style="color: red"> Note that this is a sample notebook only. You will need to fill in the proper markdown and code blocks. You might also want to make necessary changes to the structure to meet your own needs. Note also that any generic comments written in this notebook are to be removed and replace with your own words.</span>
# 
# ## Dataset
# 
# + This is a small collection of job advertisement documents (around 750 jobs).
# 
# * Consists of 776 documents corresponding to stories in five topical areas during 2004-2005.
# * Class labels: 4 (Accountin)
# 
# The loaded the pre-processed `job_ad.csv` file, with the following attributes:
# 
# | **ATTRIBUTES**   | **DESCRIPTION**                                           |
# |--------------|---------------------------------------------------------------|
# | Webindex     | 8 digit Id of the job advertisement on the website            |
# | Title        | Title of the advertised job position                          |
# | Company      | Company (employer) of the advertised job position             |
# | Description  | the description of each job advertisement                     |
# | Toekenized Description  | the tokenized description of each job advertisement                     |
# | Sentimens  | the sentiment of each job advertisement                     |
# | Catrgory  | 'Accounting_Finance', 'Engineering', 'Healthcare_Nursing', 'Sales'                   |

# ## Importing libraries 

# In[1]:


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
# vocab is a list of unique words in the corpus separated by space
vocab = list(set(chain.from_iterable(tk_description)))
print(tk_description)
len(vocab)


# In[4]:


type(tk_description[0])


# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>Task 2. Generating Feature Representations</strong></h3>

# In[8]:


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


# In[9]:


bow


# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>2.1 Saving outputs</strong></h3>
# 
# Save the count vector representation as per spectification.
# - `count_vectors.txt`
# 
# `count_vectors.txt` stores the sparse count vector representation of job advertisement descriptions in the following format. Each line of this file corresponds to one advertisement. It starts with a ‘#’ key followed by the webindex of the job advertisement, and a comma ‘,’. The rest of the line is the sparse representation of the corresponding description in the form of word_integer_index:word_freq separated by comma. Following is an example of the file format.

# In[10]:


# save count vector representation of job advertisement descriptions
with open('count_vectors.txt', 'w') as f:
    for i, description in enumerate(tk_description):
        f.write('#' + str(webindex[i]) + ',')
        for word in description:
            f.write(str(vocab.index(word)) + ':' + str(bow[i][vocab.index(word)]) + ',')
        f.write('\n')
    print("Successfully write count vector representation of job advertisement descriptions into count_vectors.txt file")


# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>Task 3. Job Advertisement Classification</strong></h3>

# ...... Sections and code blocks on buidling classification models based on different document feature represetations. 
# Detailed comparsions and evaluations on different models to answer each question as per specification. 
# 
# <span style="color: red"> You might have complex notebook structure in this section, please feel free to create your own notebook structure. </span>

# In[ ]:


# Code to perform the task...


# ## Summary
# Give a short summary and anything you would like to talk about the assessment tasks here.

# ## Couple of notes for all code blocks in this notebook
# - please provide proper comment on your code
# - Please re-start and run all cells to make sure codes are runable and include your output in the submission.   
# <span style="color: red"> This markdown block can be removed once the task is completed. </span>
