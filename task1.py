#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# 
# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>Task 1. Basic Text Pre-processing</strong></h3>
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
# Nowadays there are many job hunting websites including seek.com.au and au.indeed.com. These job hunting sites all manage a job search system, where job hunters could search for relevant jobs based on keywords, salary, and categories. In previous years, the category of an advertised job was often manually entered by the advertiser (e.g., the employer). There were mistakes made for category assignment. As a result, the jobs in the wrong class did not get enough exposure to relevant candidate groups.
# With advances in text analysis, automated job classification has become feasible; and sensible suggestions for job categories can then be made to potential advertisers. This can help reduce human data entry error, increase the job exposure to relevant candidates, and also improve the user experience of the job hunting site. In order to do so, we need an automated job ads classification system that helps to predict the categories of newly entered job advertisements.
# 
# ## Dataset
# 
# + This is a small collection of job advertisement documents (around 750 jobs).

# ## Importing libraries 

# In[1]:


# import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from collections import Counter
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
import re
import os


# In[2]:


# check the version of the main packages
print("Numpy version: ", np.__version__)
print("Pandas version: ",pd.__version__)
get_ipython().system(' python --version')


# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>1.1 Examining and loading data</strong></h3>
# 
# - Examine the data folder, including the categories and job advertisment txt documents, etc. Explain your findings here, e.g., number of folders and format of txt files, etc.
# - Load the data into proper data structures and get it ready for processing.
# - Extract webIndex and description into proper data structures.
# 
# 
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

# In[3]:


# load each folder and file inside the data folder
df = load_files(r"data")
# type of the loaded file
type(df)


# In[4]:


df['target'] # this corresponding to the index value of the 4 categories


# In[5]:


# Name of the categories
df['target_names'] # this corresponding to the name value of the 4 categories


# In[6]:


print(f'Category at index 0: {df["target_names"][0]}')
print(f'Category at index 1: {df["target_names"][1]}')
print(f'Category at index 2: {df["target_names"][2]}')
print(f'Category at index 3: {df["target_names"][3]}')


# In[7]:


# test whether it matches, just in case
emp = 10 # an example, note we will use this example throughout this exercise.
df['filenames'][emp], df['target'][emp] # from the file path we know that it's the correct class too


# In[8]:


# assign variables
full_description, category = df.data, df.target


# In[9]:


# the 10th job advertisement description
full_description[emp]


# ### ------> OBSERVATION:
# As we can see the current description is in the **binary** form. Therefore, we need to decode into normal string for further pre-processing

# In[10]:


def decode(l):
    if isinstance(l, list):
        return [decode(x) for x in l]
    else:
        return l.decode('utf-8')

full_description = decode(full_description)


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

# <h3 style="color:#ffc0cb;font-size:50px;font-family:Georgia;text-align:center;"><strong>1.2 Pre-processing data</strong></h3>

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


# #### A Few Statistics Before Any Further Pre-processing
# 
# In the following, we are interested to know a few statistics at this very begining stage, including:
# * The total number of tokens across the corpus
# * The total number of types across the corpus, i.e. the size of vocabulary 
# * The so-called, [lexical diversity](https://en.wikipedia.org/wiki/Lexical_diversity), referring to the ratio of different unique word stems (types) to the total number of words (tokens).  
# * The average, minimum and maximum number of token (i.e. document length) in the dataset.
# 
# In the following, we wrap all these up as a function, since we will use this printing module later to compare these statistic values before and after pre-processing.

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


# #### The Updated Statistics
# 
# In the above, we have done a few pre-processed steps, now let's have a look at the statistics again:
# 

# In[24]:


# specify
ignored_words = [w for w in stop_words if not ("not" in w or "n't" in w or "no" in w)]

# filter out stop words
tk_description = [[w for w in description if w not in ignored_words]                       for description in tk_description]

stats_print(tk_description)


# Recall, from the beginning, we have the following:  
# _____________________________________________
# 
# Vocabulary size:  9423
# 
# Total number of tokens:  107751
# 
# Lexical diversity:  0.08745162457889022
# 
# Total number of description: 776
# 
# Average description length: 138.85438144329896
# 
# Maximun description length: 489
# 
# Minimun description length: 12
# 
# Standard deviation of description length: 73.42099464751045
# _____________________________________________
# 
# We've shrunk more than 40% of the vocabulary.

# category

# Saving required outputs
# Save the vocabulary, bigrams and job advertisment txt as per spectification.
# - vocab.txt

# In[25]:


def save_description(descriptionFilename,tk_description):
    out_file = open(descriptionFilename, 'w') # creates a txt file and open to save the descriptions
    string = "\n".join([" ".join(description) for description in tk_description])
    out_file.write(string)
    out_file.close() # close the file

def save_sentiments(sentimentFilename,category):
    out_file = open(sentimentFilename, 'w') # creates a txt file and open to save category
    string = "\n".join([str(s) for s in category])
    out_file.write(string)
    out_file.close() # close the file


# save description into txt file
save_description('description.txt',tk_description)
# save Category into txt file
save_sentiments('category.txt',category)


# In[ ]:


print(df.data[emp]) # an example of a Category txt
print(tk_description[emp]) # an example of the pre-process Category text


# In[ ]:


# code to save output data...
# Save all job advertisement text and information in txt file
with open('job_ad.txt', 'w') as f:
    f.write("Category: " + str(category) + "\n")
    for i in range(len(tk_description)):
        f.write(full_description[i] + "\n")
        f.write("Tokenized Description: " + str(tk_description[i]) + "\n")
        f.write("Category: " + str(df['target'][i]) + "\n")
        f.write("\n")
    print("Successfully write job advertisement with the tokenized description in txt file")


# In[ ]:


def write_vocab(vocab, filename):
    with open(filename, 'w') as f:
        for i, word in enumerate(vocab):
            f.write(word + ':' + str(i) + '\n')
# convert tokenized description into a alphabetically sorted list
vocab = sorted(list(set(chain.from_iterable(tk_description))))
write_vocab(vocab, 'vocab.txt')
# print out the first 10 words in the vocabulary
print(vocab[:10])


# In[ ]:


category


# In[ ]:


# convert job ad to a dataframe
job_ad = pd.DataFrame({'Title': title, 'Webindex': webindex, 'Company': company, 'Description': description,'Tokenized Description': tk_description, 'Category': category})

# change Tokenized Description to string separated by space
job_ad['Tokenized Description'] = job_ad['Tokenized Description'].apply(lambda x: ' '.join([str(i) for i in x]))

# rename the value of category column
# Category at index 0: Accounting_Finance
# Category at index 1: Engineering
# Category at index 2: Healthcare_Nursing
# Category at index 3: Sales
df['Category'] = df['Category'].replace([0,1,2,3],['Accounting_Finance','Engineering','Healthcare_Nursing','Sales'])

# save job ad to csv file
job_ad.to_csv('job_ad.csv', index=False)

print(job_ad.info())
# print first 3 rows
job_ad.head(3)


# In[ ]:


# The .py format of the jupyter notebook
for fname in os.listdir():
    if fname.endswith('ipynb'):
        os.system(f'jupyter nbconvert {fname} --to python')


# ## Summary
# Give a short summary and anything you would like to talk about the assessment task here.

# In[ ]:




