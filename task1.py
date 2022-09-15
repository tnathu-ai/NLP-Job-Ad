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

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,

from sklearn.datasets import load_files   


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


# In[4]:


# type of the loaded file
type(df)


# In[8]:


# each folder name is a job category corresponding to the df filenames
df["filenames"]


# In[10]:


df['target'] # this means the value 0 is negative, the value 1 is positive.


# In[11]:


df['target_names'] 


# In[12]:


# test whether it matches, just in case
emp = 10 # an example, note we will use this example through out this exercise.
df['filenames'][emp], df['target'][emp] # from the file path we know that it's the correct class too


# In[16]:


description, sentiments = df.data, df.target  


# In[17]:


description[emp]


# In[18]:


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

# ### 1.2 Pre-processing data
# Perform the required text pre-processing steps.

# ...... Sections and code blocks on basic text pre-processing
# 
# 
# <span style="color: red"> You might have complex notebook structure in this section, please feel free to create your own notebook structure. </span>

# In[ ]:


# code to perform the task...


# ## Saving required outputs
# Save the vocabulary, bigrams and job advertisment txt as per spectification.
# - vocab.txt

# In[6]:


# code to save output data...


# ## Summary
# Give a short summary and anything you would like to talk about the assessment task here.

# ## Couple of notes for all code blocks in this notebook
# - please provide proper comment on your code
# - Please re-start and run all cells to make sure codes are runable and include your output in the submission.   
# <span style="color: red"> This markdown block can be removed once the task is completed. </span>

# In[ ]:





# In[19]:


jupyter nbconvert --to script "task1.ipynb"


# In[20]:


ipython nbconvert task1.ipynb --to script


# In[ ]:




