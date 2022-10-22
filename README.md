# Natural Language Processing Job Advertisement
Explore a Movie review data set, and pre-process the text review corpus. Use the pre-processed text reviews to generate data features and build a sentiment analysis model to predict sentiment of reviews.

## Project Purpose
+ Pre-process natural language text data to generate effective feature representations;
+ Document and maintain an editable transcript of the data pre-processing pipeline for professional reporting.

## Data Overview
+ A small collection of job advertisement documents (around 776 jobs) inside the `data` folder.
+ Inside the data folder, there are four different sub-folders: Accounting_Finance, Engineering, Healthcare_Nursing, and Sales, representing a job category.
+ The job advertisement text documents of a particular category are in the corresponding sub-folder.
+ Each job advertisement document is a txt file named `Job_<ID>.txt`. It contains the title, the webindex (some will also have information on the company name, some might not), and the full description of the job advertisement.

# Cumulative frequency plot for most common 50 words
![Cumulative frequency plot for most common 50 words](media/figure/cumulative_frequency.png)

# Word frequency distribution 
![Cumulative frequency plot for most common 50 words](media/figure/word_frequency.png)

## Repository Structure

```
├── Document classification with embeddings.ipynb
├── LICENSE
├── README.md
├── Task2_3_2.ipynb
├── Task2_3_2.py
├── bbcNews.txt
├── category.txt
├── count_vectors.txt
├── data
│   ├── Accounting_Finance
│   │   
│   ├── Healthcare_Nursing
│   │  
│   └── Sales
│       
├── description.txt
├── flask
│   ├── flaskjob
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── forms.cpython-310.pyc
│   │   │   └── routes.cpython-310.pyc
│   │   ├── descFT_LR.pkl
│   │   ├── desc_FT.model
│   │   ├── desc_FT.model.wv.vectors_ngrams.npy
│   │   ├── errors
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   └── handlers.cpython-310.pyc
│   │   │   └── handlers.py
│   │   ├── forms.py
│   │   ├── routes.py
│   │   ├── static
│   │   │   ├── css
│   │   │   │   └── style.css
│   │   │   ├── img
│   │   │   │   ├── logo.ico
│   │   │   │   └── logo.png
│   │   │   └── js
│   │   │       └── jquery-3.6.0.js
│   │   └── templates
│   │       ├── Accounting_Finance
│   │       │   ├── Job_00232.html
│   │       │   └── Job_00233.html
│   │       ├── Accounting_Finance.html
│   │       ├── Engineering
│   │       │   ├── Job_00001.html
│   │       │   └── Job_00002.html
│   │       ├── Engineering.html
│   │       ├── Healthcare_Nursing
│   │       │   ├── Job_00423.html
│   │       │   └── Job_00424.html
│   │       ├── Healthcare_Nursing.html
│   │       ├── Sales
│   │       │   ├── Job_00621.html
│   │       │   └── Job_00622.html
│   │       ├── Sales.html
│   │       ├── about.html
│   │       ├── admin.html
│   │       ├── errors
│   │       │   ├── 403.html
│   │       │   ├── 404.html
│   │       │   └── 500.html
│   │       ├── home.html
│   │       ├── job_ad_template.html
│   │       ├── layout.html
│   │       ├── login.html
│   │       ├── register.html
│   │       └── search.html
│   └── run.py
├── jobAd_bVector.txt
├── jobAd_cVector.txt
├── jobAd_tVector.txt
├── job_ad.csv
├── media
│   └── images
│       └── td-idf-graphic.png
├── model.ipynb
├── model.py
├── models
│   ├── FastText
│   │   ├── fast_Text_model
│   │   └── fast_Text_model.wv.vectors_ngrams.npy
│   └── note.txt
├── notebooks
│   ├── task1.ipynb
│   └── task2_3.ipynb
├── requirements.txt
├── saved_txt_files
│   ├── BinaryVectors
│   │   └── jobAd_bVector.txt
│   ├── CountVectors
│   │   └── jobAd_cVector.txt
│   ├── TfidfVectors
│   │   └── jobAd_tVector.txt
│   ├── category.txt
│   ├── count_vectors.txt
│   ├── description.txt
│   ├── title.txt
│   └── vocab.txt
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── __init__.cpython-39.pyc
│   │   ├── utils.cpython-310.pyc
│   │   └── utils.cpython-39.pyc
│   └── utils.py
├── stopwords_en.txt
├── task1.ipynb
├── task1.py
├── task2_3.ipynb
├── task2_3.py
├── title.txt
├── train_model.ipynb
├── train_model.py
├── vocab.txt
└── webindex.txt


```
