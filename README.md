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


## Repository Structure

```
├── LICENSE
|
├── README.md          <- The top-level README for developers using this project
|
├── data
│   ├── Accounting_Finance      
│   ├── Engineering                 
│   ├── Healthcare_Nursing          
│   └── Sales   
|
|
├── saved_txt_files
│   ├── category.txt      
│   ├── count_vectors.txt                 
│   ├── description.txt          
│   └── vocab.txt   
| 
│
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering)
│   ├── Accounting_Finance          <- Data from third party sources.
│   ├── Engineering                 <- Intermediate data that has been transformed.
│   ├── Healthcare_Nursing          <- The final, canonical data sets for modeling.
│   └── Sales                       <- The original, immutable data dump.
|
│                         
│                         
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
|
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
|
├── src                <- Python source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── src           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── notebooks       <- Scripts to turn raw data into features for modeling
│   │   └── task1.py
|
|
|
|
│── .gitignore                <- plain text file contains files/directories to ignore

```
