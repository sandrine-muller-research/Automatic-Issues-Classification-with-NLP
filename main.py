import json 
# import numpy as np
# import pandas as pd
# import re, nltk, spacy, string
# # spacy.require_gpu()
# import en_core_web_sm
# nlp = en_core_web_sm.load()
# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline
# import warnings
# warnings.filterwarnings('ignore')

# from plotly.offline import plot
# import plotly.graph_objects as go
# import plotly.express as px

# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
# from pprint import pprint
# from tqdm import tqdm, tqdm_notebook
# tqdm.pandas()

import requests
import re
import spacy
from tqdm import tqdm, tqdm_notebook


def clean_text(text):
  if text is not None:
    text=text.lower()  #convert to lower case
    text=re.sub(r'^\[[\w\s]\]+$',' ',text) #Remove text in square brackets
    text=re.sub(r'[^\w\s]',' ',text) #Remove punctuation
    text=re.sub(r'^[a-zA-Z]\d+\w*$',' ',text) #Remove words with numbers
    return text


def lemmatization(texts,stopwords):
    lemma_sentences = []
    for doc in tqdm(nlp.pipe(texts)):
        sent = [token.lemma_ for token in doc if token.text not in set(stopwords)]
        lemma_sentences.append(' '.join(sent))
    return lemma_sentences

def get_all_issues(url, state='all'):
    issues = []
    page = 1
    per_page = 100
    
    while True:
        params = {
            'state': state,
            'page': page,
            'per_page': per_page
        }
        headers = {
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28'
        }
        
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            new_issues = response.json()
            if not new_issues:
                break
            issues.extend(new_issues)
            page += 1
        else:
            print(f"Error: {response.status_code}")
            break
    
    return issues

# def load_issues()

if __name__ == "__main__":
    # Example usage
    owner = 'NCATSTranslator'
    repo = 'Feedback'
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"

    all_issues = get_all_issues(url)
    with open('Translator_issues', 'w') as json_file:
        json.dump(all_issues, json_file)
    print(f"Total issues: {len(all_issues)}")

    # merge title and body and clean:
    issues_body = []
    for issue in all_issues:
        if "title" in issue:
            if issue["title"] is not None:
                if "body" in issue:
                    if issue["body"] is not None:
                        issues_body.append(clean_text(issue["body"])+clean_text(issue["title"]))
                    else:
                        issues_body.append(clean_text(issue["title"]))
                else:
                    issues_body.append(clean_text(issue["title"]))
            else:
                if "body" in issue:
                    if issue["body"] is not None:
                        issues_body.append(clean_text(issue["body"]))
        else:
            if "body" in issue:
                issues_body.append(clean_text(issue["body"]))

    nlp = spacy.load('en_core_web_sm')
    stopwords = nlp.Defaults.stop_words
    issues_body = [lemmatization(issue,stopwords) for issue in issues_body]


    print("bob")