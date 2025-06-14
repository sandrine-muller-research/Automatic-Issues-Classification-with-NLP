import json 
import os
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
from tqdm import tqdm
import umap.umap_ as umap
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.cluster import KMeans






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
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Example usage
    owner = 'NCATSTranslator'
    repo = 'Feedback'
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"

    if not os.path.exists('Translator_issues.json'):
        all_issues = get_all_issues(url)
        with open('Translator_issues.json', 'w') as json_file:
            json.dump(all_issues, json_file)
    else:
        with open('Translator_issues.json', 'r') as json_file:
            all_issues = json.load(json_file)
    print(f"Total issues: {len(all_issues)}")


    # merge title and body and clean:
    issues_body = []
    for issue in all_issues:
        if "title" in issue:
            if issue["title"] is not None:
                if "body" in issue:
                    if issue["body"] is not None:
                        issues_body.append([clean_text(issue["title"])+clean_text(issue["body"])])
                    else:
                        issues_body.append([clean_text(issue["title"])])
                else:
                    issues_body.append([clean_text(issue["title"])])
            else:
                if "body" in issue:
                    if issue["body"] is not None:
                        issues_body.append([clean_text(issue["body"])])
        else:
            if "body" in issue:
                issues_body.append([clean_text(issue["body"])])

    nlp = spacy.load('en_core_web_sm')
    stopwords = nlp.Defaults.stop_words
    issues_body2 = [lemmatization(issue,stopwords) for issue in issues_body]
    issues_body3 = [b for a in issues_body2 for b in a]

    # similarity analysis:
    issues_embeddings = model.encode(issues_body3)
    umap_model = umap.UMAP(random_state=42)
    reduced_embeddings = umap_model.fit_transform(issues_embeddings)
    
    # Cluster the reduced embeddings
    n_clusters = 50  # or any number you choose
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    reduced_embeddings_clusters = kmeans.fit_predict(reduced_embeddings)
    
    cluster_keywords = []
    for cluster_id in range(n_clusters):
        cluster_texts = [text for text, label in zip(issues_body3, reduced_embeddings_clusters) if label == cluster_id]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(cluster_texts)
        # Mean TF-IDF score per term
        scores = np.asarray(X.mean(axis=0)).ravel()
        top_indices = scores.argsort()[-5:][::-1]
        cluster_keywords.append([vectorizer.get_feature_names_out()[i] for i in top_indices])
        # print(f"Cluster {cluster_id}: {keywords}")
    
    print("bob")