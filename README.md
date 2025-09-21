# Citation Prediction — NLP

Predict whether one research paper cites another using text + graph features on a citation network.  
**Course:** NLP 053 (CSE, University of Ioannina) · **Challenge:** Private Kaggle competition run by Konstantinos Skianis (Feb 24 → Jun 10, 2025) · **Metric:** Log Loss.

## Problem
Given a pair of papers `(u, v)`, predict the probability that `u` cites `v` (link prediction on a citation graph). We combine:
- **Content similarity** from abstracts,
- **Authorship relatedness**, and
- **Graph structure** (neighbors, GoW, embeddings).

## Data layout
Place the four competition files under `data_new/`:
abstracts.txt # "paper_id |--| abstract"
authors.txt # "paper_id |--| author1;author2;..."
edgelist.txt # CSV: source,target (existing citations)
test.txt # CSV: pairs without labels

## Features
- **TF–IDF cosine** of abstracts.  
- **Authors Jaccard** overlap.  
- **Common Neighbors** in the citation graph.  
- **Graph-of-Words (GoW)** per abstract: PageRank **sum** and mean **clustering coefficient** from a windowed word-cooccurrence graph.  
- **Node2Vec cosine** between paper embeddings.  

## Models
- **Random Forest** (best validation): ~200 trees, `max_depth=20`, `min_samples_leaf=2`, class weights.  
- **Logistic Regression** with `liblinear/saga`, `class_weight='balanced'`, grid‐search on `C`, scored by **neg_log_loss**.  
- **MLP** (PyTorch): 2 hidden layers (128→64), ReLU, BatchNorm, Adam, early stopping.  
Training uses stratified 80/20 split and k-fold CV. 
RF + enriched features performed best in our experiments.




