#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd


# In[16]:


import re
import json
from glob import glob
import os
from io import StringIO
from itertools import groupby
import pickle

import numpy as np
import bs4
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[17]:


cleancoffee_df = pd.read_csv("data/archive/scraped-categories-2.csv")


# In[18]:


reviews = cleancoffee_df['review'].values.tolist()


# In[19]:


def tokenize(text):
    return re.findall('[a-z]+', text.lower())


# In[20]:


def tokenize_reviews(tokenize_method,input_review):
    final = []
    for i in range(0, len(input_review)):
        text = input_review[i]
        final = final + tokenize(text)
    return final


# In[21]:


review_tokens = tokenize_reviews(tokenize, reviews)


# In[23]:


review_tokens_distinct = []
for i in range(0, len(review_tokens)):
    if review_tokens[i] not in review_tokens_distinct:
        review_tokens_distinct.append(review_tokens[i])


# In[24]:


tokens_count = {}
for i in range(0, len(review_tokens)):
    if review_tokens[i] in tokens_count:
        tokens_count[review_tokens[i]]+=1
    else:
        tokens_count[review_tokens[i]]=1


# In[25]:


keys = list(tokens_count.keys())
for key in keys:
    tokens_count[key] = tokens_count[key]/len(review_tokens)
    


# In[26]:


tokens_count = dict(sorted(tokens_count.items(), key=lambda item: item[1], reverse = True))


# In[28]:


tokens_count


# In[ ]:




