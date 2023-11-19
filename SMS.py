#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords


# In[42]:


import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']
csv_file_path = 'spam.csv'

csv_file_encoding = detect_encoding(csv_file_path)

df = pd.read_csv(csv_file_path, encoding=csv_file_encoding)


# In[43]:


csv_file_path = 'spam.csv'

csv_file_encoding = 'latin-1'

df = pd.read_csv(csv_file_path, encoding=csv_file_encoding)


# In[44]:


df.head()


# In[45]:


print(df.columns)

try:
    df = df[['v2', 'v1']]
except KeyError as e:
    print(f"")
df = df.rename(columns={'v2': 'messages', 'v1': 'label'})
print(df.head())


# In[46]:


df.isnull().sum()


# In[56]:


import re
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    
    text = text.lower()
    
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text


# In[57]:


df['clean_text'] = df['messages'].apply(clean_text)
df.head()


# In[58]:


df['clean_text'] = df['messages'].apply(clean_text)
X = df['clean_text']
y = df['label']


# In[59]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

def classify(model, X, y):
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)
    
    pipeline_model = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('clf', model)])
    pipeline_model.fit(x_train, y_train)
    
    print('Accuracy:', pipeline_model.score(x_test, y_test)*100)
    y_pred = pipeline_model.predict(x_test)
    print(classification_report(y_test, y_pred))


# In[60]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, y)


# In[61]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
classify(model, X, y)


# In[62]:


from sklearn.svm import SVC
model = SVC(C=3)
classify(model, X, y)


# In[63]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model, X, y)


# In[ ]:




