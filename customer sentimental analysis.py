#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


# In[6]:


nltk.download('stopwords')
nltk.download('punkt')


# In[7]:


stop_words = set(stopwords.words('english'))


# In[8]:


def preprocess_text(text):
    try:
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word not in string.punctuation]
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [word for word in tokens if word.isalpha()]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    except TypeError:
        return ''


# In[9]:


data = pd.read_csv("human.csv", encoding='latin1')


# In[10]:


data.head()


# In[11]:


data['preprocessed_sentence'] = data['Column1'].apply(preprocess_text)


# In[12]:


print(data.head())


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[14]:


data.dropna(subset=['Column1'], inplace=True)


# In[15]:


data.reset_index(drop=True, inplace=True)


# In[16]:


X = data['Column1']  
y = data['Column2']  


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


# In[18]:


tfidf_vectorizer = TfidfVectorizer(max_features=1000)  


# In[19]:


X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)


# In[20]:


X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[21]:


model = LogisticRegression()
model.fit(X_train_tfidf, y_train)


# In[22]:


predictions = model.predict(X_test_tfidf)


# In[23]:


accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


# In[24]:


print(classification_report(y_test, predictions))


# In[ ]:





# In[ ]:





# In[ ]:




