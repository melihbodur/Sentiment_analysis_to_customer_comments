#!/usr/bin/env python
# coding: utf-8

# In[44]:


a2=a.copy()


# In[43]:


a1=a.copy()


# In[2]:


import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')


# In[8]:


a["markalar"] = a["markalar"].replace({" 'Galaxy A11'": "Galaxy A11"," 'Galaxy A71'":"Galaxy A71"," 'Galaxy A31'":"Galaxy A31"," 'Galaxy A51'":"Galaxy A51"," 'Galaxy M31'":"Galaxy M31"," 'Galaxy M11'":"Galaxy M11"," 'Galaxt Note 10 Lite'":"Galaxt Note 10 Lite"})


# In[3]:


a=pd.read_excel("sentiment.xlsx")


# In[9]:


a


# In[18]:


with pd.option_context("display.max_colwidth",None):
    display(a)


# In[13]:


a["yorumlar"]=a.yorumlar.str[:4999]


# In[16]:


a=a[a["yorumlar"].notnull()]


# In[17]:


#noktalama işaretleri
a["yorumlar"] = a["yorumlar"].str.replace('[^\w\s]','')

#sayılar
a["yorumlar"]= a["yorumlar"].str.replace('\d','')

#stopwords
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
a["yorumlar"]= a["yorumlar"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

#lemmi
from textblob import Word
#nltk.download('wordnet')
a["yorumlar"] = a["yorumlar"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 

#buyuk-kucuk donusumu
a["yorumlar"]= a["yorumlar"].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[19]:


from google_trans_new import google_translator
translator=google_translator()


# In[21]:


a["yorumlar_en"]=a["yorumlar"].map(lambda x:translator.translate(x,lang_src="tr",lang_tgt="en"))


# In[27]:


a


# In[23]:


with pd.option_context("display.max_colwidth",None):
    display(a)


# In[24]:


from textblob import TextBlob


# In[25]:


a["sentiment"]=a["yorumlar_en"].map(lambda x :TextBlob(x).sentiment.polarity)


# In[26]:


a


# In[31]:


for i in a["sentiment"]:
    if i <0:
        a["sentiment"].replace(to_replace=i,value="Negatif",inplace=True)
    elif i>0:
        a["sentiment"].replace(to_replace=i,value="Pozitif",inplace=True)
    else:
        a["sentiment"].replace(to_replace=i,value="Nötr",inplace=True)


# In[ ]:





# In[35]:


a.head(20)


# In[36]:


a["sentiment"].value_counts()


# In[37]:


a["sentiment"].value_counts().plot(kind="bar")


# In[38]:


a.groupby(["markalar","sentiment"])[["sentiment"]].count()


# In[42]:


a.groupby(["markalar","sentiment"])[["sentiment"]].count().plot.barh()


# In[39]:


pd.crosstab(a["markalar"],a["sentiment"]).apply(lambda r:r/r.sum(),axis=1)

