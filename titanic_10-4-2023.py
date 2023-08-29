#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=sns.load_dataset("titanic")
df


# In[3]:


df.drop(columns=["pclass","sibsp","parch","fare","embarked","adult_male","alive"],axis=1,inplace=True)


# In[4]:


df


# In[5]:


df.columns.to_list()


# In[6]:


df.isnull().sum()


# In[7]:


import sklearn
from sklearn.impute import SimpleImputer


# In[8]:


imputer=SimpleImputer(missing_values=np.nan,strategy="most_frequent")


# In[9]:


imputer=imputer.fit(df)


# In[10]:


df=imputer.transform(df)


# In[11]:


df


# In[12]:


df=pd.DataFrame(df)
df


# In[13]:


df.isnull().sum()


# In[14]:


df.columns=['survived','sex', 'age', 'class', 'who', 'deck', 'embark_town', 'alone']


# In[15]:


df


# In[16]:


df["sex"].unique()


# In[17]:


df["class"].unique()


# In[18]:


df["who"].unique()


# In[19]:


df["embark_town"].unique()


# In[20]:


df["deck"].unique()


# In[21]:


df["alone"].unique()


# In[22]:


df[['sex', 'class', 'deck', 'embark_town', 'alone','who']]=df[['sex', 'class', 'deck', 'embark_town', 'alone','who']].apply(lambda x:pd.factorize(x)[0])


# In[23]:


df


# In[24]:


sex=-1
cl=-1
deck=-1
et=-1
alone=-1
who=-1
age=-1


# In[25]:


def enquiry():
    g=input("enter the gender:")
    if g=="male" or g=="Male" or g=="m" or g=="M":
        sex=0
    else:
        sex=1
    age=float(input("enter the age:"))
    c=input("enter the class:")
    if c=="Third" or c=="third" or c=="THIRD":
        cl=0
    elif c=="First" or c=="FIRST" or c=="first":
        cl=1
    else:
        cl=2
    w=input("enter the who")
    if w=="man" or w=="MAN" or w=="Man":
        who=0
    elif w=="women" or w=="WOMEN" or W=="Women":
        who=1
    else:
        who=2
    emt=input("enter the ambark_town")
    if emt=="Southampton" or emt=="SOUTHAMPTON" or emt=="southampton":
        et=0
    elif emt=="Cherbourg" or emt=="CHERBOURG" or emt=="cherbourg":
        et=1
    else:
        et=2
    d=input("enter the deck")
    if d=="C" or d=="c":
        deck=0
    elif d=="E" or d=="e":
        deck=1
    elif d=="G" or d=="g":
        deck=2
    elif d=="D" or d=="d":
        deck=3
    elif d=="A" or d=="a":
        deck=4
    elif d=="B" or d=="b":
        deck=5
    else:
        deck=6
    al=bool(input("enter the alone value:"))
    if al=="True" or al=="TRUE" or al=="true":
        alone=0
    else:
        alone=1


# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[27]:


y=df["survived"]
x=df.loc[:,df.columns!="survived"]


# In[28]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[29]:


x_train


# In[30]:


x_test


# In[31]:


y_train


# In[32]:


y_test


# In[33]:


y_train=y_train.astype("int")


# In[34]:


y_train.dtype


# In[35]:


model=LogisticRegression()


# In[36]:


model.fit(x_train,y_train)


# In[37]:


x_test


# In[38]:


enquiry()


# In[39]:


y_pred=model.predict([[sex,age,cl,who,deck,et,alone]])


# In[40]:


if y_pred==0:
    print("person dead")
if y_pred==1:
    print("person survived")
    
    

