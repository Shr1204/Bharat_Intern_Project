#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r"C:\Users\ayush\Downloads\NFLX.csv")
df


# In[3]:


df.isnull()


# In[4]:


df.isnull().sum()


# In[5]:


df.shape


# In[6]:


df=df["Close"]


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[8]:


scaler=MinMaxScaler((0,1))
data=scaler.fit_transform(np.array(df).reshape([df.shape[0],1]))


# In[9]:


time_step=100
def createData(data):
    x=[]
    y=[]
    for i in range(len(data)-time_step-1):
        x.append(data[i:(i+time_step)])
        y.append(data[i+time_step])
    return x,y


# In[10]:


x,y=createData(data)


# In[11]:


x=np.array(x)
x=x.reshape(x.shape[0],x.shape[1],1)
y=np.array(y)


# In[12]:


df.shape


# In[13]:


xtrain,xtest,ytrain,ytest=x[:int(df.shape[0]*0.8)],x[int(df.shape[0]*0.8):],y[:int(df.shape[0]*0.8)],y[int(df.shape[0]*0.8):]


# In[14]:


pip install tensorflow


# In[15]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,LSTM


# In[16]:


model=Sequential([
    LSTM(128,return_sequences=True,input_shape=xtrain[0].shape),
    LSTM(64,return_sequences=True),
    LSTM(32),
    Dense(16,activation="relu"),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss="mean_squared_error",
metrics=[tf.keras.metrics.RootMeanSquaredError()])


# In[17]:


model.fit(xtrain,ytrain,epochs=100)


# In[18]:


model.evaluate(xtest,ytest)


# In[19]:


trainPred=scaler.inverse_transform(model.predict(xtrain)).squeeze()
testPred=scaler.inverse_transform(model.predict(xtest)).squeeze()


# In[20]:


look_back=time_step
trainPredPlot=np.empty_like(df)
trainPredPlot[:]=np.nan
trainPredPlot[look_back:len(trainPred)+look_back]=trainPred
testPredPlot=np.empty_like(df)
testPredPlot[:]=np.nan
testPredPlot[len(trainPred)+look_back:len(trainPred)+look_back+len(testPred)]=testPred

plt.plot(df,label="Actual close price")
plt.plot(trainPredPlot,label="Training prediction close price")
plt.plot(testPredPlot,label="Predicted close price")
plt.legend()
plt.show()


# # Next 30 days prediction

# In[21]:


input_data=np.array(df[-time_step:])
input_data=input_data.reshape([input_data.shape[0],1])


# In[22]:


def predict(data,days=30):
    data=scaler.transform(data)
    predictions=[]
    i=1
    while(i<=days):
        nxtday=model.predict([data],verbose=0)
        predictions.append(scaler.inverse_transform(nxtday)[0])
        data[:-1]=data[1:]
        data[-1]=nxtday[0]
        i+=1
    return np.array(predictions).squeeze()      


# In[23]:


days=30
predictions=predict(input_data,days)


# In[24]:


trainPredPlot=np.zeros(shape=[len(input_data)+1+days])
trainPredPlot[:]=np.nan
trainPredPlot[len(input_data)]=input_data[-1]
trainPredPlot[len(input_data)+1:]=predictions
df_=input_data
plt.plot(df_,label="Actual close price")
plt.plot(trainPredPlot,label="Predicted close price")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




