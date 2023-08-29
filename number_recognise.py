#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import os
for dirname, _, filenames in os.walk('usps.h5'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


# importing libraries

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_curve,auc
import h5py


# In[5]:


# reading the data from the HDF5 file and loading it to x_train, y_train, x_test, y_test variables
with h5py.File('usps.h5', 'r') as f:
    print(list(f.keys()))
    x_train = f['/train/data'][:]
    y_train = f['/train/target'][:]
    x_test = f['/test/data'][:]
    y_test = f['/test/target'][:]


# In[6]:


# no. of records in training and testing data
print('No. of records in Train Dataset: ',len(x_train))
print('No. of records in Test Dataset: ',len(x_test))


# In[7]:


print('Labels of training data: ', set(y_train))


# In[8]:


# reshaping training and testing data to 1D array and then normalizing the pixel values between 0 and 1
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0


# In[9]:


# using cross validation score to calculate optimal value of K

klist = []
K_scores = []
for k in range(3,21):
   klist.append(k)
   knn = KNeighborsClassifier(n_neighbors = k, metric= 'euclidean')
   scores = cross_val_score(knn,x_train,y_train,cv=10, scoring = 'accuracy')
   K_scores.append(scores.mean())


# In[10]:


print(K_scores)


# In[11]:


#calculating Mean Squared Error (MSE)
MSE=[1-x for x in K_scores]


# In[12]:


#plotting graph to visualize error rate and k values
plt.figure(figsize=(12, 6))
plt.plot(range(3,21), MSE, linestyle="solid", marker='o',markersize=8)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[13]:


#value of k for which mse is minimum is optimal
Optimal_K = klist[MSE.index(min(MSE))]

print("The optimal value of K (neighbors) is ",Optimal_K)


# In[14]:


knn = KNeighborsClassifier(metric="euclidean", n_neighbors=3)
knn.fit(x_train, y_train)


# In[15]:


#predicting test data
y_predict= knn.predict(x_test)


# In[16]:


#calculating accuracy of trained model

accuracy=accuracy_score(y_test,y_predict)
print(f'Accuracy Score of the Model: {accuracy}')


# In[17]:


# confusion matrix
cm = confusion_matrix(y_test, y_predict)

fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=np.arange(0, 10), yticklabels=np.arange(0, 10))
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.show()


# In[18]:


# classification Report
report = classification_report(y_test, y_predict, output_dict=True)
df = pd.DataFrame(report).transpose()

fig, ax = plt.subplots(figsize=(15, 8))
ax = sns.heatmap(df.iloc[:-1, :].T, annot=True, cmap='Blues', fmt='.2f')
ax.set_xlabel('Metrics')
ax.set_ylabel('Labels')
ax.set_title('Classification Report')
plt.show()


# In[19]:


# create DataFrame to store y_test and predictions
df = pd.DataFrame({'Actual Labels': y_test, 'Predictions': y_predict})

# save DataFrame as CSV file
df.to_csv('predictions.csv', index=False)


# In[20]:


#Displaying top 15 records of dataframe
df.head(15)

