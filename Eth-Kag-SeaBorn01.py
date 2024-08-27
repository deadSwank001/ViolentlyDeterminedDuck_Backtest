#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sklearn as sklearn
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import scipy as scipy
import sklearn as sklearn
from scipy import linalg
from sklearn import linear_model

# libs for sklearn imputer
from sklearn.impute import SimpleImputer


# In[2]:


# import required libs scene above^ line 1


# THIS IS NEW ML PROJECT 01


# In[3]:


df_adv_data = pd.read_csv(
    "C:\\Users\\swank\\OneDrive\\Desktop\\Day-Tah\\Eth01Kag.csv", index_col=0)


# In[4]:


df_adv_data.head()


# In[5]:


# view size of dataset
df_adv_data.size


# In[6]:


# View the shape of dataset
df_adv_data.shape


# In[7]:


# view columns
df_adv_data.columns


# In[8]:


# create a feature object from the columns
#X_feature = df_adv_data[['Newspaper Ad Budget ($)','Radio Ad Budget ($)','TV Ad Budget ($)']]
X_feature = df_adv_data[['Open', 'High', 'Low', 'Close', 'Volume']]

# In[9]:


# view the feature object
X_feature.head()


# In[10]:


# create target object from sales column which is response to dataset
#Y_target = df_adv_data[['Sales ($)']]
Y_target = df_adv_data[['Open']]

# In[11]:


Y_target.head()
# Doesn't referrence correctly (Nevermind)


# In[12]:


# view shapes(feature/target object)
X_feature.shape


# In[13]:


Y_target.shape


# In[14]:


# split test and training data
# by default 75% training data and 25% testing data
x_train, x_test, y_train, y_test = train_test_split(
    X_feature, Y_target, random_state=1)


# In[15]:


# view shape of and test data sets for both feature and response
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[16]:


# lin regresh model [LITERALLY CREATES MODEL]
linreg = LinearRegression()
linreg.fit(x_train, y_train)


# In[17]:


# print intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)


# In[18]:


# prediciton
y_pred = linreg.predict(x_test)
y_pred


# In[19]:


# import required libraries for calculating MSE (mean square error)


# In[20]:


# Calculate the Mean Squared Error(MSE)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
