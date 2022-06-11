#!/usr/bin/env python
# coding: utf-8

# # Importing the dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# Data Collection and Analysis

# Loading a csv file into pandas dataframe

# In[2]:


Big_Mart_Data=pd.read_csv("Train.csv")
Big_Mart_Data


# Loading First 5 rows of the dataframe

# In[3]:


Big_Mart_Data.head()


# Number of datapoints and number of features

# In[4]:


Big_Mart_Data.shape


# Getting information about dataset

# In[5]:


Big_Mart_Data.info


# Categorical Features:
# 
# *Item_Identifier
# *Item_Fat_Content
# *Item_Type
# *Outlet_Identifier
# *Outlet_Size
# *Outlet_Location_Type
# *Outlet_Type

# Checking for missing values

# In[6]:


Big_Mart_Data.isnull().sum()


# Handling missing values

# Mean --> average
# 
# Mode --> more repeated value

# mean value of "Item_Weight" column

# In[7]:


Big_Mart_Data['Item_Weight'].mean()


# filling the missing values in "Item_weight column" with "Mean" value

# In[8]:


Big_Mart_Data['Item_Weight'].fillna(Big_Mart_Data['Item_Weight'].mean(), inplace=True)


# mode of "Outlet_Size" column

# In[9]:


Big_Mart_Data['Outlet_Size'].mode()


#  filling the missing values in "Outlet_Size" column with Mode

# In[10]:


mode_of_Outlet_size = Big_Mart_Data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))


# In[11]:


print(mode_of_Outlet_size)


# In[12]:


miss_values = Big_Mart_Data['Outlet_Size'].isnull() 


# In[13]:


print(miss_values)


# In[14]:


Big_Mart_Data.loc[miss_values, 'Outlet_Size'] = Big_Mart_Data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])


# Checking for missing values

# In[15]:


Big_Mart_Data.isnull().sum()


# Data Analysis

# Statistical measures about data

# In[16]:


Big_Mart_Data.describe()


# Numerical Features

# In[17]:


sns.set()


# Item Weight Distibution

# In[18]:


plt.figure(figsize=(6,6))
sns.distplot(Big_Mart_Data['Item_Weight'])
plt.show()


# Item Visibility distribution

# In[19]:


plt.figure(figsize=(6,6))
sns.distplot(Big_Mart_Data['Item_Visibility'])
plt.show()


# Item MRP distribution

# In[20]:


plt.figure(figsize=(6,6))
sns.distplot(Big_Mart_Data['Item_MRP'])
plt.show()


# Item_Outlet_Sales distribution

# In[21]:


plt.figure(figsize=(6,6))
sns.distplot(Big_Mart_Data['Item_Outlet_Sales'])
plt.show()


# Outlet_Establishment_Year column

# In[22]:


plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=Big_Mart_Data)
plt.show()


# Categorical features

# In[23]:


plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=Big_Mart_Data)
plt.show()


# Item_Type_Column

# In[24]:


plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=Big_Mart_Data)
plt.show()


# Outlet_Size Column

# In[25]:


plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=Big_Mart_Data)
plt.show()


# Data Pre-Processing

# In[26]:


Big_Mart_Data.head()


# In[27]:


Big_Mart_Data['Item_Fat_Content'].value_counts()


# In[28]:


Big_Mart_Data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)


# In[29]:


Big_Mart_Data['Item_Fat_Content'].value_counts()


# Label Encoding

# In[30]:


encoder = LabelEncoder()


# In[31]:


Big_Mart_Data['Item_Identifier'] = encoder.fit_transform(Big_Mart_Data['Item_Identifier'])

Big_Mart_Data['Item_Fat_Content'] = encoder.fit_transform(Big_Mart_Data['Item_Fat_Content'])

Big_Mart_Data['Item_Type'] = encoder.fit_transform(Big_Mart_Data['Item_Type'])

Big_Mart_Data['Outlet_Identifier'] = encoder.fit_transform(Big_Mart_Data['Outlet_Identifier'])

Big_Mart_Data['Outlet_Size'] = encoder.fit_transform(Big_Mart_Data['Outlet_Size'])

Big_Mart_Data['Outlet_Location_Type'] = encoder.fit_transform(Big_Mart_Data['Outlet_Location_Type'])

Big_Mart_Data['Outlet_Type'] = encoder.fit_transform(Big_Mart_Data['Outlet_Type'])


# In[32]:


Big_Mart_Data.head()


# Splitting features and Target

# In[33]:


X = Big_Mart_Data.drop(columns='Item_Outlet_Sales', axis=1)
Y = Big_Mart_Data['Item_Outlet_Sales']


# In[34]:


print(X)


# In[35]:


print(Y)


# In[36]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[37]:


print(X.shape, X_train.shape, X_test.shape)


# Machine Learning Model Training

# XGBoost Regressor

# In[38]:


regressor = XGBRegressor()


# In[39]:


regressor.fit(X_train, Y_train)


# Evaluation

# prediction on training data

# In[40]:


training_data_prediction = regressor.predict(X_train)


# R squared Value

# In[41]:


r2_train = metrics.r2_score(Y_train, training_data_prediction)


# In[42]:


print('R Squared value = ', r2_train)


# prediction on test data

# In[43]:


test_data_prediction = regressor.predict(X_test)


# R squared Value

# In[44]:


r2_test = metrics.r2_score(Y_test, test_data_prediction)


# In[45]:


print('R Squared value = ', r2_test)


# In[ ]:




