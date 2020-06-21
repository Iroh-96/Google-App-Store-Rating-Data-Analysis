#!/usr/bin/env python
# coding: utf-8

# # DATA ANALYSIS OF GOOGLE APP'S RATINGS
# 

# In[1]:


#Importing Libraies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Reading Data 
data = pd.read_csv('C:/Users/googleplaystore.csv')


# In[3]:


data.head(5) 


# In[4]:


data.shape


# In[5]:


data.describe() 


# In[6]:


data.boxplot()


# In[7]:


data.hist()


# In[8]:



data.info()


# In[9]:


#Data Cleaning
#Count the number of missing values in the Dataframe
data.isnull()


# In[10]:


# Count the number of missing values in each column
data.isnull().sum()


# In[11]:


#how many ratings are more than 5 - Outliers, Because we are working on that part only
data[data.Rating > 5]


# In[12]:


data.drop([10472],inplace=True)


# In[13]:


#Reviewing the data between the columns
data[10470:10475]


# In[14]:


data.boxplot()


# In[15]:


data.hist()


# In[16]:


#Remove columns that are 90% empty
threshold = len(data)* 0.1
threshold


# In[17]:


data.dropna(thresh=threshold, axis=1, inplace=True)


# In[18]:


print(data.isnull().sum())


# In[19]:


#Data Imputation and Manipulation

#Define a function impute_median
def impute_median(series):
    return series.fillna(series.median())


# In[20]:


data.Rating = data['Rating'].transform(impute_median)


# In[21]:


#count the number of null values in each column
data.isnull().sum()


# In[22]:


# modes of categorical values
print(data['Type'].mode())
print(data['Current Ver'].mode())
print(data['Android Ver'].mode())


# In[23]:


# Fill the missing categorical values with mode
data['Type'].fillna(str(data['Type'].mode().values[0]), inplace=True)
data['Current Ver'].fillna(str(data['Current Ver'].mode().values[0]), inplace=True)
data['Android Ver'].fillna(str(data['Android Ver'].mode().values[0]), inplace=True)


# In[24]:


#count the number of null values in each column
data.isnull().sum()


# In[25]:


### Let's convert Price, Reviews and Ratings into Numerical Values
data['Price'] = data['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
data['Price'] = data['Price'].apply(lambda x: float(x))
data['Reviews'] = pd.to_numeric(data['Reviews'], errors='coerce')


# In[26]:


data['Installs'] = data['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else str(x))
data['Installs'] = data['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else str(x))
data['Installs'] = data['Installs'].apply(lambda x: float(x))


# In[27]:


data.head(10)


# In[28]:


data.describe()


# In[29]:


#Data Visualization
grp = data.groupby('Category')
x = grp['Rating'].agg(np.mean)
y = grp['Price'].agg(np.sum)
z = grp['Reviews'].agg(np.mean)
print(x)
print(y)
print(z)


# In[30]:


plt.figure(figsize=(12,5))
plt.plot(x, "ro", color='g')
plt.xticks(rotation=90)
plt.show()


# In[31]:


plt.figure(figsize=(16,5))
plt.plot(x,'ro', color='r')
plt.xticks(rotation=90)
plt.title('Category wise Rating')
plt.xlabel('Categories-->')
plt.ylabel('Rating-->')
plt.show()


# In[32]:


plt.figure(figsize=(16,5))
plt.plot(y,'r--', color='b')
plt.xticks(rotation=90)
plt.title('Category wise Pricing')
plt.xlabel('Categories-->')
plt.ylabel('Prices-->')
plt.show()


# In[33]:


plt.figure(figsize=(16,5))
plt.plot(z,'bs', color='g')
plt.xticks(rotation=90)
plt.title('Category wise Reviews')
plt.xlabel('Categories-->')
plt.ylabel('Reviews-->')
plt.show()


# In[ ]:




