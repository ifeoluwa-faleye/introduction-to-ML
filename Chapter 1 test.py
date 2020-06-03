#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Step 1: Import the neccessary libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[6]:


#step 2: visualze the data
dataset = load_iris()
display (dataset) #Tiis allows you see the data you have imported 
#To visualize the data we simply do split plots and the
#To do this, we convert the data from array to pandas DataFrame
df = pd.DataFrame(dataset['data'], columns = dataset.feature_names)
grr = pd.plotting.scatter_matrix(df, c = dataset['target'], figsize =(20,20), marker = 'o')


# In[12]:


#We can also do a box plot 
df.plot(kind ='box',subplots = True, layout =(2,2), sharex = False, sharey = False)


# In[13]:


#Step 3: We split our dataset to train set and test set using the train_test_split method
X_train,X_test,y_train,y_test = train_test_split(dataset['data'],dataset['target'],random_state = 0)


# In[18]:


#Step 4: we call the fit method of the knn on the training set
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)


# In[19]:


#Step 5: We call the predict method of the knn to test our set
y_pred = knn.predict(X_test)


# In[24]:


print ('The prediction:\n{}'.format(y_pred))
print ('The prediction:\n{}'.format(dataset['target_names'][y_pred]))


# In[25]:


#Step 6: Finally we test for reliability using the score method of the knn
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:




