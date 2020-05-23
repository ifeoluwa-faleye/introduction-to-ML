#!/usr/bin/env python
# coding: utf-8

# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iris_dataset = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
iris_dataframe = pd.DataFrame(iris_dataset['data'], columns=iris_dataset.feature_names)
nh = pd.plotting.scatter_matrix(iris_dataframe, c=iris_dataset['target'], figsize=(20, 20), marker='x')


# In[28]:


print ('the description:\n{}'.format(iris_dataset['DESCR']))


# In[6]:


print ('X_test shape:\n {}'.format(X_test.shape))


# In[7]:


print('y_train shape:\n{}'.format(y_train.shape))


# In[8]:


print ('y_test shape:\n{}'.format(y_test.shape))


# In[ ]:




