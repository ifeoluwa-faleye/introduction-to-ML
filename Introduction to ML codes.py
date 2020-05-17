#!/usr/bin/env python
# coding: utf-8

# In[20]:


data = np.ones(4) 
row_indices = np.arange(4) 
col_indices = np.arange(4) 
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices))) 
print("COO representation:\n{}".format(eye_coo))


# In[11]:


import numpy as np
from scipy import sparse 
x = np.array([[1,2,3],[5,6,7]])
eye = np.eye(4)
sparse_matrix = sparse.csr_matrix(eye)
print('x:\n{}'.format(x))
print('numpy array:\n{}'.format(eye))
print('\nSciPy sparse CSR matrix:\n{}'.format(sparse_matrix))


# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
x = np.linspace(-10,10,100)
y = np.sin(x)
plt.plot(x,y,marker='x')


# In[40]:


import pandas as pd
data = {'Name':['ife','ola','dan','tope'], 'Age':[25,26,27,28],
        'Location':['kano','kano','lagos','nyc']}
data_pandas = pd.DataFrame(data)
display(data_pandas)


# In[ ]:




