#%% [markdown]
#### Classification problem
# The data set can be found here
# [Kiran Loans](https://www.kaggle.com/kirankarri/kiran-loans/downloads/kiran-loans.zip/1)

#%%
# import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

#%%
df = pd.read_csv('day02_classification/data/loans.csv')
df.head()

#%%
# any na
for i, v in enumerate(df.isna().sum()):
    if (v > 0):
        print('%s\t\t: %d' %(df.columns[i], v))

#%%
print('dataset size: %d' %len(df))
print('max na: %d' %max(df.isna().sum()))