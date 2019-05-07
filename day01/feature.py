#%% [markdown]
# Import libraries

#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#%%
df = pd.read_csv('/home/cmlee/tmp/kml03_practice/day01/data/bitflyerJPY_1-min_data_2018-06-01_to_2018-06-27.csv')

#%%
df.columns

#%%
high = df.High.copy()
low = df.Low.copy()

#%%
plt.legend()
plt.scatter(range(len(high)), high, label='High')
plt.scatter(range(len(low)), low, label='Low')

#%%
high.hist()

#%%
fig = plt.figure()
subplt = fig.add_subplot(121)
subplt.scatter(range(len(high)), high, label="High")
subplt.set_title('High')

subplt = fig.add_subplot(122)
subplt.scatter(range(len(low)), low, label="Low", color='r')
subplt.set_title('Low')
