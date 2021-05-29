#%% import library
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns

#%% [markdown]
# Download Winton file from [here](https://www.kaggle.com/c/the-winton-stock-market-challenge/data)

#%%
cols = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5'
    , 'Ret_MinusTwo', 'Ret_MinusOne', 'Ret_2', 'Ret_3']


target = 'Ret_3'

df = pd.read_csv('day02_regression/data/winton/train.csv'
    , usecols=cols
)

#%%
# Another way to determine the cols - with least na
cols = [ c for c in df.columns if c.startswith('Feature_') ]
# sort and pick the top 6 columns the least NAs
df.loc[:, cols].isna().sum().sort_values()

# get the index name
selected_cols = df.loc[:, cols].isna().sum().sort_values().index
# convert to regular array, numpy array use .values
selected_cols = selected_cols.tolist()

#%%
# verify that there are lots of NaN in Feature_1
df.isna().sum()

#%% [markdown]
# <h1>Clean up</h1>
# <p>Too much na to drop, so we drop all targets with NaN</p>
# <p>Then we get the indexed of and drop the corresponding freatures</p>
# <p>Then we fillna on all the missing features</p> 

#%%
# separate the features from the target
df_features = df.loc[:, cols[0: len(cols) - 1]]
df_target = df.loc[:, target]
df_target

#%%
# get the index of the NaN from the target
na = df_target.isna()
# same: idx = na[na].index
idx = na[na == True].index

df_features_clean = df_features.drop(index=idx)
df_target_clean = df_target.drop(index=idx)

#%%
# backfill the nan in df_features_clean
df_features_clean.fillna(df.mean(), inplace=True)
len(df_features_clean) == len(df_target_clean)

#%%
# run correlation
fig = plt.figure()
subplt = fig.add_subplot(111)

sns.heatmap(df_features_clean.corr(method='pearson'), ax=subplt, annot=True, fmt=".2f")

#%%
sns.set(color_codes=True)
pp = sns.pairplot(df_features_clean)

#%%
df_features_clean.head(20)

#%%
from sklearn.model_selection import train_test_split

print(len(df_features_clean))
print(len(df_target_clean))

X_train, X_test, y_train, y_test = train_test_split(df_features_clean, df_target_clean, random_state=42)
print('X_train: %d, y_train: %d' %(len(X_train), len(y_train)))
print('X_test: %d, y_test: %d' %(len(X_test), len(y_test)))

#%%
# Scale the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# scale/train on the X_train
X_train_scaled = scaler.fit_transform(X_train)

# do not fit_transform on the test
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled).isna().any()


#%%
# PCA plot, need to scale first
from sklearn.decomposition import PCA

pca = PCA(n_components=len(df_features_clean.columns))
pca.fit_transform(X_train_scaled)
print('Explain variance: ', pca.explained_variance_)
print('Explain variance ratio: ', pca.explained_variance_ratio_)

cumm_var = [sum(pca.explained_variance_ratio_[0: i + 1]) for i in range(len(pca.explained_variance_))]

print('Cummulative variance: ', cumm_var)

plt.plot(range(len(cumm_var)), cumm_var, marker='o', markerfacecolor='g')

# comp = []
# variances = []

# for i in range(2, len(df_features_clean.columns)):
#     pca = PCA(n_components=i)
#     pca.fit_transform(X_train_scaled)
#     print('%d: ' %i, pca.explained_variance_ratio_)
#     variances.append(sum(pca.explained_variance_ratio_))
#     comp.append(i)

# print('comp: ', comp)
# print('variances: ', variances)
# plt.plot(comp, variances, marker='o', markerfacecolor='r')

#%%
# Linear regression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import learning_curve

sgd = SGDRegressor(tol=1e-7, max_iter=10000)

train_size = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ]

print('train size = ', len(X_train_scaled))

sample_size, train_score, test_score = learning_curve(sgd, X_train_scaled, y_train
        , train_sizes=train_size, verbose=1, cv=3)

#%%
print('train sizes: ', train_size)
print('train score: ', train_score)
print('val score: ', test_score)

#%%
plt.plot(sample_size, train_score.mean(axis=1), label='Train', color='b')
plt.plot(sample_size, test_score.mean(axis=1), label='Test', color='orange')
plt.legend()

#%%
sgd = SGDRegressor(max_iter=10000)
sgd.fit(X_train_scaled, y_train)

#%%
from sklearn.metrics import r2_score, mean_squared_error

print('R-squared: ', sgd.score(X_test_scaled, y_test) * 100)

y_predict = sgd.predict(X_test_scaled)

print('MSE: ', mean_squared_error(y_test, y_predict))
print('R^2: ', r2_score(y_test, y_predict))


#%%
# pickle it
import pickle

with open('winton.pickle', 'wb') as f:
    pickle.dump({ 'scaler': scaler, 'model': sgd}, f)
    f.flush()

#%%
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(range(len(y_test)), y_test, label='Actual')
ax.plot(range(len(y_predict)), y_predict, label='Predict')
plt.legend()
plt.grid()