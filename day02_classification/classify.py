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

#%%
df.dropna(inplace=True)
print('dataset size after dropna: %d' %len(df))

#%%
# ['debt_consolidation', 'credit_card', 'all_other', 'home_improvement', 'small_business', 'major_purchase', 'educational']
df['purpose'].unique()

#%%
# Dummy variables or Label Encoder
#purpose_dummies = pd.get_dummies(df.purpose)
#df = pd.concat([df, purpose_dummies], axis=1)
#df.drop(columns=['purpose'], inplace=True)
#
#df.head(10)

#%%
# Encoding values to int
from sklearn.preprocessing import LabelEncoder

le_purpose = LabelEncoder()
le_purpose.fit(df['purpose'])
le_purpose.transform(df.purpose)

df.purpose = le_purpose.transform(df.purpose)
df

#%%
import seaborn as sns
sns.pairplot(df.iloc[:, 0: len(df) - 1])

#%%
fig, axes = plt.subplots(figsize=(10, 10))
sns.heatmap(df.iloc[:, 0: len(df) - 1].corr(), annot=True, fmt='.2f', ax=axes)

#%%
df_target = df['not.fully.paid']
df_features = df.drop(columns=['not.fully.paid'], axis=1)

df_features.columns

#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.1)

#%%
# imbalance check
y_train.hist()

print('train zeros: %d' %len(y_train.loc[y_train == 0]))
print('train ones: %d' %len(y_train.loc[y_train == 1]))
print('train total: %d' %len(y_train))

print('test zeros: %d' %len(y_test.loc[y_test == 0]))
print('test ones: %d' %len(y_test.loc[y_test == 1]))
print('test total: %d' %len(y_test))

#%%
# undersample - make major class about the same size as minor class
# no replacement
df_0 = y_train.loc[y_train == 0].sample(n = 2000, replace=False)
df_0_features = X_train.loc[df_0.index]

#%%
# oversample - make minor class about the same size as the major class
df_1 = y_train.loc[y_train == 1].sample(n = 7000, replace=True)
df_1_features = X_train.loc[df_1.index]

print('df_1: %d' %len(df_1))
print('df_1_features: %d' %len(df_1_features))

print('df_1.index', df_1.head(5).index)
print('df_1_features.index', df_1_features.head(5).index)

#%%
df_1_features.head(5)

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# using oversample
df_target = df_1 # for convenience
scaler = StandardScaler()
df_features_scaled = scaler.fit_transform(df_1_features)

#scaler returns a numpy array, convert to DataFrame for convenience
df_features_scaled = pd.DataFrame(df_features_scaled, columns=df_1_features.columns)

pca = PCA(n_components=len(df_features_scaled.columns))
pca.fit_transform(df_features_scaled)
pca.explained_variance_ratio_

#%%
cumm_var = []
print('variance ratio: ', pca.explained_variance_ratio_)

for i in range(len(pca.explained_variance_ratio_)):
        cumm_var.append(sum(pca.explained_variance_ratio_[: i]))

cumm_var = [ sum(pca.explained_variance_ratio_[: i]) for i in range(len(pca.explained_variance_ratio_)) ]

plt.plot(range(1, 1 + len(cumm_var)), cumm_var, marker='o', markerfacecolor='r')
plt.grid()

#%%
cumm_var

#%%
print('df_target.shape: ', df_target.shape)
print('df_features_scaled.shape: ', df_features_scaled.shape)

#%%
# train PCA on 12 features
pca = PCA(n_components=12)
pca.fit(df_features_scaled)