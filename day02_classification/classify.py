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
# determine if we should do a drop na
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
# duration about 90sec on deepnote
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

df_1 = y_train.loc[y_train == 1]
df_1_features = X_train.loc[df_1.index]

y_target = pd.concat([df_0, df_1], axis=0)
X_features = pd.concat([df_0_features, df_1_features], axis=0)

#%%
# oversample - make minor class about the same size as the major class
df_1 = y_train.loc[y_train == 1].sample(n = 5000, replace=True)
df_1_features = X_train.loc[df_1.index]

y_target = pd.concat([df_1, y_train], axis=0)
X_features = pd.concat([df_1_features, X_train], axis=0)

print('y_target: %d' %len(y_target))
print('X_features: %d' %len(X_features))

#%%
# shuffle the frame and then split it again
# need to shuffle df_target, SGDClassifier does not like it if all the classes are clumped together
from sklearn.utils import shuffle

big = pd.concat([X_features, y_target], axis=1)
big = shuffle(big)

len(big.columns)
y_target = big.iloc[:, 13]
X_features = big.iloc[:, :13]

print('y_target: %d' %len(y_target))
print('X_features: %d' %len(X_features))


#%%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# using oversample
df_target = df_1 # for convenience
scaler = StandardScaler()
df_features_scaled = scaler.fit_transform(X_features)

#scaler returns a numpy array, convert to DataFrame for convenience
# or just us shape
#pca = PCA(n_components=df_features.scaled.shape[1])

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
print (cumm_var)
print('df_target.shape: ', df_target.shape)
print('df_features_scaled.shape: ', df_features_scaled.shape)

#%%
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import learning_curve

pca = PCA(n_components=12)

pca.fit(df_features_scaled)

print('y_target: %d' %len(y_target))
print('df_features_scaled: %d' %len(df_features_scaled))

#%%
subset = 1500

#df_features_scaled_reduced = pca.transform(df_features_scaled)
#print('df_features_scaled_reduced: %d' %len(df_features_scaled_reduced))

classifier = SGDClassifier(max_iter=20000, early_stopping=True)
classifier.fit(df_features_scaled[:subset], y_target[:subset])

#%%

train_size = [ 0.05, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 1 ]
sample_size, train_score, validation_score = learning_curve(classifier, 
        df_features_scaled[:subset], y_target[:subset],
        verbose=1, cv=3, train_sizes=train_size, random_state=42)

#%%
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(sample_size, train_score.mean(axis=1), label='Train', color='b')
ax.plot(sample_size, validation_score.mean(axis=1), label='Validation', color='g')
plt.legend()
plt.grid()
plt.title('Learning curve for SGD')

#%%
from sklearn.svm import SVC

svm = SVC(random_state=42, gamma='auto')

sample_size, train_score, validation_score = learning_curve(svm,
        df_features_scaled[:subset], y_target[:subset],
        verbose=1, cv=3, train_sizes=train_size, random_state=42)

#%%
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(sample_size, train_score.mean(axis=1), label='Train', color='b')
ax.plot(sample_size, validation_score.mean(axis=1), label='Validation', color='g')
plt.legend()
plt.grid()
plt.title('Learning curve for SVM')

#%% [markdown]
# <h1>Evaluation Metrics</h1>

#%%
subset=len(df_features_scaled)
classifier = SGDClassifier(max_iter=2000, early_stopping=True, verbose=1)
classifier.fit(df_features_scaled[:subset], y_target[:subset])

#%%
# X_test, y_test
X_test_scaled = scaler.transform(X_test)

y_predict = classifier.predict(X_test_scaled)
print(y_predict)

#%%
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

#%% [markdown]
# <h2>Classification Report</h2>

#%%
print(classification_report(y_test, y_predict))

#%% 
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.heatmap(confusion_matrix(y_test, y_predict), ax=ax, annot=True, fmt='d')
ax.set_xlabel('Prediction')
ax.set_ylabel('Truth')

#%%
svm = SVC(gamma='auto', random_state=42)
svm.fit(df_features_scaled, y_target)

y_predict = svm.predict(X_test_scaled)

print(classification_report(y_test, y_predict))

ax = sns.heatmap(confusion_matrix(y_test, y_predict), fmt='d', annot=True)
ax.set_xlabel('Predict')
ax.set_ylabel('Truth')

#%%
# ROC/AUC
prob_sgd = classifier.decision_function(X_test_scaled)
fpr, tpr, _ = roc_curve(y_test, prob_sgd)
auc_sgd = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(fpr, tpr, label='Logistic regression: %.2f' %auc_sgd)
ax.set_xlabel('fpr = fp/all -ve')
ax.set_ylabel('tpr = tp/all +v')
ax.set_title('ROC: %.2f' %auc_sgd)
ax.plot([0.0, 1.0], [0.0, 1.0], color='r')
fig.legend()
plt.grid()

#%% [markdown]
# <h1>Polynomial Features</h1>

#%%
# polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

df_features_poly = poly.fit_transform(df_features_scaled)

svm = SVC(gamma='auto', random_state=42, verbose=1)
svm.fit(df_features_poly, y_target)

#%%
X_test_poly = poly.transform(X_test_scaled)
y_predict = svm.predict(X_test_poly)

prob_sgd = svm.decision_function(X_test_poly)
fpr, tpr, _ = roc_curve(y_test, prob_sgd)
auc_sgd = auc(fpr, tpr)

print(auc_sgd)

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(fpr, tpr, label='AUC: %.2f' %auc_sgd)
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('AUC: %.2f' %auc_sgd)
ax.plot([0.0, 1.0], [0.0, 1.0], color='r')
ax.legend()
plt.grid()