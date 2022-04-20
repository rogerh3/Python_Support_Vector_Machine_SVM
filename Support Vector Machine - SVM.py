#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Roger H Hayden III
#Udemy - The Complete Machine Learning Course with Python
#Support Vector Machine - SVM
#4/20/22


# In[66]:


#Packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')

#Used in Linear SVM Implementation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Cross Vaildation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

#Grid Search
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV 

#Support Vector Regression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


# ***
# # Linear SVM Classification
# 
# 
# * Support Vectors
# 
# * Separate with a straight line (linearly separable)
# 
# * Margin
# 
#   * Hard margin classification
#       * Strictly based on those that are at the margin between the two classes
#       * However, this is sensitive to outliers
#       
#   * Soft margin classification
#       * Widen the margin and allows for violation
#       * With Python Scikit-Learn, you control the width of the margin
#       * Control with `C` hyperparameter
#         * smaller `C` leads to a wider street but more margin violations
#         * High `C` - fewer margin violations but ends up with a smaller margin
# 
# 
# 
# **Note:**
# 
# * SVM are sensitive to feature scaling

# In[8]:


#Read in Data
df = pd.read_excel(r'C:\Users\roger\OneDrive\Desktop\iris.xlsx')
df


# In[9]:


col = ['petal_length', 'petal_width', 'species']
df.loc[:, col].head()


# In[10]:


df.species.unique()


# In[11]:


col = ['petal_length', 'petal_width']
X = df.loc[:, col]


# In[12]:


species_to_num = {'setosa': 0,
                  'versicolor': 1,
                  'virginica': 2}
df['tmp'] = df['species'].map(species_to_num)
y = df['tmp']


# Documentations on each:
# 
# * [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)
# 
#   Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
#   
#   
#   
# * [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
# 
#   C-Support Vector Classification.
#   
#   The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples

# In[14]:


C = 0.001
clf = svm.SVC(kernel='linear', C=C)
#clf = svm.LinearSVC(C=C, loss='hinge')
#clf = svm.SVC(kernel='poly', degree=3, C=C)
#clf = s
svm.SVC(kernel='rbf', gamma=0.7, C=C)
clf.fit(X, y)


# In[15]:


clf.predict([[6, 2]])


# In[16]:


Xv = X.values.reshape(-1,1)
h = 0.02
x_min, x_max = Xv.min(), Xv.max() + 1
y_min, y_max = y.min(), y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# In[17]:


z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
fig = plt.figure(figsize=(8,6))
ax = plt.contourf(xx, yy, z, cmap = 'afmhot', alpha=0.3);
plt.scatter(X.values[:, 0], X.values[:, 1], c=y, s=80, 
            alpha=0.9, edgecolors='g');


# ====================================================================

# # Linear SVM Implementation

# In[20]:


col = ['petal_length', 'petal_width']
X = df.loc[:, col]
species_to_num = {'setosa': 0,
                  'versicolor': 1,
                  'virginica': 2}
df['tmp'] = df['species'].map(species_to_num)
y = df['tmp']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8, 
                                                    random_state=0)


# Scaling Features

# In[21]:


sc_x = StandardScaler()
X_std_train = sc_x.fit_transform(X_train)


# In[22]:


C = 1.0 #0.01
clf = svm.SVC(kernel='linear', C=C)
clf.fit(X_std_train, y_train)


# Cross Validation with Train Dataset

# In[24]:


res = cross_val_score(clf, X_std_train, y_train, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))


# In[25]:


y_train_pred = cross_val_predict(clf, X_std_train, y_train, cv=3)


# In[26]:


confusion_matrix(y_train, y_train_pred)


# In[27]:


print("Precision Score: \t {0:.4f}".format(precision_score(y_train, 
                                                           y_train_pred, 
                                                           average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_train,
                                                     y_train_pred, 
                                                     average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_train,
                                             y_train_pred, 
                                             average='weighted')))


# Cross Validation with Test Dataset

# In[28]:


y_test_pred = cross_val_predict(clf, sc_x.transform(X_test), y_test, cv=3)


# In[29]:


confusion_matrix(y_test, y_test_pred)


# In[30]:


print("Precision Score: \t {0:.4f}".format(precision_score(y_test, 
                                                           y_test_pred, 
                                                           average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_test,
                                                     y_test_pred, 
                                                     average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_test,
                                             y_test_pred, 
                                             average='weighted')))


# ===================================================================

# # Polynomial Kernel

# In[32]:


C = 1.0
clf = svm.SVC(kernel='poly', degree=3, C=C, gamma='auto')
clf.fit(X, y)


# In[33]:


Xv = X.values.reshape(-1,1)
h = 0.02
x_min, x_max = Xv.min(), Xv.max() + 1
y_min, y_max = y.min(), y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# In[34]:


z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
fig = plt.figure(figsize=(8,6))
ax = plt.contourf(xx, yy, z, cmap = 'afmhot', alpha=0.3);
plt.scatter(X.values[:, 0], X.values[:, 1], c=y, s=80, 
            alpha=0.9, edgecolors='g');


# =================================================================

# # Polynomial SVM Implementation

# In[35]:


col = ['petal_length', 'petal_width']
X = df.loc[:, col]
species_to_num = {'setosa': 0,
                  'versicolor': 1,
                  'virginica': 2}
df['tmp'] = df['species'].map(species_to_num)
y = df['tmp']
X_train, X_std_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8, 
                                                        random_state=0)


# Scale Features

# In[36]:


sc_x = StandardScaler()
X_std_train = sc_x.fit_transform(X_train)


# In[37]:


C = 1.0 #0.01
clf = svm.SVC(kernel='poly', degree=10, C=C, gamma='auto') #5
clf.fit(X_std_train, y_train)


# Cross Validation with Train Dataset

# In[38]:


res = cross_val_score(clf, X_std_train, y_train, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))


# In[39]:


y_train_pred = cross_val_predict(clf, X_std_train, y_train, cv=3)


# In[40]:


confusion_matrix(y_train, y_train_pred)


# In[41]:


print("Precision Score: \t {0:.4f}".format(precision_score(y_train, 
                                                           y_train_pred, 
                                                           average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_train,
                                                     y_train_pred, 
                                                     average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_train,
                                             y_train_pred, 
                                             average='weighted')))


# Cross Validation with Test Dataset

# In[42]:


y_test_pred = cross_val_predict(clf, sc_x.transform(X_test), y_test, cv=3)


# In[43]:


confusion_matrix(y_test, y_test_pred)


# In[44]:


print("Precision Score: \t {0:.4f}".format(precision_score(y_test, 
                                                           y_test_pred, 
                                                           average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_test,
                                                     y_test_pred, 
                                                     average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_test,
                                             y_test_pred, 
                                             average='weighted')))


# =================================================================

# # Gaussian Radial Basis Function

# The kernel function can be any of the following:
# 
# * **linear**: $\langle x, x'\rangle$.
# 
# 
# * **polynomial**: $(\gamma \langle x, x'\rangle + r)^d$. 
# 
#   $d$ is specified by keyword `degree`
#   
#   $r$ by `coef0`.
# 
# 
# * **rbf**: $\exp(-\gamma \|x-x'\|^2)$. 
# 
#   $\gamma$ is specified by keyword `gamma` must be greater than 0.
# 
# 
# * **sigmoid** $(\tanh(\gamma \langle x,x'\rangle + r))$
# 
#   where $r$ is specified by `coef0`.
#   
# [scikit-learn documentation](http://scikit-learn.org/stable/modules/svm.html#svm)

# In[45]:


col = ['petal_length', 'petal_width']
X = df.loc[:, col]
species_to_num = {'setosa': 0,
                  'versicolor': 1,
                  'virginica': 2}
df['tmp'] = df['species'].map(species_to_num)
y = df['tmp']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8, 
                                                    random_state=0) #0.6


# Scale Features

# In[46]:


sc_x = StandardScaler()
X_std_train = sc_x.fit_transform(X_train)


# In[47]:


C = 1.0
clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
clf.fit(X_std_train, y_train)


# Cross Validation with Training Dataset

# In[52]:


res = cross_val_score(clf, X_std_train, y_train, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))


# In[49]:


y_train_pred = cross_val_predict(clf, X_std_train, y_train, cv=3)


# In[50]:


confusion_matrix(y_train, y_train_pred)


# In[51]:


print("Precision Score: \t {0:.4f}".format(precision_score(y_train, 
                                                           y_train_pred, 
                                                           average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_train,
                                                     y_train_pred, 
                                                     average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_train,
                                             y_train_pred, 
                                             average='weighted')))


# Grid Search

# In[54]:


pipeline = Pipeline([('clf', svm.SVC(kernel='rbf', C=1, gamma=0.1))]) 


# In[55]:


params = {'clf__C':(0.1, 0.5, 1, 2, 5, 10, 20), 
          'clf__gamma':(0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1)} 


# In[56]:


svm_grid_rbf = GridSearchCV(pipeline, params, n_jobs=-1,
                            cv=3, verbose=1, scoring='accuracy') 


# In[57]:


svm_grid_rbf.fit(X_train, y_train) 


# In[58]:


svm_grid_rbf.best_score_


# In[59]:


best = svm_grid_rbf.best_estimator_.get_params() 


# In[60]:


for k in sorted(params.keys()): 
    print('\t{0}: \t {1:.2f}'.format(k, best[k]))


# In[61]:


y_test_pred = svm_grid_rbf.predict(X_test)


# In[62]:


confusion_matrix(y_test, y_test_pred)


# In[63]:


print("Precision Score: \t {0:.4f}".format(precision_score(y_test, 
                                                           y_test_pred, 
                                                           average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_test,
                                                       y_test_pred, 
                                                       average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_test,
                                               y_test_pred, 
                                               average='weighted')))


# In[64]:


Xv = X.values.reshape(-1,1)
h = 0.02
x_min, x_max = Xv.min(), Xv.max() + 1
y_min, y_max = y.min(), y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# In[65]:


z = svm_grid_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
fig = plt.figure(figsize=(8,6))
ax = plt.contourf(xx, yy, z, cmap = 'afmhot', alpha=0.3);
plt.scatter(X.values[:, 0], X.values[:, 1], c=y, s=80, 
            alpha=0.9, edgecolors='g');


# ===================================================================

# # Support Vector Regression

# In[68]:


#Read in Data
df = pd.read_csv(r'C:\Users\roger\OneDrive\Desktop\Boston_Housing.csv', delim_whitespace = True, header = None)
df


# In[69]:


col_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.columns = col_name
df.head()


# In[83]:


X = df[['LSTAT']].values
y = df['MEDV'].values


# In[84]:


svr = SVR(gamma='auto')
svr.fit(X, y)


# In[85]:


sort_idx = X.flatten().argsort()


# In[86]:


plt.figure(figsize=(10,8))
plt.scatter(X[sort_idx], y[sort_idx])
plt.plot(X[sort_idx], svr.predict(X[sort_idx]), color='k')

plt.xlabel('LSTAT')
plt.ylabel('MEDV');


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42)


# Linear Kernel

# In[88]:


svr = SVR(kernel='linear')
svr.fit(X_train, y_train)


# In[89]:


y_train_pred = svr.predict(X_train)


# In[90]:


y_test_pred = svr.predict(X_test)


# In[91]:


print("MSE train: {0:.4f}, test: {1:.4f}".      format(mean_squared_error(y_train, y_train_pred), 
             mean_squared_error(y_test, y_test_pred)))


# In[92]:


print("R^2 train: {0:.4f}, test: {1:.4f}".      format(r2_score(y_train, y_train_pred),
             r2_score(y_test, y_test_pred)))


# Polynomial

# In[93]:


svr = SVR(kernel='poly', C=1e3, degree=2, gamma='auto')
svr.fit(X_train, y_train)


# In[94]:


y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)


# In[95]:


print("MSE train: {0:.4f}, test: {1:.4f}".      format(mean_squared_error(y_train, y_train_pred), 
             mean_squared_error(y_test, y_test_pred)))
print("R^2 train: {0:.4f}, test: {1:.4f}".      format(r2_score(y_train, y_train_pred),
             r2_score(y_test, y_test_pred)))


# RBF Kernel

# In[96]:


svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr.fit(X_train, y_train)


# In[97]:


y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)


# In[98]:


print("MSE train: {0:.4f}, test: {1:.4f}".      format(mean_squared_error(y_train, y_train_pred), 
             mean_squared_error(y_test, y_test_pred)))
print("R^2 train: {0:.4f}, test: {1:.4f}".      format(r2_score(y_train, y_train_pred),
             r2_score(y_test, y_test_pred)))


# ***
# # Advantages and Disadvantages
# 
# 
# 
# The **advantages** of support vector machines are:
# * Effective in high dimensional spaces.
# * Uses only a subset of training points (support vectors) in the decision function.
# * Many different Kernel functions can be specified for the decision function.
#     * Linear
#     * Polynomial
#     * RBF
#     * Sigmoid
#     * Custom
# 
# 
# The **disadvantages** of support vector machines include:
# * Beware of overfitting when num_features > num_samples.
# * Choice of Kernel and Regularization can have a large impact on performance
# * No probability estimates
