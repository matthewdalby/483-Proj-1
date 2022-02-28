#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


survey_data = pd.read_csv("HappinessData-1.csv")
column_to_move = survey_data.pop("Unhappy/Happy")
survey_data.insert(len(survey_data.columns), "Unhappy/Happy", column_to_move)


# In[3]:


survey_data = survey_data.dropna()


# In[4]:


print(survey_data)


# In[5]:


correlation = survey_data.corr()


# In[6]:


print(correlation)

#Duc's observation:
#The highest correlation between the features and unhappy/happy is 
#city services availability, community maintenance, 
#and availability of community room.


# In[7]:


#KNN using sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[8]:


X = survey_data.iloc[:, :-1].values
y = survey_data.iloc[:, -1].values


# In[9]:


print(X)
print(y)


# In[10]:


#Seperate test set and training set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)


# In[11]:


print(X_train)
print(y_train)


# In[12]:


print(X_test)
print(y_test)


# In[13]:


#Sklearn's KNN algorithm with 5 neighbors
classifier_default = KNeighborsClassifier(n_neighbors=5)
classifier_default.fit(X_train, y_train)
y_default_pred = classifier_default.predict(X_test)


# In[14]:


#Classification Report
classification_report_default = metrics.classification_report(y_test, y_default_pred)
print(classification_report_default)


# In[15]:


#Finding the perfect numbers of neighbors for sklearn's KNN algorithm
k_range = range(1, 40)
errors_list = []


# In[16]:


for k in k_range:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    errors_list.append(1.0 - metrics.accuracy_score(y_test,y_pred))


# In[17]:


#Plot the accuracy values for sklearn's KNN algorithm with the range of [1, 40]
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[18]:


plt.plot(k_range,errors_list, linestyle='dashed', marker='o')
plt.xlabel("K Value")
plt.ylabel("Error Rate")


# In[19]:


#Predicting with custom KNN and euclidean distance
from duc_knn import KNN
from duc_distance import euclidean_distance


# In[20]:


KNN_euclidean_default = KNN(euclidean_distance)
KNN_euclidean_default.fit(X_train, y_train)
y_euclidean_pred = KNN_euclidean_default.predict(X_test)


# In[21]:


classification_report_euclidean = metrics.classification_report(y_test, y_euclidean_pred)
print(classification_report_euclidean)


# In[22]:


errors_euclidean_list = []

for k in k_range:
    classifier = KNN(euclidean_distance, k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    errors_euclidean_list.append(1.0 - metrics.accuracy_score(y_test,y_pred))


# In[23]:


plt.plot(k_range, errors_euclidean_list, linestyle='dashed', marker='o')
plt.xlabel("K Value")
plt.ylabel("Error Rate")


# In[24]:


#Predicting with custom KNN and cosine distance
from duc_distance import cosine_distance


# In[25]:


KNN_cosine_default = KNN(cosine_distance)
KNN_cosine_default.fit(X_train, y_train)
y_cosine_pred = KNN_cosine_default.predict(X_test)


# In[26]:


classification_report_cosine = metrics.classification_report(y_test, y_cosine_pred)
print(classification_report_cosine)


# In[27]:


errors_cosine_list = []

for k in k_range:
    classifier = KNN(cosine_distance, k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    errors_cosine_list.append(1.0 - metrics.accuracy_score(y_test,y_pred))


# In[28]:


plt.plot(k_range, errors_cosine_list, linestyle='dashed', marker='o')
plt.xlabel("K Value")
plt.ylabel("Error Rate")

