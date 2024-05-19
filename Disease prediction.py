#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


training = pd.read_csv('/Users/balwiiredavis/Desktop/Training.csv')


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[5]:


training.head()


# In[6]:


training.columns


# In[7]:


# Drop unnamed column

training = training.loc[:, ~training.columns.str.contains('Unnamed: 133')]


# In[8]:


training.head()


# In[9]:


# Split data into features and target

X = training.drop(columns=['prognosis'])
y = training['prognosis']


# In[10]:


# Encode the target variable

le = LabelEncoder()
y_encoded = le.fit_transform(y)


# In[11]:


# Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# In[12]:


# Standardize the features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[13]:


# train the model

model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)


# In[14]:


# Make predictions

y_pred = model.predict(X_test)


# In[15]:


# Evaluate the model

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)


# In[16]:


# Load the testing dataset

testing = pd.read_csv('/Users/balwiiredavis/Desktop/project/archive(32)/Testing.csv')


# In[17]:


testing.head()


# In[18]:


# Split the testing data into features and target

X_test = testing.drop(columns=['prognosis'])
y_test = testing['prognosis']


# In[19]:


# Encode the target variable

y_test_encoded = le.transform(y_test)


# In[20]:


# Standardize the features

X_test = scaler.transform(X_test)


# In[21]:


# Make predictions on the testing data

y_pred_test = model.predict(X_test)


# In[22]:


# Evaluate the model

accuracy_test = accuracy_score(y_test_encoded, y_pred_test)
conf_matrix_test = confusion_matrix(y_test_encoded, y_pred_test)
class_report_test = classification_report(y_test_encoded, y_pred_test)


print(f'Testing Accuracy: {accuracy_test}')
print('Testing Confusion Matrix:')
print(conf_matrix_test)
print('Testing Classification Report:')
print(class_report_test)


# In[ ]:




