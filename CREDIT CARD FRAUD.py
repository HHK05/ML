#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


df_train=pd.read_csv("fraudTrain.csv")
df_train.head()


# In[3]:


df_test=pd.read_csv("fraudTest.csv")
df_test.head()


# In[4]:


df_train.info()


# In[5]:


df_test.info()


# In[6]:


df_train.describe()


# In[7]:


df_test.describe()


# In[8]:


df_train.isnull().sum()


# In[9]:


df_test.isnull().sum()


# In[10]:


df_train.dropna(inplace=True)


# In[11]:


df_test.dropna(inplace=True)


# In[12]:


df_train.describe()


# In[13]:


df_test.describe()


# In[14]:


plt.figure(figsize=(6,4))
sns.countplot(x=df_train['is_fraud'])
plt.title("understanding stats")
plt.xlabel("number of fraud occured")
plt.ylabel("entire plot")
plt.show()


# In[15]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x=df_train['is_fraud'], y=df_train['amt'])
plt.title('Transaction Amount vs. Fraud')
plt.xlabel('Is Fraud')
plt.ylabel('Transaction Amount')
plt.show()


# In[16]:


plt.figure(figsize=(8,6))
sns.countplot(x=df_train['gender'],hue=df_train['is_fraud'])
plt.title("Gender Based analysis")
plt.xlabel("Gender")
plt.ylabel("fraudes")
plt.show()


# In[17]:


# Time analysis: Extract hours and days from 'trans_date_trans_time'
df_train['trans_hour'] = pd.to_datetime(df_train['trans_date_trans_time']).dt.hour
df_train['trans_day'] = pd.to_datetime(df_train['trans_date_trans_time']).dt.dayofweek

# Plot hourly distribution of fraud
plt.figure(figsize=(10, 6))
sns.countplot(x='trans_hour', hue='is_fraud', data=df_train)
plt.title('Hourly Distribution of Fraudulent Transactions')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.legend(title='Is Fraud')
plt.show()


# In[18]:


plt.figure(figsize=(8,6))
sns.countplot(x=df_train["trans_day"],hue=df_train['is_fraud'])
plt.title("DAY WISE DISTRIBUTION")
plt.xlabel("day")
plt.xticks([0,1,2,3,4,5,],['MON','TUE','WED','THRU','FRI','SAT'])
plt.ylabel("frauding")
plt.show()


# In[19]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[20]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[21]:


encoder=OneHotEncoder(drop='first')
catogorical_value=['gender','category','state']
train_encoded=encoder.fit_transform(df_train[catogorical_value]).toarray()
test_encoded=encoder.fit_transform(df_test[catogorical_value]).toarray()


# In[22]:


scaler=StandardScaler()
numeric=['amt', 'lat', 'long','city_pop', 'unix_time', 'merch_lat', 'merch_long']
sc_train=scaler.fit_transform(df_train[numeric])
sc_test=scaler.fit_transform(df_test[numeric])


# In[23]:


final_train= pd.concat([pd.DataFrame(train_encoded), pd.DataFrame(sc_train)], axis=1)
final_test = pd.concat([pd.DataFrame(test_encoded), pd.DataFrame(sc_test)], axis=1)


# In[24]:


train_target=df_train['is_fraud']
test_target=df_test['is_fraud']


# In[25]:


# here we can either use dummy values to balance or SMOtE to balance teh data sets


# In[26]:


smote=SMOTE(random_state=36) # or we can use 42 as according 
x_train_resample,y_train_resample=smote.fit_resample(final_train,train_target)


# In[27]:


x_shuffled,y_shuffled=shuffle(x_train_resample,y_train_resample,random_state=42)


# In[28]:


x_train,x_test,y_train,y_test=train_test_split(x_shuffled,y_shuffled,test_size=0.5)


# In[29]:


# now for validation lets have a copy of train and test data known has copytrain and copytest


# In[30]:


x_copy_train=x_train
x_copy_test=x_test


# In[31]:


x_train.shape


# In[32]:


y_train.shape


# In[33]:


x_test.shape


# In[36]:


y_test.shape


# In[35]:


Lreg=LogisticRegression()


# In[36]:


Lreg.fit(x_train,y_train)


# In[37]:


Lreg_pred=Lreg.predict(x_test)


# In[38]:


Lreg_accuracy = accuracy_score(y_test, Lreg_pred)


# In[39]:


print(Lreg_accuracy)


# In[40]:


cm=confusion_matrix(y_test,Lreg_pred)


# In[41]:


print(cm)
plt.figure(figsize=(8,4))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title("final predicted values")
plt.xlabel("confusion matrix")
plt.ylabel("actual")
plt.show()


# In[42]:


x_train = x_train[:30000]
y_train = y_train[:30000]


# In[46]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(rf_accuracy)


# In[47]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
knn_pred = knn_model.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print(knn_accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




