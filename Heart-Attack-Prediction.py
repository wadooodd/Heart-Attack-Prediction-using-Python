#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Data Preprocessing Part 1

# In[2]:


df=pd.read_csv('Blood_Pressure_data.csv')


# In[3]:


print(df.shape)
print(df.dtypes)


# In[4]:


df.info()


# In[5]:


df.head(10).T


# In[6]:


df.tail(10).T


# # Exploratory Data Analysis

# In[7]:


print("Majority of the population is Caucasians followed by African American:")
df['cast'].value_counts().plot(kind='bar',color = list('ygbkrmc'))


# In[8]:


print("Gender data comprises of 55% male whereas 45% are female: ")
df['gender'].value_counts().plot(kind='bar',color = list('bgrkymc'))


# In[9]:


print("Details of age data are as follows:")
df['age group'].value_counts().plot(kind='bar',color = list('ygbkrmc'))


# In[10]:


print("Majority of weight data was missing so we had to drop that column:")
df['weight'].value_counts().plot(kind='bar',color = list('ygbkrmc'))


# In[11]:


print("Max Glu Serum:")
df['max_glu_serum'].value_counts().plot(kind='bar',color = list('ygbkrmc'))


# In[12]:


print("A1C Result:")
df['A1Cresult'].value_counts().plot(kind='bar',color = list('gybcrkm'))


# In[13]:


print("There is change in medicines for almost 40–45% of the patients:")
df['change'].value_counts().plot(kind='bar',color = list('gybcrkm'))


# In[14]:


print("It seems that almost 75% of the patients were taking the medicines:")
df['Med'].value_counts().plot(kind='bar',color = list('ybgcrkm'))


# In[15]:


print("Time spent in  hospital:")
df['time_in_hospital'].value_counts().plot(kind='line',color = 'y')


# In[16]:


print("We can see from the following scatter plot that how random is the data of diagnosis columns:")
plt.figure(figsize=(30, 15))
diag1=df['diag_1']
diag2=df['diag_2']
plt.scatter(diag1,diag2)
plt.show()


# In[17]:


print("Insulin dosage:")
plt.figure(figsize=(15, 7))
plt.xlabel("Insulin")
df['insulin'].value_counts().plot(kind='pie',autopct="%0.2f%%")


# # Data Preprocessing Part 2

# In[18]:


print(df['label'].value_counts())


# In[19]:


df=df.drop(['id','patient_no','weight','payer_code','medical_specialty'],1)


# In[20]:


df.head(10).T


# In[21]:


df['gender'].value_counts()


# In[22]:


df=df[df['gender']!='Unknown/Invalid']


# In[23]:


df['gender'].value_counts()


# In[24]:


df['age group'].value_counts()


# In[25]:


df['age group'] = df['age group'].str[1:].str.split('-',expand=True)[0]
df['age group'] = df['age group'].astype(int)


# In[26]:


df['age group'].value_counts()


# In[27]:


df.head(20).T


# In[28]:


df = df.drop(['citoglipton','examide'],1)


# In[29]:


df.head().T


# In[30]:


for col in df.columns:
    if df[col].dtype == 'int64':
         print(col,df[col][df[col] == '?'].count())


# In[31]:


for col in df.columns:
    if df[col].dtype == object:
         print(col,df[col][df[col] == '?'].count())


# In[32]:


df=df.fillna(0)
df=df.replace(['?'],0)


# In[33]:


df.head(10).T


# In[34]:


df['label'] = df['label'].replace('>5', 1)
df['label'] = df['label'].replace('<30', 1)
df['label'] = df['label'].replace('NO', 0)


# In[35]:


diag_cols = ['diag_1','diag_2','diag_3']
for col in diag_cols:
    df[col] = df[col].str.replace('E','-')
    df[col] = df[col].str.replace('V','-')


# In[36]:


a={'None':0,'Norm':100,'>200':200,'>300':300}
df["max_glu_serum"]=df["max_glu_serum"].map(a)


# In[37]:


b={'None':0,'Norm':5,'>7':7,'>8':8}
df["A1Cresult"]=df["A1Cresult"].map(b)


# In[38]:


c={'No':-1,'Ch':1}
df["change"]=df["change"].map(c)


# In[39]:


d={'No':-1,'Yes':1}
df["Med"]=df["Med"].map(d)


# In[40]:


tests_cols = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide',
 'tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','insulin','glyburide-metformin',
'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']
replace_words={'Up':10,'Down':-10,'Steady':0,'No':-20}
for col in tests_cols:
    df[col] = df[col].replace(replace_words)


# In[41]:


e={'Male':0,'Female':1}
df["gender"]=df["gender"].map(e)


# In[42]:


f={'Caucasian':0,'AfricanAmerican':1,'Hispanic':2,'Other':2,'Asian':2}
df["cast"]=df["cast"].map(f)


# In[43]:


df.dropna(inplace=True)


# In[44]:


df['cast'].value_counts()


# In[45]:


df['gender'].value_counts()


# In[46]:


df.head(10).T


# In[47]:


df['cast'] = df['cast'].astype('Int64')


# In[48]:


df


# # Modelling

# In[49]:


X=df.drop("label",1)
X.dtypes


# In[50]:


X=pd.get_dummies(X)
X


# In[51]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print (X.head(10).T)


# In[52]:


y=df['label']
y


# ## Logistic Regression

# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,LogisticRegression
from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
logit = LogisticRegression(C=1, penalty='l1', solver='liblinear')
logit.fit(X_train, y_train)


# In[54]:


print('Shape of X_train = ', X_train.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of y_test = ', y_test.shape)


# In[55]:


logit_pred = logit.predict(X_test)

pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(logit_pred, name = 'Predict'), margins = True)


# In[65]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
print ("Accuracy : " , accuracy_score(y_test,logit_pred)*100)                                                          
print("Report : \n", classification_report(y_test, logit_pred))
print("F1 Score : ",f1_score(y_test, logit_pred, average='macro')*100)

Accuracy_lg=(accuracy_score(y_test,logit_pred)*100)
F1_Score_lg=(f1_score(y_test, logit_pred, average='macro')*100)


# ## Decision Tree 

# In[57]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=28, criterion = "entropy", min_samples_split=1000)
dtree.fit(X_train, y_train)


# In[58]:


dtree_pred = dtree.predict(X_test)
pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(dtree_pred, name = 'Predict'), margins = True)


# In[66]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
print ("Accuracy : " , accuracy_score(y_test,dtree_pred)*100)  
print("Report : \n", classification_report(y_test, dtree_pred))
print("F1 Score : ",f1_score(y_test, dtree_pred, average='macro')*100)

Accuracy_dt=(accuracy_score(y_test,dtree_pred)*100)
F1_Score_dt=(f1_score(y_test, dtree_pred, average='macro')*100)


# ## Random Forest

# In[61]:


from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
clf1=RandomForestClassifier()
clf1.fit(X_train,y_train)
pred=clf1.predict(X_test)
clf1.score(X_test,y_test)


# In[62]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)

Accuracy_rf=(accuracy_score(y_test,pred)*100)
F1_Score_rf=(f1_score(y_test, pred, average='macro')*100)


# In[63]:


plt.figure(figsize=(15, 8))
ax = plt.subplot(111)

models = ['Logistic Regression', 'Decision Tree', 'Random Forests']
values = [Accuracy_lg, Accuracy_dt, Accuracy_rf]
model = np.arange(len(models))

plt.bar(model, values, align='center', width = 0.17, alpha=0.7, color = 'yellow', label= 'accuracy')
plt.xticks(model, models)
           

           
ax = plt.subplot(111)

models = ['Logistic Regression', 'Decision Tree', 'Random Forests']
values = [F1_Score_lg, F1_Score_dt, F1_Score_rf]
model = np.arange(len(models))

plt.bar(model+0.15, values, align='center', width = 0.17, alpha=0.7, color = 'blue', label = 'F1 Score')
plt.xticks(model, models)

plt.ylabel('Machine Learning Models')
plt.title('Model Comparison')
    
# removing the axis on the top and right of the plot window
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()

plt.show()


# In[64]:


predictions = pd.DataFrame(columns=['LogisticRegression',
                                    'DecisionTree',
                                    'RandomForest'])
predictions['LogisticRegression'] = logit_pred
predictions['DecisionTree'] = dtree_pred
predictions['RandomForest'] = pred

# Lets take a look at the end of the dataframe
predictions.head(20).T


# In[ ]:




