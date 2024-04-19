# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:19:03 2024

@author: Tom Tremerel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#import datas
df = pd.read_excel('E Commerce Dataset.xlsx','E Comm')
#check the size of the data set
df.shape
#check if all values are consistent
DD = df.describe().T
df.head()
df.dtypes
#check if there are some null values, and indeed there are
df.isnull().sum()
#check if we have duplicated values
df.duplicated().sum()
df = df.drop_duplicates()

#cleanling of the data with the names of the columns 
column = ['Tenure','WarehouseToHome','HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CouponUsed','OrderCount','DaySinceLastOrder']

for col in column :
    median = df[col].median()
    df[col].fillna(median, inplace=True)

df.isnull().sum()

df = df.drop('CustomerID', axis = 1)

#EDA
plt.figure(figsize=(10,5))
sns.histplot(df['SatisfactionScore'],bins = 10)

Payment_count = df["PreferredPaymentMode"].value_counts()
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.set_palette('pastel')
colors = sns.color_palette()
plt.pie(Payment_count,colors = colors,autopct='%1.1f%%', labels = Payment_count.index, shadow = True)
plt.title("méthodes de paiement")

Device_count = df['PreferredLoginDevice'].value_counts()
plt.figure(figsize=(10,4))
plt.pie(Device_count,labels=Device_count.index, autopct='%1.1f%%')
#I just replace mobile phone by phone, cause it's the same
df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace('Mobile Phone','Phone')

Sexe_count = df['Gender'].value_counts()
plt.figure(figsize=(10,4))
plt.pie(Sexe_count, labels=Sexe_count.index, autopct='%1.1f%%')

plt.figure(figsize=(10,5))
plt.scatter(df['PreferredLoginDevice'], df['Churn'])

df = df.drop('MaritalStatus')

df.columns
count_col = []
hist_col = []

for column in df.columns: 
    if df[column].nunique() <= 20:
        count_col.append(column)
    else:
        hist_col.append(column)

nb_count = len(count_col)
nb_hist = len(hist_col)
plot_num= 1
plt.figure(figsize=(10,35))
for col in count_col  :
    plt.subplot(nb_count,2,plot_num)     
    sns.countplot(data = df, x= col)
    plot_num+= 1
    plt.tight_layout()

plot_num= 1
plt.figure(figsize=(10,15))
for col in hist_col :
    plt.subplot(4,2,plot_num)
    sns.countplot(data = df, x= col)
    plot_num+=1
    plt.tight_layout()


plot_num= 1
plt.figure(figsize=(10,35))
for col in count_col  :
    if df[col].nunique() <= 5 and col != "Churn":
        plt.subplot(nb_count,2,plot_num)     
        sns.countplot(data = df, x= col, hue="Churn")
    plot_num+= 1
    plt.tight_layout()

plot_num=1
plt.figure(figsize=(10,35))
for col in count_col :
    if df[col].nunique() <= 8 and col != "Churn" :
        plt.subplot(10, 2, plot_num)
        plt.pie(data = df, x= df[col].value_counts(),autopct='%1.1f%%',labels= df[col].value_counts().index, wedgeprops=dict(width=0.8,edgecolor="w"), shadow=True )
    plot_num+= 1
    plt.tight_layout()
    
#%%

# Building model

cat_df = df.select_dtypes(include="O")    

X = df.drop(columns=["Churn"])
y = df["Churn"]
    
cat_col = X.select_dtypes(include="O").columns
num_col = []



X.columns
for column in X.columns:
    if column not in cat_col:
        num_col.append(column)
        
categorical_col = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])

numerical_col = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

transformer_data = ColumnTransformer([
    ('categorical_col', categorical_col, cat_col),
    ('numerical_col', numerical_col, num_col)
])
classifier = XGBClassifier()

model = Pipeline([
    ('transformer', transformer_data),
    ('classifier', classifier)
])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

train_acc = []
val_acc = []

for i in range(100):
    
    model.fit(X_train, y_train)

    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    
    train_acc.append(train_accuracy)
    val_acc.append(test_accuracy)

plt.figure(figsize=(8, 6))
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Test Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Courbe d\'accuracy au cours de l\'entraînement')
plt.legend()
plt.show()
print(train_accuracy)
print(test_accuracy)
        

        