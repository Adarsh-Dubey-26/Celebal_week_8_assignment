# -*- coding: utf-8 -*-
"""Celebal_Week_8_assignmennt

Original file is located at
    https://colab.research.google.com/drive/1iL8yhS2Vf_HCurvCkScxdRK58HZ9r6fo
"""

pip install pandas faiss-cpu openai sentence-transformers

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/content/Training Dataset.csv')

df.head()

df.info()

df.describe()

df.shape

df.columns

missing_values = df.isnull().sum()

missing_columns = missing_values[missing_values > 0]

missing_columns

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

df.isnull().sum()

num = df.select_dtypes(include=['int64' , 'float64'])
num

for i in df.columns :
  x = df[i].value_counts()
  print(f'{i} = {len(x)}')

Cat = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']]
Cat

Cor = num.corr()
Cor

plt.figure(figsize=(8, 6))
sns.heatmap(Cor, annot=True, linewidths=0.6)
plt.title('Heatmap of Correlation Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount', hue='Loan_Status')
plt.title('Applicant Income vs Loan Amount by Loan Status')
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='LoanAmount', y='Credit_History', hue='Loan_Status')
plt.title('Loan Amount vs Credit History by Loan Status')
plt.xlabel('Loan Amount')
plt.ylabel('Credit History')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='CoapplicantIncome', y='Loan_Amount_Term', hue='Loan_Status')
plt.title('Coapplicant Income vs Loan Amount Term by Loan Status')
plt.xlabel('Coapplicant Income')
plt.ylabel('Loan Amount Term')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='ApplicantIncome', hue='Loan_Status', kde=True)
plt.title('Histogram: Loan_Status vs ApplicantIncome')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='LoanAmount', hue='Loan_Status', kde=True)
plt.title('Histogram: Loan_Status vs LoanAmount')
plt.tight_layout()
plt.show()

num.plot (kind = 'box' , subplots= True , figsize=(15,5) , layout = None)

for i in Cat.columns :
  x = Cat[i].value_counts()
  plt.figure(figsize=(8,6))
  plt.pie( x , labels=x.index , autopct='%1.1f%%' , wedgeprops= {'edgecolor': 'white'} )
  plt.title(i)

cancellation_comparison= pd.crosstab(df['Married'] , df['Loan_Status'])
cancellation_comparison.plot(kind='bar', color=['#4CAF50', '#F44336'])
plt.title('Count Plot: Loan_Status vs Married')
plt.ylabel('count')
plt.show()

cancellation_comparison= pd.crosstab(df['Education'] , df['Loan_Status'])
cancellation_comparison.plot(kind='bar',color=['#ffbb78', '#aec7e8'])
plt.title('Count Plot: Loan_Status vs Education')
plt.ylabel('count')
plt.show()

Cat.columns

plt.figure(figsize=(8, 6))
sns.countplot(x='Loan_Status', hue='Gender', data=df)
plt.title('Count Plot: Loan_Status vs Gender')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Loan_Status', hue='Property_Area', data=df)
plt.title('Count Plot: Loan_Status vs Property_Area')
plt.show()

# Question 1 : What is the relationship between the applicant’s income and the requested loan amount ?

Cor

sns.heatmap(Cor, annot=True)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df , x='ApplicantIncome', y='LoanAmount' )
plt.title("Scatter plot between Applicant Income and Loan Amount")
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.show()

# Question 2 : What is the impact of credit history on the likelihood of loan approval?

plt.figure(figsize=(8, 6))
sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
plt.title("Loan Approval Based on Credit History")
plt.xlabel("Credit History")
plt.ylabel("Count of Loans")
plt.show()

credit_history_percentage = df.groupby('Credit_History')['Loan_Status'].value_counts(normalize=True) * 100
credit_history_percentage
