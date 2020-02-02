# -*- coding: utf-8 -*-
"""
Created on Wed Sep  25 22:55:20 2019

@author: Amenallah 
"""
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


#Read the data
df=pd.read_csv('news.csv')
#Get shape and head
print('the shape is: ', df.shape)
print("##################################################")
print('first five lines ' , df.head())
print("##################################################")

#Get labels
labels=df.label
print('the labels are as' , labels.head())
print("##################################################")

x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


#now we Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#and then Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)



#initializatin of PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#pPredict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print("##################################################")
print(f'Accuracy: {round(score*100,2)}%')




#print out a confusion matrix to gain insight into the number of false and true negatives and positives
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])