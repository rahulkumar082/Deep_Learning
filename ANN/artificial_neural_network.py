# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:40:04 2020

@author: Rahul Kumar
"""

# Importing the Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Part 1: Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding the categorical data
# Label Encoding Gender Column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One Hot Encoding Geography Column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into test set and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2: Building the ANN

# Initialize the ANN
ann = tf.keras.models.Sequential()

# Add the input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Add the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Add the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
