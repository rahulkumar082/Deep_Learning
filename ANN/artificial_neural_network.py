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
from sklearn.metrics import confusion_matrix, accuracy_score

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


# Part 3: Training the ANN

# Compile the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the ANN on training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Take input from user
user_input = []
user_input.append(int(input("Credit Score? ")))
user_input.append(input("Geography? France / Spain / Germany "))
user_input.append(input("Gender? Male / Female "))
user_input.append(int(input("Age? In Years integer value ")))
user_input.append(int(input("Tenure? ")))
user_input.append(int(input("Balance? In Dollars ")))
user_input.append(int(input("Number of Products? ")))
user_input.append(int(input("Have Credit Card? Yes[1] / No[0] ")))
user_input.append(int(input("Is Active member? Yes[1] / No[0] ")))
user_input.append(int(input("Estimated Salary? In Dollars ")))
user_input[2] = int(le.transform([user_input[2]]))
user_input = np.array(ct.transform([user_input]))
user_input = sc.transform(user_input)

# Probability of the user / customer with above information will leave bank?
leave_probability = ann.predict(user_input)
print("Will this Customer leave the bank? " +
      str(leave_probability[0][0] > 0.5))


# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((
                        y_pred.reshape(len(y_pred), 1),
                        y_test.reshape(len(y_test), 1)
                      ), axis=1))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
print("Confusion Matrix: ", cm)
print("Accuracy Score: ", score)