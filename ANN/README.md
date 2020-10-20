# ANN (Artificial Neural Network)

**Problem Statement**: Given is the dataset for Bank Customers ("Churn_Modelling.csv"). We have to consider all possible columns and predict whether the customer will leave the bank or not?
## Step-by-step Approach

### Import Libraries such as:
* numpy
* pandas
* tensorflow
* Label Encoder (encode target labels with value between 0 and n_classes-1, such as Gender)
* Column Transformer (allows different columns or column subsets of the input to be transformed separately)
* One Hot Encoder (encode categorical features as a numeric array)
* Standard Scalar

### Part 1: Data Preprocessing
* Import the dataset using pandas.read_csv
* Encode the "Gender" column using Label Encoder
* Encode the "Geography" column using One Hot Encoder
* Split the dataset to Training and Test Set
* Feature Scaling (this is always necessary while training a neural network to normalize the range of independent variables) using Standard Scalar
