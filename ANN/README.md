# ANN (Artificial Neural Network)

**Problem Statement**: Given is the dataset for Bank Customers ("Churn_Modelling.csv"). We have to consider all possible columns and predict whether the customer will leave the bank or not?
## Step-by-step Approach

### Import Libraries such as:
* numpy
* pandas
* tensorflow
* LabelEncoder (encode target labels with value between 0 and n_classes-1, such as Gender)
* ColumnTransformer (allows different columns or column subsets of the input to be transformed separately)
* OneHotEncoder (encode categorical features as a numeric array)
* StandardScalar

### Part 1: Data Preprocessing
* Import the dataset using pandas.read_csv
* Encode the "Gender" column using Label Encoder
* Encode the "Geography" column using One Hot Encoder
* Split the dataset to Training and Test Set
* Feature Scaling (this is always necessary while training a neural network to normalize the range of independent variables) using Standard Scalar

### Part 2: Building the ANN
* Initialise the ANN using Sequential from `tf.keras.models` library
* **Input layer** is added automatically on the basis of columns in the dataset
* **First Hidden Layer**:
  * Use `add `function of `Sequential` class for first hidden layer
    * For fully connected layer, we use `Dense` class from `tf.keras.layers` library inside `add` function
      * Parameters passed to `Dense` Class:
      * units -> Number of neurons/nodes. Use experimentation for better results/No thumb rule for this
      * activation -> The function attached to each neuron in the network, and determines whether it should be activated (use `relu`)
* **Second Hidden Layer**:
  * Follow same procedure as in `First Hidden Layer`. [Note: There is currently no theoretical reason to use neural networks with any more than two hidden layers. In fact, for many practical problems, there is no reason to use any more than one hidden layer]
* **Output Layer**:
  * Follow same procedure as in `First Hidden Layer`, except
    * In `Dense` class, use parameters in this way:
      * units -> Depends upon classes present in the output column, in the present dataset, it is either true or false, so use `1`
      * activation -> Use `sigmoid` function
