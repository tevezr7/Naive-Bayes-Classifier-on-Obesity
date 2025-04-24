import pandas as pd
import numpy as np

def preprocess_data():
    df = pd.read_csv('data.csv') # Load the dataset

    df = df.drop(columns=['MTRANS']) # Drop transportation column as it is not needed for the analysis, we will only be using ordinal to work best with naive bayes


    ordinal_map = {
        'no': 0,
        'yes': 1,
        'Female': 0,
        'Male': 1,
        'Sometimes': 1,
        'Frequently': 2,
        'Always': 3,
    } # Map ordinal values to integers
    ordinal_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC','CAEC', 'CALC'] # List of ordinal columns
    for col in ordinal_columns:
        df[col] = df[col].map(ordinal_map) # Map the ordinal values to integers

    class_map = { 
        'Normal_Weight': 0,
        'Overweight_Level_I': 0,
        'Overweight_Level_II': 0,
        'Obesity_Type_I': 1,
        'Obesity_Type_II': 1,
        'Obesity_Type_III': 1,
        'Insufficient_Weight': 0,
    }# Map class values to integers

    class_column = 'NObeyesdad' # Class column name
    df[class_column] = df[class_column].map(class_map) # Map the class values to 

    x = df.drop(columns=[class_column]).values # Features
    y = df[class_column].values # Target variable

    x_mean = np.mean(x, axis=0) # Mean of the features
    x_std = np.std(x, axis=0) + 1e-9 # Standard deviation of the features
    x_normalized = (x - x_mean) / x_std # Standardize the features

    print("Class distribution:", np.unique(y, return_counts=True)) # Print the class distribution

    print("Mean of features:", x_mean) # Print the mean of the features

    print("Standard deviation of features:", x_std) # Print the standard deviation of the features

    print("Normalized features:", x_normalized) # Print the normalized features
    print("Shape of features:", x_normalized.shape) # Print the shape of the features
    print("Shape of target variable:", y.shape) # Print the shape of the target variable

    #train test split manual

    data = np.hstack((x_normalized, y.reshape(-1, 1))) # Combine features and target variable
    np.random.seed(1) # Set random seed
    np.random.shuffle(data) # Shuffle the data

    x_shuffled = data[:, :-1] # Features
    y_shuffled = data[:, -1].astype(int) # Target variable

    train_size = int(0.8 * len(data)) # Train size

    x_train = x_shuffled[:train_size] # Training features
    x_test = x_shuffled[train_size:] # Testing features
    y_train = y_shuffled[:train_size] # Training target variable
    y_test = y_shuffled[train_size:] # Testing target variable

    print("Shape of training features:", x_train.shape) # Print the shape of the training features
    print("Shape of testing features:", x_test.shape) # Print the shape of the testing features
    print("Shape of training target variable:", y_train.shape) # Print the shape of the training target variable
    print("Shape of testing target variable:", y_test.shape) # Print the shape of the testing target variable
    print("Class distribution in training set:", np.unique(y_train, return_counts=True)) # Print the class distribution in the training set
    print("Class distribution in testing set:", np.unique(y_test, return_counts=True)) # Print the class distribution in the testing set
    print("Training set size:", len(x_train)) # Print the training set size
    print("Testing set size:", len(x_test)) # Print the testing set size
    return x_train, x_test, y_train, y_test
