from preprocessing import preprocess_data
from Naive_Bayes_Implementation import NaiveBayes

import numpy as np

x_train, x_test, y_train, y_test = preprocess_data() # Preprocess the data

model = NaiveBayes() # Create an instance of the Naive Bayes classifier
model.fit(x_train, y_train) # Fit the model to the training data

y_pred = model.predict(x_test) # Predict the class labels for the test data
accuracy = np.mean(y_pred == y_test) # Calculate the accuracy of the model
print(f"Accuracy: {accuracy * 100:.2f}%") # Print the accuracy of the model

TP = np.sum((y_pred == 1) & (y_test == 1)) # True Positives
TN = np.sum((y_pred == 0) & (y_test == 0)) # True Negatives
FP = np.sum((y_pred == 0) & (y_test == 1)) # False Positives
FN = np.sum((y_pred == 1) & (y_test == 0)) # False Negatives
print(f"True Positives: {TP}") # Print True Positives
print(f"True Negatives: {TN}") # Print True Negatives
print(f"False Positives: {FP}") # Print False Positives
print(f"False Negatives: {FN}") # Print False Negatives
print(f"Precision: {TP / (TP + FP) * 100:.2f}%") # Print Precision
print(f"Recall: {TP / (TP + FN) * 100:.2f}%") # Print Recall
print(f"F1 Score: {2 * ((TP / (TP + FP) * TP / (TP + FN)) / ((TP / (TP + FP) + TP / (TP + FN))) * 100):.2f}%") # Print F1 Score
