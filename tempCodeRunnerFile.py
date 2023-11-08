# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset from Scikit-learn
iris = datasets.load_iris_excel(gg.xlsx)
X = iris.data()  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target (species: 0-setosa, 1-versicolor, 2-virginica)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create a K-Nearest Neighbors (KNN) classifier
k = 3  # You can adjust this value to choose the number of neighbors
classifier = KNeighborsClassifier(n_neighbors=k)

# Train the KNN classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the classifier's performance
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)

# Display the results
print("Confusion Matrix:")
print(confusion_mat)
print("\nClassification Report:")
print(classification_rep)
