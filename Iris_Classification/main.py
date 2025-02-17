import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Dataset details
print("Iris dataset keys:", iris.keys())
print("Target names:", iris.target_names)
print("Feature names:", iris.feature_names)
print("Shape of data:", X.shape)
print("First five samples:\n", X[:5])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Visualization - Scatter plot of first two features
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Iris Data - First Two Features")
plt.show()

# Display sample predictions
sample_idx = np.random.choice(len(X_test), 5, replace=False)
for i in sample_idx:
    print(f"Sample {i}: True Label = {iris.target_names[y_test[i]]}, Predicted = {iris.target_names[y_pred[i]]}")
