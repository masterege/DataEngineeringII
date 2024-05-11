import matplotlib.pyplot as plt
import numpy as np
#SK-Learn
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



data = fetch_covtype()

X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Validation set
#X_train_val, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Print the shapes of the partitions
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

#RF Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

#RF Score
score = model.score(X_test, y_test)
print(f"Accuracy of the RF Classifier with default parameters: {score}")

# Explore the parameters used in the Random Forest implementation
print(f"Parameters used in the RandomForestClassifier: \n{model.get_params()}")

