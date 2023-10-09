import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Generate a random dataset for binary classification
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3 and 4: Train the RBF kernel SVC with different values of C and record the test accuracy
C_values = np.logspace(-3, 12, num=15)  # Vary C from 10^(-3) to 10^3
test_accuracies = []

for C in C_values:
    model = SVC(kernel='rbf', C=C, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_accuracies.append(test_accuracy)

# Step 5: Plot the results using seaborn
sns.set_style('whitegrid')
plt.figure(figsize=(8, 6))
plt.plot(C_values, test_accuracies, marker='o')
plt.xscale('log')
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Test Accuracy')
plt.title('RBF SVM Test Accuracy vs. Regularization Parameter (C)')
plt.show()
