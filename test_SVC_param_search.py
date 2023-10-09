import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC


# Function to perform hyperparameter search and return the optimal values of C and gamma
def hyperparameter_search(X_train, y_train):
    param_grid = {'C': np.logspace(-3, 3, num=7), 'gamma': np.logspace(-3, 3, num=7)}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_['C'], grid_search.best_params_['gamma']


# Number of datasets for hyperparameter search
num_datasets = 14
num_splits = 5

# Lists to store optimal values of C and gamma for each dataset
optimal_C_values = []
optimal_gamma_values = []

for _ in range(num_datasets):
    # Step 1: Generate a random dataset for binary classification
    X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=None)

    # Lists to store optimal values of C and gamma for each split
    split_optimal_C_values = []
    split_optimal_gamma_values = []

    for _ in range(num_splits):
        # Step 2: Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

        # Step 3: Perform hyperparameter search and get the optimal values of C and gamma for this split
        optimal_C, optimal_gamma = hyperparameter_search(X_train, y_train)

        # Step 4: Append the optimal values to the lists for this split
        split_optimal_C_values.append(optimal_C)
        split_optimal_gamma_values.append(optimal_gamma)

    # Append the optimal values for all splits of this dataset to the main lists
    optimal_C_values.extend(split_optimal_C_values)
    optimal_gamma_values.extend(split_optimal_gamma_values)

# Step 5: Plot the optimal values of C and gamma using box and whisker plots
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))

# Box and whisker plot for optimal values of C
plt.subplot(1, 2, 1)
sns.boxplot(data=optimal_C_values, color='skyblue')
plt.yscale('log')
plt.ylim(10 ** -3, 10 ** 3)
plt.xlabel('Optimal Value of C')
plt.title('Box and Whisker Plot for Optimal Values of C')

# Box and whisker plot for optimal values of gamma
plt.subplot(1, 2, 2)
sns.boxplot(data=optimal_gamma_values, color='lightgreen')
plt.yscale('log')
plt.ylim(10 ** -3, 10 ** 3)
plt.xlabel('Optimal Value of gamma')
plt.title('Box and Whisker Plot for Optimal Values of gamma')

plt.tight_layout()
plt.show()
