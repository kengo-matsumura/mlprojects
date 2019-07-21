# Machine Learning - Classification
# Code examples are inspired from the DataCamp course 'Supervised Learning with Scikit-Learn'.
# Working with a variety of datasets that are available in the sklearn.datasets library.

# Load required packages

# numpy, or Numerical Python, is a very useful package for data science, and is essential for most tasks.
import numpy as np

# sklearn, or Scikit-Learn, is another useful package for data science which abstracts a lot of the low-level work required
# to create models. It also contains famous sample datasets such as the iris and MNIST datasets.
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Let's start with importing the wine dataset from sklearn.datasets
# Handy tip - after the period operator, press Tab to bring up a list of possible class objects.
wines = datasets.load_wine()

# The sklearn datasets are stored as a Bunch type, which is similar to a dict and contains keys:
wines.keys()
# >>> dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

# Let's store the data in a matrix X, and the target column in a vector y:
X = wines.data
y = wines.target

# Look at the dimensions:
X.shape
# >>> (178, 13)
y.shape
# (178,)

# Split the data into training and testing sets using the train_test_split function from sklearn.
# This is an important part of the process in order to be able to see how our model generalises to unseen data.
# If we run the model on test data which was part of the training data, then we are likely to see a higher accuracy
# because the model has already seen it before. When this does not generalise to new data, this is called overfitting.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10, stratify = y)
# test_size is an argument that determines the split - in this case, we have split 75% of the data into the training set,
# and 25% of the data into the testing set.
# The random_state argument is a seed which allows us to pseudo-randomly sample data for our training and testing sets.
# Pseudo-random means that it isn't exactly random, and the seed allows us to generate the same training and testing sets.


# Instantiate a knn object using the KNeighborsClassifier which we loaded earlier:
knn = KNeighborsClassifier(n_neighbors = 5)
# n_neighbors is a hyperparameter that determines how many neighboring points to consider when determining the outcome.
# We will briefly delve into hyperparameter tuning later when observing the model complexity curves.

# Fit the knn model to the training data:
knn.fit(X_train, y_train)

# Evaluate the accuracy of the model on unseen data:
knn.score(X_test, y_test)
# >>> 0.6888888888888889
