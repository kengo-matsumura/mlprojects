# Machine Learning - Classification
# Code examples are inspired from the DataCamp course 'Supervised Learning with Scikit-Learn'.
# Working with a variety of datasets that are available in the sklearn.datasets library.

# Load required packages

# numpy, or Numerical Python, is a very useful package for data science, and is essential for most tasks.
import numpy as np

# matplotlib is essential for plotting graphs.
import matplotlib.pyplot as plt

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

# This is not a very good accuracy score - can we do better?
# Let's plot a model complexity curve on the training and testing data with different numbers of neighbors.
# We are essentially performing 'hyperparameter tuning', which is the process of tuning or changing the hyperparameters
# in a function to achieve a more optimal result. A more robust approach would be to use a grid search or random search
# that is available in the sklearn.model_selection package.

def plotKNN(start_number = 1, num_iter = 10):
  ''' This function generates the training and testing accuracy scores for a specified number of iterations
      and plots the model complexity curve.
  '''
  
  # Create the range of n_neighbors to iterate through.
  neighbors = np.arange(start_number, start_number + num_iter)
  
  # Create empty numpy arrays for storing the training and testing accuracies later.
  train_accuracy = np.empty(len(neighbors))
  test_accuracy = np.empty(len(neighbors))

  for i, k in enumerate(neighbors):
    
    # Instantiate the knn model for k
    knn = KNeighborsClassifier(k)
    
    # Fit the data to the model
    knn.fit(X_train, y_train)
    
    # Evaluate the accuracy score and add it to the training and testing arrays
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
  
  # Now that the accuracy scores are stored in the arrays, let's plot them against the different values of n_neighbors:
  plt.title('k-NN: Accuracies for varying number of neighbors')
  plt.plot(neighbors, train_accuracy, label = 'Training accuracy')
  plt.plot(neighbors, test_accuracy, label = 'Testing accuracy')
  plt.legend()
  plt.xlabel('Number of neighbors')
  plt.ylabel('Accuracy')
  plt.show()
  
# The defined function plotKNN will generate a plot for neighbors 1-9 by default.
# If we run plotKNN(1, 50), we can see that the optimal number of neighbors for performing on unseen data seems to be 13 or 33.
plotKNN(1, 50)

# Let's see the accuracies:
knn = KNeighborsClassifier(n_neighbors = 13)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
# >>> 0.8222222222222222

knn = KNeighborsClassifier(n_neighbors = 33)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
# >>> 0.8444444444444444

# We can see that we can get the accuracy up to as high as 84% by tuning the hyperparameter.

# The next question is though, will this consistently produce high accuracies on different variations of the train and test split?
# If we change the random_state in the train_test_split function, you can see that the plots look very different.
# What is the next step?
