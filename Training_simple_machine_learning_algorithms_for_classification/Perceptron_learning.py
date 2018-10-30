"""

Perceptron_learning algorithm in Python
author: Sergio Masa Avís
git-author: https://github.com/sergioma295
Reference Book: Python Machine Learning 2nd Edition by Sebastian Raschka] (https://sebastianraschka.com), Packt Publishing Ltd. 2017
Project Interpeter: 3.6.3 anaconda3 python
"""

# Libraries
import numpy as np              # https://sebastianraschka.com/pdf/books/dlb/appendix_f_numpy-intro.pdf
                                # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html?highlight=randomstate
import pandas as pd             # https://pandas.pydata.org/pandas-docs/stable/10min.html
import matplotlib.pyplot as plt # http://matplotlib.org/users/beginner.html
from matplotlib.colors import ListedColormap


class Perceptron(object):
    """ Perceptron classifier.
    Definition of the perceptron as a Python class. It allows us to use a objetct
    that can learn from data via a fit method and make predictions via a separate predict method.
    As a convention, we append an underscore (_) to attributes that are not being created upon the initialization
    of the object but by calling the object's other methods.

    Initialization the wights in self.w_ to a vector R^(m+1) where m standos for the number of dimensions(features) in the dataset
    We add 1 for the first element in this vector that represents the bias unit. self.w_[0] represents the bias unit.
    The reason we don't initialize the weights to zero is that the learning rate eta only has
    an effect on the classification outcome if the weights are initialized too non-zero values. If all the wights are initialized to zero, the
    learning rate parameter eta affects only the scale of the weight vector, not the direction.

    After the weights have been iitialized, the fit method loops over all individual samples in the training set and updates the weigths according to the perceptron learning rule.
    The class labels are predicted by the predict method, which is called in the fit method to predict the class label for the weight update, but predict can also be sued
    to predict the class labels of new data after we have fitted our model.

    ---------------
    | Parameters: |
    ---------------
    Learning rate, eta: it's a float between 0.0 and 1.0
    Number of iteration, n_iter: it's a int which refer passes over the training dataset. Number of epochs
    Random number generator seed for random weight initialization, random_state: int

    ---------------
    | Attributes: |
    ---------------
    Weights after fitting, w_: 1d array
    Number of misclassfications (updates) in each epoch, errors_ : list

    """
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        """ Constructor Perceptron class
        Parameter list:
        eta = 0.01
        n_iter = 50
        random_state = 1
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state


    def fit(self,X,y):
        """ Fit training data

        :param X: Training vectors -> {array-like}, shape = [n_samples, n_features]
        :param y: Target values    -> array-like, shape = [n_samples]
        :return: self: object

        """
        # separate RandomState objects can also be useful if we run our code in non-sequential order
        # If we want to visualize rgen. rgen.rand(number of numbers)
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        # numpy.random.normal(loc=0.0,scale=1.0,size=None). Draw random samples form a normal(Gaussian) distribuition.
        #    loc : float or array_like of floats
        #          Mean (“centre”) of the distribution.

        #    scale : float or array_like of floats
        #            Standard deviation (spread or “width”) of the distribution.

        #    size : int or tuple of ints, optional
        #            Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if loc and scale are both scalars. Otherwise, np.broadcast(loc, scale).size samples are drawn.

        # Returns:

        #          out : ndarray or scalar

        #          Drawn samples from the parameterized normal distribution.

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input
        np.dot function that is used in the net_input mehod simply calculates the vector dot product wtx
        """

        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)






# ## Training a perceptron model on the Iris dataset

# ...

# ### Reading in the Iris data




df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data', header=None)
df.tail()



#
# ### Note:
#
#
# You can find a copy of the Iris dataset (and all other datasets used in this book) in the code bundle of this book, which you can use if you are working offline or the UCI server at https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data is temporarily unavailable. For instance, to load the Iris dataset from a local directory, you can replace the line
#
#     df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#         'machine-learning-databases/iris/iris.data', header=None)
#
# by
#
#     df = pd.read_csv('your/local/path/to/iris.data', header=None)
#



# df = pd.read_csv('iris.data', header=None)
# df.tail()




# ### Plotting the Iris data

# ### Training the perceptron model
# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.figure('Visual Data and Number updates vs Epochs')
plt.subplot(211)


# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

#
plt.subplot(212)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.savefig('Plot_images/Data_Visual_and_Number_updates_vs_Epochs', dpi=300)
plt.show()

# ### A function for plotting decision regions




