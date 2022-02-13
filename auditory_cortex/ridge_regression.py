import numpy as np

class RidgeRegression():
  # include solver parameter for flexible implementation of gradient descent
  # solution in future, alpha is used in place of lambda to mimic scikit-learn
  def __init__(self, alpha=1.0):
      self.alpha = alpha

  def fit(self, X, y):
    X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]

    self.X_intercept = X_with_intercept
    # number of columns in matrix of X including intercept
    dimension = X_with_intercept.shape[1]
        # Identity matrix of dimension compatible with our X_intercept Matrix
    A = np.identity(dimension)
    # set first 1 on the diagonal to zero so as not to include a bias term for
    # the intercept
    A[0, 0] = 0
    # We create a bias term corresponding to alpha for each column of X not
    # including the intercept
    A_biased = self.alpha * A
    # print(A_biased.shape)
    Beta = np.linalg.inv(X_with_intercept.T.dot(
        X_with_intercept) + A_biased).dot(X_with_intercept.T).dot(y)
    self.Beta = Beta
    return self

  def predict(self, X):
      Beta = self.Beta
      X_predictor = np.c_[np.ones((X.shape[0], 1)), X]
      self.predictions = X_predictor.dot(Beta)
      return self.predictions