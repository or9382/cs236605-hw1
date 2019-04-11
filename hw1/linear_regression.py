import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.model_selection import KFold, ParameterGrid


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = (X @ self.weights_).reshape((-1))
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # Use only numpy functions.

        w_opt = None
        # ====== YOUR CODE: ======
        reg = self.reg_lambda * (np.eye(X.shape[1], X.shape[1]))
        reg[0, 0] = 0
        w_opt = np.linalg.inv(X.T @ X + X.shape[0] * reg)
        w_opt = w_opt @ (X.T @ y.reshape((-1, 1)))
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X: A tensor of shape (N,D) where N is the batch size or of shape
            (D,) (which assumes N=1).
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X)

        xb = None
        # ====== YOUR CODE: ======
        if len(X.shape) > 1:
            xb = np.hstack((np.ones((X.shape[0], 1)), X))
        else:
            xb = np.hstack((np.ones(1), X))
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=3):
        self.degree = degree

        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)
        # check_is_fitted(self, ['n_features_', 'n_output_features_'])

        # Note: You can count on the order of features in the Boston dataset
        # (this class is "Boston-specific"). For example X[:,1] is the second
        # feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        new_features = X[:, [0, 6, 7, 12]].copy()
        new_features[:, 0] = 1.0 / np.power(new_features[:, 0], 2)
        new_features[:, 1] = np.log(100 - new_features[:, 1])
        new_features[:, 2] = np.log(new_features[:, 2])
        new_features[:, 3] = np.power(np.e, -new_features[:, 3])

        X_transformed = np.hstack((X, new_features))

        poly = PolynomialFeatures(self.degree)

        X_transformed = poly.fit_transform(X_transformed)

        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # ====== YOUR CODE: ======
    target_vals = df.loc[:, target_feature].values.copy()
    feat_vals = df.loc[:, df.columns != target_feature].values.copy()

    target_vals -= np.mean(target_vals)
    feat_vals -= np.mean(feat_vals, axis=0)

    feat_std = np.sqrt(np.sum(np.power(feat_vals, 2), axis=0))
    taret_std = np.linalg.norm(target_vals)

    cov = (feat_vals.T @ target_vals).reshape(-1)

    roh = cov / (feat_std * taret_std)

    top_indices = np.abs(roh).argsort()[-n:][::-1]
    top_n_features = df.columns[top_indices]
    top_n_corr = roh[top_indices]
    # ========================

    return top_n_features, top_n_corr


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    #
    # Notes:
    # - You can implement it yourself or use the built in sklearn utilities
    #   (recommended). See the docs for the sklearn.model_selection package
    #   http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    # - If your model has more hyperparameters (not just lambda and degree)
    #   you should add them to the search.
    # - Use get_params() on your model to see what hyperparameters is has
    #   and their names. The parameters dict you return should use the same
    #   names as keys.
    # - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    params = {'linearregressor__reg_lambda': lambda_range,
              'bostonfeaturestransformer__degree': degree_range}

    kf = KFold(n_splits=k_folds)
    best_params = ParameterGrid(params)[0]
    best_mse = np.inf
    best_r_2 = 0.0

    for p_dict in ParameterGrid(params):
        cur_acc = 0.0
        curr_r_2 = 0.0
        model.set_params(**p_dict)
        for train_index, test_index in kf.split(X):
            model.fit(X[train_index], y=y[train_index])
            mse, rsq = evaluate_accuracy(y[test_index], model.predict(X[test_index]))
            cur_acc += mse
            curr_r_2 += rsq

        cur_acc /= k_folds
        curr_r_2 /= k_folds

        if curr_r_2 > best_r_2:
            best_r_2 = curr_r_2
            best_params = p_dict
    # ========================

    return best_params


def evaluate_accuracy(y: np.ndarray, y_pred: np.ndarray):
    """
    Calculates mean squared error (MSE) and coefficient of determination (R-squared).
    :param y: Target values.
    :param y_pred: Predicted values.
    :return: A tuple containing the MSE and R-squared values.
    """
    mse = np.mean((y - y_pred) ** 2)
    rsq = 1 - mse / np.var(y)
    return mse.item(), rsq.item()

