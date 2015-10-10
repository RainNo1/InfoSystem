from abc import ABCMeta, abstractmethod
import inspect
import itertools
from math import floor, ceil, sqrt
import sys
import time
import warnings
from scipy import linalg
from numpy.lib.stride_tricks import as_strided
from scipy import sparse
from sklearn.base import _pprint
from sklearn.externals.joblib.parallel import Parallel, delayed, cpu_count
from sklearn.linear_model import cd_fast
from sklearn.linear_model.base import center_data, sparse_center_data
from sklearn.linear_model.least_angle import LassoLars
from sklearn.linear_model.omp import orthogonal_mp_gram
from sklearn.metrics.metrics import r2_score
from sklearn.utils import gen_even_slices
from sklearn.utils.extmath import safe_sparse_dot, randomized_svd
from sklearn.utils.validation import array2d, safe_asarray, atleast2d_or_csc,\
    check_random_state

import numpy as np


class BaseEstimator(object):
    """Base class for all estimators in scikit-learn

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their __init__ as explicit keyword
    arguments (no *args, **kwargs).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        try:
            # fetch the constructor or the original constructor before
            # deprecation wrapping if any
            init = getattr(cls.__init__, 'deprecated_original', cls.__init__)

            # introspect the constructor arguments to find the model parameters
            # to represent
            args, varargs, kw, default = inspect.getargspec(init)
            if not varargs is None:
                raise RuntimeError('scikit learn estimators should always '
                                   'specify their parameters in the signature'
                                   ' of their init (no varargs).')
            # Remove 'self'
            # XXX: This is going to fail if the init is a staticmethod, but
            # who would do this?
            args.pop(0)
        except TypeError:
            # No explicit __init__
            args = []
        args.sort()
        return args

    def get_params(self, deep=True):
        """Get parameters for the estimator

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        """
        out = dict()
        for key in self._get_param_names():
            # catch deprecation warnings
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
            if len(w) and w[0].category == DeprecationWarning:
                # if the parameter is deprecated, don't show it
                continue

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of the estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.iteritems():
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if not name in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if not key in valid_params:
                    raise ValueError('Invalid parameter %s ' 'for estimator %s'
                                     % (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name),),)

    def __str__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name,
                           _pprint(self.get_params(deep=True),
                                   offset=len(class_name), printer=str,),)

class LinearModel(BaseEstimator):
    """Base class for Linear Models"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    def decision_function(self, X):
        """Decision function of the linear model

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Returns predicted values.
        """
        X = safe_asarray(X)
        return safe_sparse_dot(X, self.coef_.T) + self.intercept_

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Returns predicted values.
        """
        return self.decision_function(X)

    _center_data = staticmethod(center_data)

    def _set_intercept(self, X_mean, y_mean, X_std):
        """Set the intercept_
        """
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_std
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_.T)
        else:
            self.intercept_ = 0.

class RegressorMixin(object):
    """Mixin class for all regression estimators in scikit-learn"""

    def score(self, X, y):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the
        regression sum of squares ((y - y_pred) ** 2).sum() and v is the
        residual sum of squares ((y_true - y_true.mean()) ** 2).sum().
        Best possible score is 1.0, lower values are worse.


        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training set.

        y : array-like, shape = [n_samples]

        Returns
        -------
        z : float
        """

       
        return r2_score(y, self.predict(X))
    
    
class ElasticNet(LinearModel, RegressorMixin):
    """Linear Model trained with L1 and L2 prior as regularizer

    Minimizes the objective function::

            1 / (2 * n_samples) * ||y - Xw||^2_2 +
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::

            a * L1 + b * L2

    where::

            alpha = a + b and l1_ratio = a / (a + b)

    The parameter l1_ratio corresponds to alpha in the glmnet R package while
    alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio
    = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,
    unless you supply your own sequence of alpha.

    Parameters
    ----------
    alpha : float
        Constant that multiplies the penalty terms. Defaults to 1.0
        See the notes for the exact mathematical meaning of this
        parameter
        alpha = 0 is equivalent to an ordinary least square, solved
        by the LinearRegression object in the scikit. For numerical
        reasons, using alpha = 0 with the Lasso object is not advised
        and you should prefer the LinearRegression object.

    l1_ratio : float
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For
        l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1
        penalty.  For 0 < l1_ratio < 1, the penalty is a combination of L1 and
        L2.

    fit_intercept: bool
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    normalize : boolean, optional
        If True, the regressors X are normalized

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to 'auto' let us decide. The Gram
        matrix can also be passed as argument. For sparse input
        this option is always True to preserve sparsity.

    max_iter: int, optional
        The maximum number of iterations

    copy_X : boolean, optional, default False
        If True, X will be copied; else, it may be overwritten.

    tol: float, optional
        The tolerance for the optimization: if the updates are
        smaller than 'tol', the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than tol.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive: bool, optional
        When set to True, forces the coefficients to be positive.

    Attributes
    ----------
    `coef_` : array, shape = (n_features,)
        parameter vector (w in the cost function formula)

    `sparse_coef_` : scipy.sparse matrix, shape = (n_features, 1)
        `sparse_coef_` is a readonly property derived from `coef_`

    `intercept_` : float | array, shape = (n_targets,)
        independent term in decision function.

    `dual_gap_` : float
        the current fit is guaranteed to be epsilon-suboptimal with
        epsilon := `dual_gap_`

    `eps_` : float
        `eps_` is used to check if the fit converged to the requested
        `tol`

    Notes
    -----
    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a fortran contiguous numpy array.
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, precompute='auto', max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=True,
                 rho=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if rho is not None:
            self.l1_ratio = rho
            warnings.warn("rho was renamed to l1_ratio and will be removed "
                          "in 0.15", DeprecationWarning)
        self.coef_ = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.intercept_ = 0.0

    def fit(self, X, y, Xy=None, coef_init=None):
        """Fit model with coordinate descent

        Parameters
        -----------
        X: ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y: ndarray, shape = (n_samples,) or (n_samples, n_targets)
            Target
        Xy : array-like, optional
            Xy = np.dot(X.T, y) that can be precomputed. It is useful
            only when the Gram matrix is precomputed.
        coef_init: ndarray of shape n_features or (n_targets, n_features)
            The initial coeffients to warm-start the optimization

        Notes
        -----

        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a fortran contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """
        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator", stacklevel=2)
        X = atleast2d_or_csc(X, dtype=np.float64, order='F',
                             copy=self.copy_X and self.fit_intercept)
        # From now on X can be touched inplace
        y = np.asarray(y, dtype=np.float64)
        # now all computation with X can be done inplace
        fit = self._sparse_fit if sparse.isspmatrix(X) else self._dense_fit
        fit(X, y, Xy, coef_init)
        return self

    def _dense_fit(self, X, y, Xy=None, coef_init=None):

        # copy was done in fit if necessary
        X, y, X_mean, y_mean, X_std = center_data(
            X, y, self.fit_intercept, self.normalize, copy=False)

        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_samples, n_features = X.shape
        n_targets = y.shape[1]
        
        precompute = self.precompute
        if hasattr(precompute, '__array__') \
                and not np.allclose(X_mean, np.zeros(n_features)) \
                and not np.allclose(X_std, np.ones(n_features)):
            # recompute Gram
            precompute = 'auto'
            Xy = None

        coef_ = self._init_coef(coef_init, n_features, n_targets)
        dual_gap_ = np.empty(n_targets)
        eps_ = np.empty(n_targets)

        l1_reg = self.alpha*self.l1_ratio * n_samples
        l2_reg = 0.0#self.alpha * (1.0 - self.l1_ratio) * n_samples

        # precompute if n_samples > n_features
        if hasattr(precompute, '__array__'):
            Gram = precompute
        elif precompute or (precompute == 'auto' and n_samples > n_features):
            Gram = np.dot(X.T, X)
        else:
            Gram = None
        
        for k in xrange(n_targets):
            if Gram is None:
                coef_[k, :], dual_gap_[k], eps_[k] = \
                    cd_fast.enet_coordinate_descent(
                        coef_[k, :], l1_reg, l2_reg, X, y[:, k], self.max_iter,
                        self.tol, True)
            else:
                Gram = Gram.copy()
                if Xy is None:
                    this_Xy = np.dot(X.T, y[:, k])
                else:
                    this_Xy = Xy[:, k]
                coef_[k, :], dual_gap_[k], eps_[k] = \
                    cd_fast.enet_coordinate_descent_gram(
                        coef_[k, :], l1_reg, l2_reg, Gram, this_Xy, y[:, k],
                        self.max_iter, self.tol, True)

            if dual_gap_[k] > eps_[k]:
                warnings.warn('Objective did not converge for ' +
                              'target %d, you might want' % k +
                              ' to increase the number of iterations')

        self.coef_, self.dual_gap_, self.eps_ = (np.squeeze(a) for a in
                                                 (coef_, dual_gap_, eps_))
        self._set_intercept(X_mean, y_mean, X_std)

        # return self for chaining fit and predict calls
        return self

    def _sparse_fit(self, X, y, Xy=None, coef_init=None):

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes.\n" +
                             "Note: Sparse matrices cannot be indexed w/" +
                             "boolean masks (use `indices=True` in CV).")

        # NOTE: we are explicitly not centering the data the naive way to
        # avoid breaking the sparsity of X
        X_data, y, X_mean, y_mean, X_std = sparse_center_data(
            X, y, self.fit_intercept, self.normalize)

        if y.ndim == 1:
            y = y[:, np.newaxis]

        n_samples, n_features = X.shape[0], X.shape[1]
        n_targets = y.shape[1]

        coef_ = self._init_coef(coef_init, n_features, n_targets)
        dual_gap_ = np.empty(n_targets)
        eps_ = np.empty(n_targets)

        l1_reg = self.alpha * self.l1_ratio* n_samples
     
        l2_reg =0.0 #self.alpha * (1.0 - self.l1_ratio) * n_samples

        for k in xrange(n_targets):
            coef_[k, :], dual_gap_[k], eps_[k] = \
                cd_fast.sparse_enet_coordinate_descent(
                    coef_[k, :], l1_reg, l2_reg, X_data, X.indices,
                    X.indptr, y[:, k], X_mean / X_std,
                    self.max_iter, self.tol, True)

            if dual_gap_[k] > eps_[k]:
                warnings.warn('Objective did not converge for ' +
                              'target %d, you might want' % k +
                              ' to increase the number of iterations')

        self.coef_, self.dual_gap_, self.eps_ = (np.squeeze(a) for a in
                                                 (coef_, dual_gap_, eps_))
        self._set_intercept(X_mean, y_mean, X_std)

        # return self for chaining fit and predict calls
        return self

    def _init_coef(self, coef_init, n_features, n_targets):
        if coef_init is None:
            if not self.warm_start or self.coef_ is None:
                coef_ = np.zeros((n_targets, n_features), dtype=np.float64)
            else:
                coef_ = self.coef_
        else:
            coef_ = coef_init

        if coef_.ndim == 1:
            coef_ = coef_[np.newaxis, :]
        if coef_.shape != (n_targets, n_features):
            raise ValueError("X and coef_init have incompatible "
                             "shapes (%s != %s)."
                             % (coef_.shape, (n_targets, n_features)))

        return coef_

    @property
    def sparse_coef_(self):
        """ sparse representation of the fitted coef """
        return sparse.csr_matrix(self.coef_)

    def decision_function(self, X):
        """Decision function of the linear model

        Parameters
        ----------
        X : numpy array or scipy.sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        T : array, shape = (n_samples,)
            The predicted decision function
        """
        if sparse.isspmatrix(X):
            return np.ravel(safe_sparse_dot(self.coef_, X.T, dense_output=True)
                            + self.intercept_)
        else:
            return super(ElasticNet, self).decision_function(X)


class TransformerMixin(object):
    """Mixin class for all transformers in scikit-learn"""

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)

class SparseCodingMixin(TransformerMixin):
    """Sparse coding mixin"""

    def _set_sparse_coding_params(self, n_components,
                                  transform_algorithm='omp',
                                  transform_n_nonzero_coefs=None,
                                  transform_alpha=None, split_sign=False,
                                  n_jobs=1):
        self.n_components = n_components
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.split_sign = split_sign
        self.n_jobs = n_jobs

    def transform(self, X, y=None):
        """Encode the data as a sparse combination of the dictionary atoms.

        Coding method is determined by the object parameter
        `transform_algorithm`.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Test data to be transformed, must have the same number of
            features as the data used to train the model.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Transformed data

        """
        # XXX : kwargs is not documented
        X = array2d(X)
        n_samples, n_features = X.shape
        
        code = sparse_encode(
            X, self.components_, algorithm=self.transform_algorithm,
            n_nonzero_coefs=self.transform_n_nonzero_coefs,
            alpha=self.transform_alpha, n_jobs=self.n_jobs)

        if self.split_sign:
            # feature vector is split into a positive and negative side
            n_samples, n_features = code.shape
            split_code = np.empty((n_samples, 2 * n_features))
            split_code[:, :n_features] = np.maximum(code, 0)
            split_code[:, n_features:] = -np.minimum(code, 0)
            code = split_code

        return code

# XXX : could be moved to the linear_model module
def sparse_encode(X, dictionary, gram=None, cov=None, algorithm='lasso_lars',
                  n_nonzero_coefs=None, alpha=None, copy_cov=True, init=None,
                  max_iter=1000, n_jobs=1):
    """Sparse coding

    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::

        X ~= code * dictionary

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        Data matrix

    dictionary: array of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows for meaningful
        output.

    gram: array, shape=(n_components, n_components)
        Precomputed Gram matrix, dictionary * dictionary'

    cov: array, shape=(n_components, n_samples)
        Precomputed covariance, dictionary' * X

    algorithm: {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than alpha from
        the projection dictionary * X'

    n_nonzero_coefs: int, 0.1 * n_features by default
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case.

    alpha: float, 1. by default
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threhold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.

    init: array of shape (n_samples, n_components)
        Initialization value of the sparse codes. Only used if
        `algorithm='lasso_cd'`.

    max_iter: int, 1000 by default
        Maximum number of iterations to perform if `algorithm='lasso_cd'`.

    copy_cov: boolean, optional
        Whether to copy the precomputed covariance matrix; if False, it may be
        overwritten.

    n_jobs: int, optional
        Number of parallel jobs to run.

    Returns
    -------
    code: array of shape (n_samples, n_components)
        The sparse codes

    See also
    --------
    sklearn.linear_model.lars_path
    sklearn.linear_model.orthogonal_mp
    sklearn.linear_model.Lasso
    SparseCoder
    """
    dictionary = array2d(dictionary)
    X = array2d(X)
    n_samples, n_features = X.shape
    n_components = dictionary.shape[0]

    if gram is None and algorithm != 'threshold':
        gram = np.dot(dictionary, dictionary.T)
    if cov is None:
        copy_cov = False
        cov = np.dot(dictionary, X.T)

    if algorithm in ('lars', 'omp'):
        regularization = n_nonzero_coefs
        if regularization is None:
            regularization = max(n_features / 10, 1)
    else:
        regularization = alpha
        if regularization is None:
            regularization = 1.

    if n_jobs == 1 or algorithm == 'threshold':
        return _sparse_encode(X, dictionary, gram, cov=cov,
                              algorithm=algorithm,
                              regularization=regularization, copy_cov=copy_cov,
                              init=init, max_iter=max_iter)

    # Enter parallel code block
    code = np.empty((n_samples, n_components))
    slices = list(gen_even_slices(n_samples, n_jobs))

    code_views = Parallel(n_jobs=n_jobs)(
        delayed(_sparse_encode)(
            X[this_slice], dictionary, gram, cov[:, this_slice], algorithm,
            regularization=regularization, copy_cov=copy_cov,
            init=init[this_slice] if init is not None else None,
            max_iter=max_iter)
        for this_slice in slices)
    for this_slice, this_view in zip(slices, code_views):
        code[this_slice] = this_view
    return code


def _sparse_encode(X, dictionary, gram, cov=None, algorithm='lasso_lars',
                   regularization=None, copy_cov=True,
                   init=None, max_iter=1000):
    """Generic sparse coding

    Each column of the result is the solution to a Lasso problem.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        Data matrix.

    dictionary: array of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows.

    gram: None | array, shape=(n_components, n_components)
        Precomputed Gram matrix, dictionary * dictionary'
        gram can be None if method is 'threshold'.

    cov: array, shape=(n_components, n_samples)
        Precomputed covariance, dictionary * X'

    algorithm: {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than regularization
        from the projection dictionary * data'

    regularization : int | float
        The regularization parameter. It corresponds to alpha when
        algorithm is 'lasso_lars', 'lasso_cd' or 'threshold'.
        Otherwise it corresponds to n_nonzero_coefs.

    init: array of shape (n_samples, n_components)
        Initialization value of the sparse code. Only used if
        `algorithm='lasso_cd'`.

    max_iter: int, 1000 by default
        Maximum number of iterations to perform if `algorithm='lasso_cd'`.

    copy_cov: boolean, optional
        Whether to copy the precomputed covariance matrix; if False, it may be
        overwritten.

    Returns
    -------
    code: array of shape (n_components, n_features)
        The sparse codes

    See also
    --------
    sklearn.linear_model.lars_path
    sklearn.linear_model.orthogonal_mp
    sklearn.linear_model.Lasso
    SparseCoder
    """
    if X.ndim == 1:
        X = X[:, np.newaxis]
    n_samples, n_features = X.shape
    if cov is None and algorithm != 'lasso_cd':
        # overwriting cov is safe
        copy_cov = False
        cov = np.dot(dictionary, X.T)

    if algorithm == 'lasso_lars':
        alpha = float(regularization) / n_features  # account for scaling
        try:
            err_mgt = np.seterr(all='ignore')
            lasso_lars = LassoLars(alpha=alpha, fit_intercept=False,
                                   verbose=False, normalize=False,
                                   precompute=gram, fit_path=False)
      
            lasso_lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lasso_lars.coef_
        finally:
            np.seterr(**err_mgt)

    elif algorithm == 'lasso_cd':
        alpha = float(regularization) / n_features  # account for scaling
        clf = Lasso(alpha=alpha, fit_intercept=False, precompute=None,
                    max_iter=max_iter,positive=True)
        clf.fit(dictionary.T, X.T, Xy=cov, coef_init=init)
        new_code = clf.coef_

    elif algorithm == 'lars':
        print "lars not fit this method"
    elif algorithm == 'threshold':
        new_code = ((np.sign(cov) *
                    np.maximum(np.abs(cov) - regularization, 0)).T)

    elif algorithm == 'omp':
        norms_squared = np.sum((X ** 2), axis=1)
        new_code = orthogonal_mp_gram(gram, cov, regularization, None,
                                      norms_squared, copy_Xy=copy_cov).T
    else:
        raise ValueError('Sparse coding method must be "lasso_lars" '
                         '"lasso_cd",  "lasso", "threshold" or "omp", got %s.'
                         % algorithm)
    return new_code


class SparseCoder1(BaseEstimator, SparseCodingMixin):
    """Sparse coding

    Finds a sparse representation of data against a fixed, precomputed
    dictionary.

    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::

        X ~= code * dictionary

    Parameters
    ----------
    dictionary : array, [n_components, n_features]
        The dictionary atoms used for sparse coding. Lines are assumed to be
        normalized to unit norm.

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', \
    'threshold'}
        Algorithm used to transform the data:
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than alpha from
        the projection ``dictionary * X'``

    transform_n_nonzero_coefs : int, ``0.1 * n_features`` by default
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case.

    transform_alpha : float, 1. by default
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.

    split_sign : bool, False by default
        Whether to split the sparse feature vector into the concatenation of
        its negative part and its positive part. This can improve the
        performance of downstream classifiers.

    n_jobs : int,
        number of parallel jobs to run


    """

    def __init__(self, dictionary, transform_algorithm='omp',
                 transform_n_nonzero_coefs=None, transform_alpha=None,
                 split_sign=False, n_jobs=1):
        self._set_sparse_coding_params(dictionary.shape[0],
                                       transform_algorithm,
                                       transform_n_nonzero_coefs,
                                       transform_alpha, split_sign, n_jobs)
        self.components_ = dictionary

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.
        """
        return self
    
# Lasso model

class Lasso(ElasticNet):
    """Linear Model trained with L1 prior as regularizer (aka the Lasso)

    The optimization objective for Lasso is::

        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Technically the Lasso model is optimizing the same objective function as
    the Elastic Net with l1_ratio=1.0 (no L2 penalty).

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0
        alpha = 0 is equivalent to an ordinary least square, solved
        by the LinearRegression object in the scikit. For numerical
        reasons, using alpha = 0 is with the Lasso object is not advised
        and you should prefer the LinearRegression object.

    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional
        If True, the regressors X are normalized

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to 'auto' let us decide. The Gram
        matrix can also be passed as argument. For sparse input
        this option is always True to preserve sparsity.

    max_iter: int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than 'tol', the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than tol.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive : bool, optional
        When set to True, forces the coefficients to be positive.


    Attributes
    ----------
    `coef_` : array, shape = (n_features,)
        parameter vector (w in the cost function formula)

    `sparse_coef_` : scipy.sparse matrix, shape = (n_features, 1)
        `sparse_coef_` is a readonly property derived from `coef_`

    `intercept_` : float
        independent term in decision function.

    `dual_gap_` : float
        the current fit is guaranteed to be epsilon-suboptimal with
        epsilon := `dual_gap_`

    `eps_` : float
        `eps_` is used to check if the fit converged to the requested
        `tol`


    Notes
    -----
    The algorithm used to fit the model is coordinate descent.

    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a fortran contiguous numpy array.
    """

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute='auto', copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=True):
        super(Lasso, self).__init__(
            alpha=alpha, l1_ratio=1.0, fit_intercept=fit_intercept,
            normalize=normalize, precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=True)
        
        
def dict_learning1(X, n_components, alpha, max_iter=1000, tol=1e-8,
                  method='cd', n_jobs=1, dict_init=None, code_init=None,
                  callback=None, verbose=False, random_state=None,
                  n_atoms=None):
    """Solves a dictionary learning matrix factorization problem.

        (U^*, V^*) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                     (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code.
    method: {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.
    Returns
    -------
     errors: array
        Vector of errors at each iteration.

    """

    if not n_atoms is None:
        n_components = n_atoms
        warnings.warn("Parameter n_atoms has been renamed to"
                      "'n_components' and will be removed in release 0.14.",
                      DeprecationWarning, stacklevel=2)

    if method not in ('lars', 'cd'):
        raise ValueError('Coding method not supported as a fit algorithm.')
    method = 'lasso_' + method

    t0 = time.time()
    # Avoid integer division problems
    alpha = float(alpha)
    random_state = check_random_state(random_state)

    if n_jobs == -1:
        n_jobs = cpu_count()

    # Init the code and the dictionary with SVD of Y
    if code_init is not None and dict_init is not None:
        code = np.array(code_init, order='F')
        # Don't copy V, it will happen below
        dictionary = dict_init
    else:
        code, S, dictionary = linalg.svd(X, full_matrices=False)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
   
    if n_components <= r:  # True even if n_components=None
        code = code[:, :n_components]
        dictionary = dictionary[:n_components, :]
    else:
        code = np.c_[code, np.zeros((len(code), n_components - r))]
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]

    # Fortran-order dict, as we are going to access its row vectors
    dictionary = np.array(dictionary, order='F')
    residuals = 0
    errors = []
    current_cost = np.nan

    if verbose == 1:
        print '[dict_learning]',
    for ii in xrange(max_iter):
        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            print ("Iteration % 3i "
                   "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)"
                   % (ii, dt, dt / 60, current_cost))

        # Update code
        if ii%2== 0:
            code = code_init           
        else:
            code = sparse_encode(X, dictionary, algorithm="lasso_cd", alpha=alpha,
                                     init=None, n_jobs=n_jobs)

   
        # Update dictionary
        dictionary, residuals = _update_dict(dictionary.T, X.T, code.T,
                                             verbose=verbose, return_r2=True,
                                             random_state=random_state)
        dictionary = dictionary.T
        # Cost function
        current_cost = 0.5 * residuals + alpha * np.sum(np.abs(code))
    
        errors.append(current_cost)
      
        if ii > 0:
            dE = errors[-2] - errors[-1]
            # assert(dE >= -tol * errors[-1])
       
            if dE*dE < tol * errors[-1]:
                if verbose == 1:
                    # A line return
                    print "this his"
                elif verbose:
                    print "--- Convergence reached after %d iterations" % ii
                break
        if ii % 5 == 0 and callback is not None:
            callback(locals())

    return code, dictionary, errors
def dict_learning2(X, n_components, alpha, max_iter=1000, tol=1e-8,
                  method='cd', n_jobs=1, dict_init=None, code_init=None,
                  callback=None, verbose=False, random_state=None,
                  n_atoms=None):
    """Solves a dictionary learning matrix factorization problem.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                     (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        Data matrix.

    n_components: int,
        Number of dictionary atoms to extract.

    alpha: int,
        Sparsity controlling parameter.

    max_iter: int,
        Maximum number of iterations to perform.

    tol: float,
        Tolerance for the stopping condition.

    method: {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    n_jobs: int,
        Number of parallel jobs to run, or -1 to autodetect.

    dict_init: array of shape (n_components, n_features),
        Initial value for the dictionary for warm restart scenarios.

    code_init: array of shape (n_samples, n_components),
        Initial value for the sparse code for warm restart scenarios.

    callback:
        Callable that gets invoked every five iterations.

    verbose:
        Degree of output the procedure will print.

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    Returns
    -------
    code: array of shape (n_samples, n_components)
        The sparse code factor in the matrix factorization.

    dictionary: array of shape (n_components, n_features),
        The dictionary factor in the matrix factorization.

    errors: array
        Vector of errors at each iteration.

    See also
    --------
    dict_learning_online
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    """

    if not n_atoms is None:
        n_components = n_atoms
        warnings.warn("Parameter n_atoms has been renamed to"
                      "'n_components' and will be removed in release 0.14.",
                      DeprecationWarning, stacklevel=2)

    if method not in ('lars', 'cd'):
        raise ValueError('Coding method not supported as a fit algorithm.')
    method = 'lasso_' + method

    t0 = time.time()
    # Avoid integer division problems
    alpha = float(alpha)
    random_state = check_random_state(random_state)

    if n_jobs == -1:
        n_jobs = cpu_count()

    # Init the code and the dictionary with SVD of Y
    if code_init is not None and dict_init is not None:
        
        code = np.array(code_init, order='F')
        # Don't copy V, it will happen below
        dictionary = dict_init
    else:
        code, S, dictionary = linalg.svd(X, full_matrices=False)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
   
    if n_components <= r:  # True even if n_components=None
        code = code[:, :n_components]
        dictionary = dictionary[:n_components, :]
    else:
        code = np.c_[code, np.zeros((len(code), n_components - r))]
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]

    # Fortran-order dict, as we are going to access its row vectors
    dictionary = np.array(dictionary, order='F')

    residuals = 0

    errors = []
    current_cost = np.nan

    if verbose == 1:
        print '[dict_learning]',
    for ii in xrange(max_iter):
        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            print ("Iteration % 3i "
                   "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)"
                   % (ii, dt, dt / 60, current_cost))

        # Update code
        if ii== 0:
            code = code_init
        else:
                   
            code = sparse_encode(X, dictionary, algorithm="lasso_lars", alpha=alpha,
                                     init=None, n_jobs=n_jobs)
        # Update dictionary
        dictionary, residuals = _update_dict(dictionary.T, X.T, code.T,
                                             verbose=verbose, return_r2=True,
                                             random_state=random_state)
        dictionary = dictionary.T
        # Cost function
        current_cost = 0.5 * residuals + alpha * np.sum(np.abs(code))
        errors.append(current_cost)

        if ii > 0:
            dE = errors[-2] - errors[-1]
            # assert(dE >= -tol * errors[-1])

            if dE*dE < tol * errors[-1]:
                if verbose == 1:
                    # A line return
                    print "this his"
                elif verbose:
                    print "--- Convergence reached after %d iterations" % ii
                break
        if ii % 5 == 0 and callback is not None:
            callback(locals())

    return code, dictionary, errors

def _update_dict(dictionary, Y, code, verbose=False, return_r2=False,
                 random_state=None):

    n_components = len(code)
    n_samples = Y.shape[0]
    random_state = check_random_state(random_state)
    # Residuals, computed 'in-place' for efficiency
  
    R = -np.dot(dictionary, code) 

    R += Y
    R = np.asfortranarray(R)
    ger, = linalg.get_blas_funcs(('ger',), (dictionary, code))
    for k in xrange(n_components):
        # R <- 1.0 * U_k * V_k^T + R
        R = 0.5*ger(1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)

        
        dictionary[:, k] = np.dot(R, code[k, :].T)
        
        # Scale k'th atom
        atom_norm_square = np.dot(dictionary[:, k], dictionary[:, k])
        
        if atom_norm_square < 1e-10:
            if verbose == 1:
                sys.stdout.write("+")
                sys.stdout.flush()
            elif verbose:
                print "Adding new random atom"
            
            dictionary[:, k] = random_state.randn(n_samples)
            # Setting corresponding coefs to 0
            code[k, :] = 0.0
            dictionary[:, k] /= sqrt(np.dot(dictionary[:, k],
                                            dictionary[:, k]))
        else:
            dictionary[:, k] /= sqrt(atom_norm_square)
            
            # R <- -1.0 * U_k * V_k^T + R
            R = ger(-1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
           

    if return_r2:
        R **= 2
        # R is fortran-ordered. For numpy version < 1.6, sum does not
        # follow the quick striding first, and is thus inefficient on
        # fortran ordered data. We take a flat view of the data with no
        # striding
        R = as_strided(R, shape=(R.size, ), strides=(R.dtype.itemsize,))
        R = np.sum(R)
        return dictionary, R
    return dictionary


if __name__=="__main__":
    print "test"
    dictionary_A = [[0.7,0.3,0.3,0,0,0.6],[0,0,0.5,0.7,0.2,0.7]]

    D = np.empty((len(dictionary_A), len(dictionary_A[0])))
    for i, vec in enumerate(dictionary_A):
        D[i] = vec 
#     D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
#     print D
   
    coder = SparseCoder1(dictionary=D, transform_n_nonzero_coefs=1,
                            transform_alpha=0.5, transform_algorithm="lasso_cd")
#     print D
    y = np.array([0.9,0.1,0.2,0.8,0.7,0.0])
    y1 = np.array([0.9,0.1,0.0,0.7,0.2,0.7])
#     print coder.transform(y1)
    print coder.transform(y)
    
    Y= np.empty((2, 6))
    Y[0]= y
    Y[1]= y1
    
    x = [1,0]
    x1 = [0,1]
    X = np.empty((2,2))
    X[0] = x
    X[1] = x1


    V, U, E = dict_learning2(Y, 2, 0.5,
                                tol=1e-6, max_iter=1000,
                                method='cd',
                                n_jobs=1,
                                code_init=X,
                                dict_init=D,
                                verbose=False,
                                random_state=None)
     
    print U
    coder = SparseCoder1(dictionary=U, transform_n_nonzero_coefs=2,
                            transform_alpha=0.5, transform_algorithm="omp")
 
    y = np.array([0.9,0.1,0.2,0.8,0.7,0.0])
    print  coder.transform(y)
#     

    y = np.array([0.6,0.1,0.3,0,0.1,0.5])
    print  coder.transform(y)
     
    
    
    
    
    
    
    
    
    
    
    
    
    