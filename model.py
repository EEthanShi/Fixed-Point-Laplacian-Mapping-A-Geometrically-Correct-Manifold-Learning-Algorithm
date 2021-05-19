# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 10:41:58 2021

Includes the main models, including:
    FLE: fixed point Laplacian
    TC: a wrapper function that implements TC with inconsistency corrected

@author: Andi
"""
import numpy as np
import gudhi


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.deprecation import deprecated
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.graph import graph_shortest_path
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer


import numpy as np
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import eigsh

from sklearn.base import BaseEstimator, TransformerMixin, _UnstableArchMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.neighbors import NearestNeighbors



import torch
from torch import nn, optim



def FLE(Lap, C, C_index):
    """
    Inputs:
        Lap: the full laplacian-like matrix, which is N x N
        C: the fixed points coordinate in d
        C_index: the fixed points index
    Output:
        Y: optimal solution including fixed points (see paper for more details)
    """
    N = Lap.shape[0] # no. of total samples
    d = C.shape[1]

    Y_index = np.array([i for i in range(N) if i not in C_index ]) # index for other points
    
    # extract the block matrix
    Ly = Lap[Y_index, :][:, Y_index] # (N - Nc) x (N - Nc)
    #Lc = Lap[C_index, :][:, C_index] # Nc x Nc
    Lyc = Lap[Y_index, :][:, C_index] # (N- Nc) x Nc
    
    # find out the solution by -Ly^+ Lyc C
    Y_opt = - np.linalg.pinv(Ly, hermitian=True) * Lyc * C
    
    # insert the fixed points according to the index
    Y = np.zeros((N, d))
    Y[Y_index] = Y_opt
    Y[C_index] = C
    
    return Y


def TC(data, dim, fix_inconsistencies, radius, max_time):
    """
    A wrapper function to construct tangential complex and fix inconsistencies using perturbation
        Input: 
            data: point cloud
            dim: intrinsic dimension of manifold
            fix_inconsistencies: Boolen whether to fix inconsistency by perturbation
            radius: perturbation radius
            max_time: time limit for perturbation
        Output: a tc object
    """
    tc = gudhi.TangentialComplex(intrisic_dim = dim, points=data)
    tc.compute_tangential_complex()
    if fix_inconsistencies and (tc.num_inconsistent_simplices() > 0):
        print('Fixing {} inconsistencies using perturbation'.format(tc.num_inconsistent_simplices()))
        tc.fix_inconsistencies_using_perturbation(radius,max_time)
        print('Finished! Number of remaining inconsistencies:', tc.num_inconsistent_simplices())
    return tc




#%% ISOMAP

class myIsomap(TransformerMixin, BaseEstimator):
    """Isomap Embedding
    Non-linear dimensionality reduction through Isometric Mapping
    Read more in the :ref:`User Guide <isomap>`.
    Parameters
    ----------
    
    Andy: Everything is the same except for the _fit_transform function, takes
    in a graph, which is precomputed.    
    
    ----------
    n_neighbors : integer
        number of neighbors to consider for each point.
    n_components : integer
        number of coordinates for the manifold
    eigen_solver : ['auto'|'arpack'|'dense']
        'auto' : Attempt to choose the most efficient solver
        for the given problem.
        'arpack' : Use Arnoldi decomposition to find the eigenvalues
        and eigenvectors.
        'dense' : Use a direct solver (i.e. LAPACK)
        for the eigenvalue decomposition.
    tol : float
        Convergence tolerance passed to arpack or lobpcg.
        not used if eigen_solver == 'dense'.
    max_iter : integer
        Maximum number of iterations for the arpack solver.
        not used if eigen_solver == 'dense'.
    path_method : string ['auto'|'FW'|'D']
        Method to use in finding shortest path.
        'auto' : attempt to choose the best algorithm automatically.
        'FW' : Floyd-Warshall algorithm.
        'D' : Dijkstra's algorithm.
    neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
        Algorithm to use for nearest neighbors search,
        passed to neighbors.NearestNeighbors instance.
    n_jobs : int or None, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    metric : string, or callable, default="minkowski"
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a :term:`Glossary <sparse graph>`.
        .. versionadded:: 0.22
    p : int, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        .. versionadded:: 0.22
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
        .. versionadded:: 0.22
    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.
    kernel_pca_ : object
        :class:`~sklearn.decomposition.KernelPCA` object used to implement the
        embedding.
    nbrs_ : sklearn.neighbors.NearestNeighbors instance
        Stores nearest neighbors instance, including BallTree or KDtree
        if applicable.
    dist_matrix_ : array-like, shape (n_samples, n_samples)
        Stores the geodesic distance matrix of training data.
    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import Isomap
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = Isomap(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)
    References
    ----------
    .. [1] Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric
           framework for nonlinear dimensionality reduction. Science 290 (5500)
    """
    @_deprecate_positional_args
    def __init__(self, *, n_components=2, eigen_solver='auto',
                 tol=0, max_iter=None, path_method='auto',
                 n_jobs=None, metric='minkowski',
                 p=2, metric_params=None):
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.n_jobs = n_jobs
        self.metric = metric
        self.p = p
        self.metric_params = metric_params

    def _fit_transform(self, G):

        self.kernel_pca_ = KernelPCA(n_components=self.n_components,
                                     kernel="precomputed",
                                     eigen_solver=self.eigen_solver,
                                     tol=self.tol, max_iter=self.max_iter,
                                     n_jobs=self.n_jobs)


        self.dist_matrix_ = graph_shortest_path(G,
                                                method=self.path_method,
                                                directed=False)
        G = self.dist_matrix_ ** 2
        G *= -0.5

        self.embedding_ = self.kernel_pca_.fit_transform(G)

    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `training_data_` was deprecated in version 0.22 and"
        " will be removed in 0.24."
    )
    @property
    def training_data_(self):
        check_is_fitted(self)
        return self.nbrs_._fit_X

    def reconstruction_error(self):
        """Compute the reconstruction error for the embedding.
        Returns
        -------
        reconstruction_error : float
        Notes
        -----
        The cost function of an isomap embedding is
        ``E = frobenius_norm[K(D) - K(D_fit)] / n_samples``
        Where D is the matrix of distances for the input data X,
        D_fit is the matrix of distances for the output embedding X_fit,
        and K is the isomap kernel:
        ``K(D) = -0.5 * (I - 1/n_samples) * D^2 * (I - 1/n_samples)``
        """
        G = -0.5 * self.dist_matrix_ ** 2
        G_center = KernelCenterer().fit_transform(G)
        evals = self.kernel_pca_.lambdas_
        return np.sqrt(np.sum(G_center ** 2) - np.sum(evals ** 2)) / G.shape[0]

    def fit(self, X, y=None):
        """Compute the embedding vectors for data X
        Parameters
        ----------
        X : {array-like, sparse graph, BallTree, KDTree, NearestNeighbors}
            Sample data, shape = (n_samples, n_features), in the form of a
            numpy array, sparse graph, precomputed tree, or NearestNeighbors
            object.
        y : Ignored
        Returns
        -------
        self : returns an instance of self.
        """
        self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.
        Parameters
        ----------
        X : {array-like, sparse graph, BallTree, KDTree}
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.
        y : Ignored
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self._fit_transform(X)
        return self.embedding_

    def transform(self, X):
        """Transform X.
        This is implemented by linking the points X into the graph of geodesic
        distances of the training data. First the `n_neighbors` nearest
        neighbors of X are found in the training data, and from these the
        shortest geodesic distances from each point in X to each point in
        the training data are computed in order to construct the kernel.
        The embedding of X is the projection of this kernel onto the
        embedding vectors of the training set.
        Parameters
        ----------
        X : array-like, shape (n_queries, n_features)
            If neighbors_algorithm='precomputed', X is assumed to be a
            distance matrix or a sparse graph of shape
            (n_queries, n_samples_fit).
        Returns
        -------
        X_new : array-like, shape (n_queries, n_components)
        """
        check_is_fitted(self)
        distances, indices = self.nbrs_.kneighbors(X, return_distance=True)

        # Create the graph of shortest distances from X to
        # training data via the nearest neighbors of X.
        # This can be done as a single array operation, but it potentially
        # takes a lot of memory.  To avoid that, use a loop:

        n_samples_fit = self.nbrs_.n_samples_fit_
        n_queries = distances.shape[0]
        G_X = np.zeros((n_queries, n_samples_fit))
        for i in range(n_queries):
            G_X[i] = np.min(self.dist_matrix_[indices[i]] +
                            distances[i][:, None], 0)

        G_X **= 2
        G_X *= -0.5

        return self.kernel_pca_.transform(G_X)


#%% LLE & LSTA

def _init_arpack_v0(size, random_state):
    """Initialize the starting vector for iteration in ARPACK functions.
    Initialize a ndarray with values sampled from the uniform distribution on
    [-1, 1]. This initialization model has been chosen to be consistent with
    the ARPACK one as another initialization can lead to convergence issues.
    Parameters
    ----------
    size : int
        The size of the eigenvalue vector to be initialized.
    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator used to generate a
        uniform distribution. If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is the
        random number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.
    Returns
    -------
    v0 : ndarray of shape (size,)
        The initialized vector.
    """
    random_state = check_random_state(random_state)
    v0 = random_state.uniform(-1, 1, size)
    return v0

def barycenter_weights(G, X, indices, reg=1e-3):
    """Compute barycenter weights of X from Y along the first axis
    We estimate the weights to assign to each point in Y[indices] to recover
    the point X[i]. The barycenter weights sum to 1.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_dim)
    Y : array-like, shape (n_samples, n_dim)
    indices : array-like, shape (n_samples, n_dim)
            Indices of the points in Y used to compute the barycenter
    reg : float, default=1e-3
        amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim
    Returns
    -------
    B : array-like, shape (n_samples, n_neighbors)
    Notes
    -----
    See developers note for more information.
    """
    n_samples = G.shape[0]

    B = np.zeros((n_samples, n_samples))

    # this might raise a LinalgError if G is singular and has trace
    # zero
    for i in range(n_samples):
        ind_nbrs = indices[1][indices[0] == i]
        
        A = X[ind_nbrs]
        C = A - X[i]  # broadcasting
        CCT = np.dot(C, C.T)
        trace = np.trace(CCT)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        CCT.flat[::len(ind_nbrs) + 1] += R
        v = np.ones(len(ind_nbrs), dtype=X.dtype)
        w = solve(CCT, v, sym_pos=True)
        B[i, ind_nbrs] = w / np.sum(w)
    return B

def barycenter_kneighbors_graph(G, X, n_neighbors, reg=1e-3, n_jobs=None):
    """Computes the barycenter weighted graph of k-Neighbors for points in X
    Parameters
    ----------
    X : {array-like, NearestNeighbors}
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array or a NearestNeighbors object.
    n_neighbors : int
        Number of neighbors for each sample.
    reg : float, default=1e-3
        Amount of regularization when solving the least-squares
        problem. Only relevant if mode='barycenter'. If None, use the
        default.
    n_jobs : int or None, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Returns
    -------
    A : sparse matrix in CSR format, shape = [n_samples, n_samples]
        A[i, j] is assigned the weight of edge that connects i to j.
    See Also
    --------
    sklearn.neighbors.kneighbors_graph
    sklearn.neighbors.radius_neighbors_graph
    """
    ind = G.nonzero()
    
    data = barycenter_weights(G, X, ind, reg=reg)
    
    return csr_matrix(data)


def null_space(M, k, k_skip=1, eigen_solver='arpack', tol=1E-6, max_iter=100,
               random_state=None):
    """
    Find the null space of a matrix M.
    Parameters
    ----------
    M : {array, matrix, sparse matrix, LinearOperator}
        Input covariance matrix: should be symmetric positive semi-definite
    k : int
        Number of eigenvalues/vectors to return
    k_skip : int, default=1
        Number of low eigenvalues to skip.
    eigen_solver : {'auto', 'arpack', 'dense'}, default='arpack'
        auto : algorithm will attempt to choose the best method for input data
        arpack : use arnoldi iteration in shift-invert mode.
                    For this method, M may be a dense matrix, sparse matrix,
                    or general linear operator.
                    Warning: ARPACK can be unstable for some problems.  It is
                    best to try several random seeds in order to check results.
        dense  : use standard dense matrix operations for the eigenvalue
                    decomposition.  For this method, M must be an array
                    or matrix type.  This method should be avoided for
                    large problems.
    tol : float, default=1e-6
        Tolerance for 'arpack' method.
        Not used if eigen_solver=='dense'.
    max_iter : int, default=100
        Maximum number of iterations for 'arpack' method.
        Not used if eigen_solver=='dense'
    random_state : int, RandomState instance, default=None
        Determines the random number generator when ``solver`` == 'arpack'.
        Pass an int for reproducible results across multiple function calls.
        See :term: `Glossary <random_state>`.
    """
    if eigen_solver == 'auto':
        if M.shape[0] > 200 and k + k_skip < 10:
            eigen_solver = 'arpack'
        else:
            eigen_solver = 'dense'

    if eigen_solver == 'arpack':
        v0 = _init_arpack_v0(M.shape[0], random_state)
        try:
            eigen_values, eigen_vectors = eigsh(M, k + k_skip, sigma=0.0,
                                                tol=tol, maxiter=max_iter,
                                                v0=v0)
        except RuntimeError as e:
            raise ValueError(
                "Error in determining null-space with ARPACK. Error message: "
                "'%s'. Note that eigen_solver='arpack' can fail when the "
                "weight matrix is singular or otherwise ill-behaved. In that "
                "case, eigen_solver='dense' is recommended. See online "
                "documentation for more information." % e
            ) from e

        return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])
    elif eigen_solver == 'dense':
        if hasattr(M, 'toarray'):
            M = M.toarray()
        eigen_values, eigen_vectors = eigh(
            M, eigvals=(k_skip, k + k_skip - 1), overwrite_a=True)
        index = np.argsort(np.abs(eigen_values))
        return eigen_vectors[:, index], np.sum(eigen_values)
    else:
        raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)


@_deprecate_positional_args
def locally_linear_embedding(
        G, X, *, n_neighbors, n_components, reg=1e-3, eigen_solver='auto',
        tol=1e-6, max_iter=100, method='standard', hessian_tol=1E-4,
        modified_tol=1E-12, random_state=None, n_jobs=None):
    """Perform a Locally Linear Embedding analysis on the data.
    Read more in the :ref:`User Guide <locally_linear_embedding>`.
    Parameters
    ----------
    X : {array-like, NearestNeighbors}
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array or a NearestNeighbors object.
    n_neighbors : int
        number of neighbors to consider for each point.
    n_components : int
        number of coordinates for the manifold.
    reg : float, default=1e-3
        regularization constant, multiplies the trace of the local covariance
        matrix of the distances.
    eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
        auto : algorithm will attempt to choose the best method for input data
        arpack : use arnoldi iteration in shift-invert mode.
                    For this method, M may be a dense matrix, sparse matrix,
                    or general linear operator.
                    Warning: ARPACK can be unstable for some problems.  It is
                    best to try several random seeds in order to check results.
        dense  : use standard dense matrix operations for the eigenvalue
                    decomposition.  For this method, M must be an array
                    or matrix type.  This method should be avoided for
                    large problems.
    tol : float, default=1e-6
        Tolerance for 'arpack' method
        Not used if eigen_solver=='dense'.
    max_iter : int, default=100
        maximum number of iterations for the arpack solver.
    method : {'standard', 'hessian', 'modified', 'ltsa'}, default='standard'
        standard : use the standard locally linear embedding algorithm.
                   see reference [1]_
        hessian  : use the Hessian eigenmap method.  This method requires
                   n_neighbors > n_components * (1 + (n_components + 1) / 2.
                   see reference [2]_
        modified : use the modified locally linear embedding algorithm.
                   see reference [3]_
        ltsa     : use local tangent space alignment algorithm
                   see reference [4]_
    hessian_tol : float, default=1e-4
        Tolerance for Hessian eigenmapping method.
        Only used if method == 'hessian'
    modified_tol : float, default=1e-12
        Tolerance for modified LLE method.
        Only used if method == 'modified'
    random_state : int, RandomState instance, default=None
        Determines the random number generator when ``solver`` == 'arpack'.
        Pass an int for reproducible results across multiple function calls.
        See :term: `Glossary <random_state>`.
    n_jobs : int or None, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Returns
    -------
    Y : array-like, shape [n_samples, n_components]
        Embedding vectors.
    squared_error : float
        Reconstruction error for the embedding vectors. Equivalent to
        ``norm(Y - W Y, 'fro')**2``, where W are the reconstruction weights.
    References
    ----------
    .. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
        by locally linear embedding.  Science 290:2323 (2000).
    .. [2] Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
        linear embedding techniques for high-dimensional data.
        Proc Natl Acad Sci U S A.  100:5591 (2003).
    .. [3] Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
        Embedding Using Multiple Weights.
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382
    .. [4] Zhang, Z. & Zha, H. Principal manifolds and nonlinear
        dimensionality reduction via tangent space alignment.
        Journal of Shanghai Univ.  8:406 (2004)
    """
    if eigen_solver not in ('auto', 'arpack', 'dense'):
        raise ValueError("unrecognized eigen_solver '%s'" % eigen_solver)

    if method not in ('standard', 'hessian', 'modified', 'ltsa'):
        raise ValueError("unrecognized method '%s'" % method)

    M_sparse = (eigen_solver != 'dense')

    if method == 'standard':
        W = barycenter_kneighbors_graph(
            G, X, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)

        # we'll compute M = (I-W)'(I-W)
        # depending on the solver, we'll do this differently
        if M_sparse:
            M = eye(*W.shape, format=W.format) - W
            M = (M.T * M).tocsr()
        else:
            M = (W.T * W - W.T - W).toarray()
            M.flat[::M.shape[0] + 1] += 1  # W = W - I = W - I


    elif method == 'ltsa':
        
        N = X.shape[0]
        
        M = np.zeros((N, N))
        
        indices = G.nonzero()

        for i in range(N):
            nbrs_ind = indices[1][indices[0] == i]
            Xi = X[ nbrs_ind ]
            Xi -= Xi.mean(0)

            # compute n_components largest eigenvalues of Xi * Xi^T
            Ci = np.dot(Xi, Xi.T)
            v = eigh(Ci)[1][:, ::-1]

            Gi = np.zeros((len(nbrs_ind), n_components + 1))
            Gi[:, 1:] = v[:, :n_components]
            Gi[:, 0] = 1. / np.sqrt(len(nbrs_ind))

            GiGiT = np.dot(Gi, Gi.T)

            nbrs_x, nbrs_y = np.meshgrid(nbrs_ind, nbrs_ind)
            M[nbrs_x, nbrs_y] -= GiGiT
            M[nbrs_ind, nbrs_ind] += 1

    return null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver,
                      tol=tol, max_iter=max_iter, random_state=random_state)


class myLocallyLinearEmbedding(TransformerMixin,
                             _UnstableArchMixin, BaseEstimator):
    """Locally Linear Embedding
    Read more in the :ref:`User Guide <locally_linear_embedding>`.
    Parameters
    ----------
    n_neighbors : int, default=5
        number of neighbors to consider for each point.
    n_components : int, default=2
        number of coordinates for the manifold
    reg : float, default=1e-3
        regularization constant, multiplies the trace of the local covariance
        matrix of the distances.
    eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
        auto : algorithm will attempt to choose the best method for input data
        arpack : use arnoldi iteration in shift-invert mode.
                    For this method, M may be a dense matrix, sparse matrix,
                    or general linear operator.
                    Warning: ARPACK can be unstable for some problems.  It is
                    best to try several random seeds in order to check results.
        dense  : use standard dense matrix operations for the eigenvalue
                    decomposition.  For this method, M must be an array
                    or matrix type.  This method should be avoided for
                    large problems.
    tol : float, default=1e-6
        Tolerance for 'arpack' method
        Not used if eigen_solver=='dense'.
    max_iter : int, default=100
        maximum number of iterations for the arpack solver.
        Not used if eigen_solver=='dense'.
    method : {'standard', 'hessian', 'modified', 'ltsa'}, default='standard'
        - `standard`: use the standard locally linear embedding algorithm. see
          reference [1]_
        - `hessian`: use the Hessian eigenmap method. This method requires
          ``n_neighbors > n_components * (1 + (n_components + 1) / 2``. see
          reference [2]_
        - `modified`: use the modified locally linear embedding algorithm.
          see reference [3]_
        - `ltsa`: use local tangent space alignment algorithm. see
          reference [4]_
    hessian_tol : float, default=1e-4
        Tolerance for Hessian eigenmapping method.
        Only used if ``method == 'hessian'``
    modified_tol : float, default=1e-12
        Tolerance for modified LLE method.
        Only used if ``method == 'modified'``
    neighbors_algorithm : {'auto', 'brute', 'kd_tree', 'ball_tree'}, \
                          default='auto'
        algorithm to use for nearest neighbors search,
        passed to neighbors.NearestNeighbors instance
    random_state : int, RandomState instance, default=None
        Determines the random number generator when
        ``eigen_solver`` == 'arpack'. Pass an int for reproducible results
        across multiple function calls. See :term: `Glossary <random_state>`.
    n_jobs : int or None, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Attributes
    ----------
    embedding_ : array-like, shape [n_samples, n_components]
        Stores the embedding vectors
    reconstruction_error_ : float
        Reconstruction error associated with `embedding_`
    nbrs_ : NearestNeighbors object
        Stores nearest neighbors instance, including BallTree or KDtree
        if applicable.
    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import LocallyLinearEmbedding
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = LocallyLinearEmbedding(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)
    References
    ----------
    .. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
        by locally linear embedding.  Science 290:2323 (2000).
    .. [2] Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
        linear embedding techniques for high-dimensional data.
        Proc Natl Acad Sci U S A.  100:5591 (2003).
    .. [3] Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
        Embedding Using Multiple Weights.
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382
    .. [4] Zhang, Z. & Zha, H. Principal manifolds and nonlinear
        dimensionality reduction via tangent space alignment.
        Journal of Shanghai Univ.  8:406 (2004)
    """
    @_deprecate_positional_args
    def __init__(self, *, n_neighbors=5, n_components=2, reg=1E-3,
                 eigen_solver='auto', tol=1E-6, max_iter=100,
                 method='standard', hessian_tol=1E-4, modified_tol=1E-12,
                 neighbors_algorithm='auto', random_state=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self.random_state = random_state
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs

    def _fit_transform(self, G, X):

        random_state = check_random_state(self.random_state)
        
        self.embedding_, self.reconstruction_error_ = \
            locally_linear_embedding(
                G, X, n_neighbors=self.n_neighbors,
                n_components=self.n_components,
                eigen_solver=self.eigen_solver, tol=self.tol,
                max_iter=self.max_iter, method=self.method,
                hessian_tol=self.hessian_tol, modified_tol=self.modified_tol,
                random_state=random_state, reg=self.reg, n_jobs=self.n_jobs)

    def fit(self, G, X):
        """Compute the embedding vectors for data X
        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            training set.
        y : Ignored
        Returns
        -------
        self : returns an instance of self.
        """
        self._fit_transform(G, X)
        return self

    def fit_transform(self, G, X):
        """Compute the embedding vectors for data X and transform X.
        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            training set.
        y : Ignored
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self._fit_transform(G, X)
        return self.embedding_

    def transform(self, G, X):
        """
        Transform new points into embedding space.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        X_new : array, shape = [n_samples, n_components]
        Notes
        -----
        Because of scaling performed by this method, it is discouraged to use
        it together with methods that are not scale-invariant (like SVMs)
        """
        check_is_fitted(self)

        X = check_array(X)
        ind = self.nbrs_.kneighbors(X, n_neighbors=self.n_neighbors,
                                    return_distance=False)
        weights = barycenter_weights(X, self.nbrs_._fit_X, ind, reg=self.reg)
        X_new = np.empty((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            X_new[i] = np.dot(self.embedding_[ind[i]].T, weights[i])
        return X_new

#%% AE
class Autoencoder(nn.Module):
    """Makes the main denoising auto
    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(in_shape, 64),
            nn.ReLU(),
            nn.Linear(64, enc_shape)
        )
        
        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, in_shape)
        )
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

