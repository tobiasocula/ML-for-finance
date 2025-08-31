import numpy as np
from scipy.linalg import cholesky, solve_triangular

def log_emission_matrix(returns, mus, Sigmas):
    """
    returns: T x N array of returns
    mus: K x N mean returns (over time) per regime and asset
    Sigmas: K x N x N the covariance matrix per regime

    returns: logB, T x K, where logB[t,k] = log p(r_t | s_t = k)
    """

    T, N = returns.shape
    K = len(mus)
    logB = np.empty((T, K))
    
    for k in range(K):
        mu = mus[k,:]        # shape (N,)
        Sigma = Sigmas[k,:,:]  # shape (N,N)
        L = cholesky(Sigma, lower=True)
        diffs = returns - mu  # T x N
        y = solve_triangular(L, diffs.T, lower=True).T  # T x N
        quad = np.sum(y*y, axis=1)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        logB[:, k] = -0.5 * (N*np.log(2*np.pi) + logdet + quad)
        
    return logB  # shape T x K