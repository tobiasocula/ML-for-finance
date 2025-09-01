import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

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
    
    # for k in range(K):
    #     mu = mus[k,:]        # shape (N,)
    #     Sigma = Sigmas[k,:,:]  # shape (N,N)
    #     L = cholesky(Sigma, lower=True)
    #     diffs = returns - mu  # T x N
    #     y = solve_triangular(L, diffs.T, lower=True).T  # T x N
    #     quad = np.sum(y*y, axis=1)
    #     logdet = 2.0 * np.sum(np.log(np.diag(L)))
    #     logB[:, k] = -0.5 * (N*np.log(2*np.pi) + logdet + quad)
        
    # return logB  # shape T x K

    for k in range(K):
        for t in range(T):
            mu = mus[k,:]        # shape (N,)
            Sigma = Sigmas[k,:,:]  # shape (N,N)
            r = returns[t,:]
            logB[t,k] = multivariate_normal.logpdf(r, mu, Sigma)
        
    return logB



def forward_log(logB, logPi, logP):
    """
    logB: (T x K) log emission matrix
    logPi: (K,) log initial probabilities
    logP: (K x K) log transition matrix
    returns logAlpha: (T x K)
    """
    T, K = logB.shape

    alpha = np.empty((T,K))
    alpha[0,:] = logPi + logB[0,:]

    for t in range(1, T):
        for j in range(K):
            term = logsumexp(np.array([alpha[t-1,i] + logP[i,j] for i in range(K)]))
            alpha[t,j] = term + logB[t,j]

    loglik = logsumexp(alpha[-1,:])
    return alpha, loglik

def backward_log(logB, logP):
    """
    logB: (T x K)
    logP: (K x K)
    returns logBeta: (T x K)
    """
    T, K = logB.shape

    beta = np.zeros((T,K))

    for t in reversed(range(T-1)):
        for i in range(K):
            beta[t,i] = logsumexp([logP[i,j] + logB[t+1,j] + beta[t+1,j] for j in range(K)])

    return beta

def gamma_xi(alpha, beta, logB, logP):
    """
    alpha: (T x K) log alpha
    beta:  (T x K) log beta
    logB:  (T x K) log emission matrix
    logP:  (K x K) log transition matrix

    Returns:
        gamma: (T x K)
        xi:    (T-1 x K x K)
    """
    T, K = alpha.shape
    log_xi = np.empty((T-1, K, K))
    log_gamma = np.empty((T, K))

    for t in range(T-1):
        # gamma
        s = alpha[t,:] + beta[t,:]
        log_gamma[t,:] = s - logsumexp(s)

        # xi
        log_num = alpha[t, :, None] + logP + logB[t+1, :][None, :] + beta[t+1, :][None, :]
        logZ_t = logsumexp(log_num)
        log_xi[t] = log_num - logZ_t

    # last gamma
    log_gamma[T-1,:] = alpha[T-1,:] + beta[T-1,:] - logsumexp(alpha[T-1,:] + beta[T-1,:])

    return log_gamma, log_xi
"""
def Q(gamma, xi, pis, logP, logB):
    '''
    gamma: T x K
    xi: T x K
    pis: K
    logP: K x K
    rs: T x N
    logB: T x K
    '''
    term1 = np.dot(gamma[0, :] * pis)
    term2 = np.sum(xi * logP)
    term3 = np.sum(gamma * logB)
    return term1 + term2 + term3
"""

def Estep(logpi, logP, mus, Sigmas, returns):
    """
    computes expected values for hidden variables (market regime) given current parameters pi, logP, mu, Sigma
    computes posterior probabilities (gamma and xi)
    logpi: shape K
    logP: shape K x K
    mus: shape K x N
    Sigmas: shape K x N x N
    returns: T x N
    """

    logB = log_emission_matrix(returns, mus, Sigmas)
    alpha, _ = forward_log(logB, logpi, logP)
    beta = backward_log(logB, logP)
    gamma, xi = gamma_xi(alpha, beta, logB, logP)
    return gamma, xi


def update_P(gamma, xi):
    # xi: T-1 x K x K
    # gamma: T x K
    P_new = np.sum(xi, axis=0) / np.sum(gamma[:-1,:], axis=0)[:, None]
    return P_new

def update_pi(gamma):
    # gamma: T x K
    pis_new = gamma[0,:] / np.sum(gamma[0,:])  # normalize just in case
    return pis_new

def update_gaussians(r, gamma):
    T, N = r.shape
    K = gamma.shape[1]
    mu_new = np.zeros((K, N))
    Sigma_new = np.zeros((K, N, N))

    for k in range(K):
        weights = gamma[:,k][:,None]  # T x 1
        mu_new[k] = np.sum(weights * r, axis=0) / np.sum(weights)
        diffs = r - mu_new[k]
        Sigma_new[k] = (diffs.T * weights.ravel()) @ diffs / np.sum(weights)

    return mu_new, Sigma_new

def M_step(r, gamma, xi):
    pi_new = update_pi(gamma)
    P_new = update_P(gamma, xi)
    mu_new, Sigma_new = update_gaussians(r, gamma)
    return pi_new, P_new, mu_new, Sigma_new


def EM_HMM(r, K, max_iter=100, tol=1e-6, verbose=True):
    """
    r: T x N array of returns
    K: number of hidden regimes
    max_iter: maximum EM iterations
    tol: convergence tolerance for log-likelihood
    """
    T, N = r.shape
    
    # -------------------------------
    # 1️⃣ Initialize parameters
    # -------------------------------
    pi = np.ones(K) / K               # initial probabilities
    P = np.ones((K,K)) / K            # uniform transition matrix
    mu = np.random.randn(K, N)        # random initial means
    Sigma = np.array([np.eye(N) for _ in range(K)])  # identity covariances

    loglik_old = -np.inf

    for iteration in range(max_iter):
        # -------------------------------
        # 2️⃣ E-step
        # -------------------------------
        logB = log_emission_matrix(r, mu, Sigma)
        alpha, loglik = forward_log(logB, np.log(pi), np.log(P))
        beta = backward_log(logB, np.log(P))
        gamma, xi = gamma_xi(alpha, beta, logB, np.log(P))

        # -------------------------------
        # 3️⃣ M-step
        # -------------------------------
        pi, P, mu, Sigma = M_step(r, gamma, xi)

        # -------------------------------
        # 4️⃣ Check convergence
        # -------------------------------
        if verbose:
            print(f"Iteration {iteration+1}, log-likelihood = {loglik:.6f}")
        if np.abs(loglik - loglik_old) < tol:
            if verbose:
                print("Converged!")
            break
        loglik_old = loglik

    return pi, P, mu, Sigma, gamma, xi

