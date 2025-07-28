from scipy.optimize import minimize
import numpy as np
import random
from ..datacode.get_correct_date_range import correct_csvs

def portfolio_variance(w, cov):
    return w.T @ cov @ w

def neg_sortino(w, R, R_target=0.0):
    """
    Returns sortino ratio (positive return per downside stdev)
    """
    R_p = R @ w # (T, 1), returns per timestamp
    mean_excess_return = np.mean(R_p) - R_target
    downside_returns = np.minimum(R_p - R_target, 0)
    downside_dev = np.sqrt(np.mean(downside_returns ** 2))
    if downside_dev == 0:
        return np.inf
    return -mean_excess_return / downside_dev

def neg_sharpe(w, returns, returntarget=0.0):
    totalreturns = returns @ w
    cov = np.cov(returns, rowvar=False)
    volatility = np.sqrt(portfolio_variance(w, cov))
    if volatility == 0:
        return np.inf
    return -(np.mean(totalreturns) - returntarget) / volatility

def optimize_sharpe(returns, returntarget=0.0):
    """
    Optimal function that maximizes Sharpe ratio.
    """
    n = returns.shape[1]

    w0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    result = minimize(
        neg_sharpe, w0,
        args=(returns, returntarget),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x

def optimize_variance(R):
    """
    Optimalization function
    Determines optimal weights for minimizing variance of returns
    """
    cov = np.cov(R, rowvar=False)
    n = cov.shape[0]
    init_w = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    result = minimize(portfolio_variance, init_w, args=(cov,), bounds=bounds, constraints=cons)
    return result.x

def optimize_sortino(R, R_target=0.0):
    """
    Optimalization function
    Determines optimal asset weights for maximizing sortino ratio
    """
    n = R.shape[1] # n assets
    w0 = np.ones(n) / n  # start with equal weights

    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1  # fully invested
    })
    bounds = [(0, 1) for _ in range(n)]  # no short-selling

    result = minimize(
        neg_sortino, w0,
        args=(R, R_target),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': True}
    )

    #return result.x, -result.fun  # optimal weights, max Sortino
    return result.x

def risk_parity_loss(weights, returns):
    """
    Loss function of the risk parity method of optimizing weights.
    Goal is to minimize this.
    """
    cov = np.cov(returns, rowvar=False)
    stdev = np.sqrt(portfolio_variance(weights, cov))
    Sw = cov @ weights.T
    n = len(weights)
    return sum((weights[i] * Sw[i] / stdev - stdev / n)**2 for i in range(n))

def risk_parity(returns):
    """
    Optimalization function regarding the concept of risk parity.
    Paper: https://docslib.org/doc/5064524/an-introduction-to-risk-parity-hossein-kazemi
    This method aims to optimize asset weight allocations such that every asset's risk quantity
    contributes equally to the portfolio in question.
    """

    n = returns.shape[1]
    bounds = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n  # start with equal weights

    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1  # fully invested
    })

    result = minimize(
        risk_parity_loss, w0,
        args=(returns,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': True}
    )

    return result.x

def ENB1(returns):
    """
    Evaluation function, determines amount of risk factors.
    Eigenvals determine amount of principal components (each one
    is linked to the magnitude of the corresponding eigenvector in the
    covariance matrix's SVD).
    If all variance is concentrated in one eigenvalue (one dominant risk factor),
    then ENB ~= 1. If not, ENB is larger.
    From paper: https://portfoliooptimizer.io/blog/the-effective-number-of-bets-measuring-portfolio-diversification/
    Related to effective rank of a matrix.
    """
    cov = np.cov(returns, rowvar=False)
    L, _ = np.linalg.eig(cov)
    return sum(L)**2 / sum(k**2 for k in L)

def ENB2(returns, weights):
    """
    Evaluation function, determines amount of risk factors
    This is actually the inverse of the HH-index:
    https://en.wikipedia.org/wiki/Herfindahl%E2%80%93Hirschman_index
    """

    cov = np.cov(returns, rowvar=False)
    L, M = np.linalg.eig(cov)

    MT_w = np.matmul(M.T, weights.T)
    finalbot = 0
    bot = sum(MT_w[j]**2 * L[j] for j in range(len(weights)))
    for w_i, v_i in zip(MT_w, L):
        p_i = v_i * w_i**2 / bot
        finalbot += p_i**2
    return 1 / finalbot

def ENB3(returns):
    """
    Evaluation function for effective number of bets
    Similar to ENB2, but this uses the exact equation found in
    https://www.researchgate.net/publication/37450697_The_Effective_Rank_A_Measure_of_Effective_Dimensionality
    """
    cov = np.cov(returns, rowvar=False)
    L, M = np.linalg.eig(cov)
    norm_eigenvals = L / sum(L)
    return -sum(eigv * np.log(eigv) for eigv in norm_eigenvals)


def random_portfolios(all_tickers, subset_size, datafolders, attempts,
                     optimize_funcs, eval_funcs):
    """
    Create random portfolios out of all_tickers, with size subset_size.
    Optimize funcs: should accept returns as parameter
    Eval funcs: should accept returns and weights as params
    """

    dfs = correct_csvs(all_tickers, datafolders)

    allportfolios = []

    for _ in range(attempts):

        idx = random.sample(range(len(dfs)), k=subset_size) # list of dfs
        assets = [dfs[j] for j in idx]
        tickers = [all_tickers[j] for j in idx]
        returns = np.empty(shape=(len(dfs[0])-1, len(all_tickers)))
        for i, df in enumerate(assets):
            returns[:,i] = df['Close'].pct_change().values[1:]
        
        allmetrics = []
        allweights = []
        for opfunc in optimize_funcs:
            optimal_weights = opfunc(returns)
            allweights.append(optimal_weights)
            for evalfunc in eval_funcs:
                m = evalfunc(optimal_weights, returns)
                allmetrics.append(m)

        res = {
            'tickers': tickers,
            'allweights': allweights,
            'allmetrics': allmetrics
        }
        allportfolios.append(res)

    return allportfolios
        
def optimize_portfolio(all_tickers, subset_size, datafolders, attempts,
                     optimize_func, eval_func):
    """Similar to random_portfolios, but decides best portfolio based on optimize_func and eval_func"""
    
    dfs = correct_csvs(all_tickers, datafolders)

    best_weights = None
    best_metric = 0
    best_tickers = None

    for _ in range(attempts):

        idx = random.sample(range(len(dfs)), k=subset_size) # list of dfs
        assets = [dfs[j] for j in idx]
        tickers = [all_tickers[j] for j in idx]
        returns = np.empty(shape=(len(dfs[0])-1, len(all_tickers)))
        for i, df in enumerate(assets):
            returns[:,i] = df['Close'].pct_change().values[1:]

        optimal_weights = optimize_func(returns)
        metric = eval_func(returns, optimal_weights)
        if metric > best_metric:
            best_metric = metric
            best_weights = optimal_weights
            best_tickers = tickers

    return best_tickers, best_weights, best_metric

        
        





