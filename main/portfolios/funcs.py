from scipy.optimize import minimize
import numpy as np
import random
from ..datacode.get_correct_date_range import correct_csvs

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

    def portfolio_variance(w):
        return w.T @ cov @ w

    result = minimize(portfolio_variance, init_w, bounds=bounds, constraints=cons)
    return result.x

def sortino_loss(w, R, R_target=0.0):
    """
    Returns sortino ratio (positive return per downside stdev)
    """
    R_p = R @ w # (T, 1), returns per timestamp
    mean_excess_return = np.mean(R_p) - R_target
    downside_returns = np.minimum(R_p - R_target, 0)
    downside_dev = np.sqrt(np.mean(downside_returns ** 2))
    if downside_dev == 0:
        return np.inf  # avoid divide by zero
    return -mean_excess_return / downside_dev  # minimize negative Sortino

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
        sortino_loss, w0,
        args=(R, R_target),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': True}
    )

    #return result.x, -result.fun  # optimal weights, max Sortino
    return result.x

def ENB2(returns, weights):
    """
    Evaluation function, determines amount of risk factors
    """

    cov = np.cov(returns, rowvar=False)
    L, M = np.linalg.eig(cov)

    MT_w = np.matmul(M.T, weights.T)
    finalbot = 0
    bot = sum(MT_w[j]**2 * L[j] for j in range(len(weights)))
    for w_i, v_i in zip(MT_w, L):
        p_i = v_i * w_i**2 / bot
        finalbot += p_i**2
    return 1/finalbot

def random_portfolios(all_tickers, subset_size, datafolders, attempts,
                     optimize_funcs, eval_funcs):
    """
    Create random portfolios out of all_tickers, with size subset_size.
    Optimize funcs: should accept returns as parameter
    Eval funcs: should accept returns and weights as params
    """

    dfs = correct_csvs(all_tickers, datafolders)

    print('lengths:'); print([len(k) for k in dfs])

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
            'allweights': allweights, # is list of lists
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

    for j in range(attempts):

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

        
        





