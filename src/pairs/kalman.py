
import numpy as np
import pandas as pd


def kalman_hedge_ratio(log_a, log_b, delta=1e-4, obs_noise=1e-3):
    log_a = np.asarray(log_a, dtype=float)
    log_b = np.asarray(log_b, dtype=float)

    n = len(log_a)
    theta = np.zeros(2)                  # [beta, alpha]
    P = np.eye(2)                        # uncertainty of theta
    Q = delta / (1 - delta) * np.eye(2)  # how fast beta may drift
    R = obs_noise                        # price noise

    betas, alphas, errors, variances = (np.zeros(n) for _ in range(4))
    for t in range(n):
        x = np.array([log_b[t], 1.0])
        P = P + Q
        y_hat = x @ theta
        e = log_a[t] - y_hat
        S = x @ P @ x + R
        K = P @ x / S                
        theta = theta + K * e
        P = P - np.outer(K, x) @ P
        betas[t], alphas[t] = theta
        errors[t], variances[t] = e, S
    return betas, alphas, errors, variances


def spread_zscore(errors, window=60, burn_in=60):
    errors = pd.Series(errors)

    # shift(1) so today's error is judged against yesterday's history only.
    rolling = errors.shift(1).rolling(window, min_periods=window)
    z = (errors - rolling.mean()) / rolling.std()

    z.iloc[:burn_in] = np.nan
    return z
