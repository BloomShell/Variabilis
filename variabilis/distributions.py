# Distributions...
import numpy as np
from scipy.special import gammaln
from scipy.stats import kurtosis


class Normal(object):

    @staticmethod
    def log_likelihood(resids, sigma2):
        return np.sum(0.5 * (np.log(2 * np.pi) + np.log(sigma2) + resids ** 2 / sigma2))


class StudentsT(object):

    @staticmethod
    def log_likelihood(resids, sigma2, nu):
        log_lik = gammaln((nu + 1) / 2) - gammaln(nu / 2) - np.log(np.pi * (nu - 2)) / 2
        log_lik -= 0.5 * (np.log(sigma2))
        log_lik -= ((nu + 1) / 2) * (np.log(1 + (resids ** 2.0) / (sigma2 * (nu - 2))))
        return np.sum(log_lik) * (-1)

    @staticmethod
    def nu(returns):
        k = kurtosis(returns, fisher=False)
        return max((4.0 * k - 6.0) / (k - 3.0) if k > 3.75 else 12.0, 4.0)