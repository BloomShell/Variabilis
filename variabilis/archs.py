# Volaitily procesess...
import numpy as np
from scipy.optimize import minimize
from distributions import Normal, StudentsT
from recursions import gjrgarch_recursive, garch_recursion


ARCH_BOUNDS = \
{
    "alpha": (0, 1),
    "beta": (0, 1),
    "delta": (0, 1),
    "theta": (-1, 2),
    "nu": (2, 500),
}


class GARCH(object):

    """

    parameters
    --------------------------
    parameters.0 = mu
    parameters.1 = omega
    parameters.2 = alpha
    parameters.3 = beta
    parameters.4 = nu

    """

    def __init__(self, dist="normal"):

        GARCH.__cotrols__(dist)
        self.distribution = Normal if dist == "normal" else StudentsT
        if self.distribution == Normal:
            self.fun = GARCH.gaussian_variance
        else:
            self.fun = GARCH.students_variance
        self.constraints = ({"fun": lambda parameters: np.array(
            [1 - parameters[2] - parameters[3]]), "type": "ineq"})
        self.fitted = False

    @staticmethod
    def __cotrols__(dist):
        if not dist in ["normal", "t"]:
            return ValueError("Not recognized distribution; Allowed are: {'normal', 't'}")

    @staticmethod
    def gaussian_variance(parameters, returns):
        sigma2, resids = garch_recursion(parameters, returns)
        return Normal.log_likelihood(resids, sigma2)

    @staticmethod
    def students_variance(parameters, returns):
        sigma2, resids = garch_recursion(parameters, returns)
        return StudentsT.log_likelihood(resids, sigma2, parameters[4])

    def fit(self, returns):
        m = returns.mean()
        v = returns.var()

        if self.distribution == Normal:
            x0 = [m, v * .01, .03, .90],
            bounds = [(-10 * m, 10 * m), (np.finfo(np.float64).eps, 2*v),
                           ARCH_BOUNDS['alpha'], ARCH_BOUNDS['beta']]
        elif self.distribution == StudentsT:
            x0 = [m, v * .01, .03, .90, self.distribution.nu(returns)],
            bounds = [(-10 * m, 10 * m), (np.finfo(np.float64).eps, 2*v),
                           ARCH_BOUNDS['alpha'], ARCH_BOUNDS['beta'], ARCH_BOUNDS['nu']]

        # Minimize the LL function in terms of returns...
        self.results = minimize(
            fun=self.fun,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=self.constraints,
            args=(returns,),
            options={'disp': False},
            tol=1e-8,
        )

        # Fitted parameters...
        self.ll = self.results.fun
        self.parameters = self.results.x
        self.sigma2, self.resids = garch_recursion(self.parameters, returns)
        self.conditional_volatility = np.sqrt(self.sigma2)
        self.fitted = True

    def forecast(self, horizon):
        if not self.fitted:
            raise ValueError("[G-ARCH] Must fit the model before calling forecast.")
        forward_resids = np.zeros(shape=(horizon))
        forward_sigma2 = np.zeros(shape=(horizon))
        forward_sigma2[0] = self.parameters[1] + self.parameters[2] * self.resids[-1] ** 2 + \
                            self.parameters[3] * self.conditional_volatility[-1] ** 2
        forward_resids[0] = np.sqrt(forward_sigma2[0])
        for t in range(1, horizon):
            forward_sigma2[t] = self.parameters[1] + self.parameters[2] * forward_resids[t - 1] ** 2 + \
                    self.parameters[3] * forward_sigma2[t - 1]
            forward_resids[t] = np.sqrt(forward_sigma2[t])

        return forward_sigma2


class GJRGARCH(object):
    """

    parameters
    -----------------------
    parameters.0 = mu
    parameters.1 = omega
    parameters.2 = alpha
    parameters.3 = theta
    parameters.4 = beta
    parameters.5 = nu

    """

    def __init__(self, dist="normal"):

        GJRGARCH.__cotrols__(dist)
        self.distribution = Normal if dist == "normal" else StudentsT
        self.fun = GJRGARCH.gaussian_variance if self.distribution == Normal else GJRGARCH.students_variance
        self.constraints = ({"fun": lambda parameters: np.array(
            [1 - parameters[2] - parameters[3] / 2 - parameters[4]]), "type": "ineq"})
        self.fitted = False

    @staticmethod
    def __cotrols__(dist):
        if not dist in ["normal", "t"]:
            return ValueError("Not recognized distribution; Allowed are: {'normal', 't'}")

    @staticmethod
    def gaussian_variance(parameters, returns):
        sigma2, resids = gjrgarch_recursive(parameters, returns)
        return Normal.log_likelihood(resids, sigma2)

    @staticmethod
    def students_variance(parameters, returns):
        sigma2, resids = gjrgarch_recursive(parameters, returns)
        return StudentsT.log_likelihood(resids, sigma2, parameters[5])

    def fit(self, returns):

        m = returns.mean()
        v = returns.var()

        if self.distribution == Normal:
            x0 = [m, v * .01, .03, .09, .90],
            bounds = [(-10 * m, 10 * m), (np.finfo(np.float64).eps, 2 * v),
                           ARCH_BOUNDS['alpha'], ARCH_BOUNDS['theta'],
                      ARCH_BOUNDS['beta']]

        elif self.distribution == StudentsT:
            x0 = [m, v * .01, .03, .09, .90,
                       self.distribution.nu(returns)],
            bounds = [(-10 * m, 10 * m), (np.finfo(np.float64).eps, 2 * v),
                           ARCH_BOUNDS['alpha'], ARCH_BOUNDS['theta'],
                      ARCH_BOUNDS['beta'], ARCH_BOUNDS['nu']]

        # Minimize the LL function in terms of returns...
        self.results = minimize(
            fun=self.fun,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=self.constraints,
            args=(returns),
            options={'disp': False},
            tol=1e-8,
        )

        # Fitted parameters...
        self.ll = self.results.fun
        self.parameters = self.results.x
        self.sigma2, self.resids = gjrgarch_recursive(self.parameters, returns)
        self.conditional_volatility = np.sqrt(self.sigma2)
        self.fitted = True

    def forecast(self, horizon):
        if not self.fitted:
            raise ValueError("[G-ARCH] Must fit the model before calling forecast.")

        forward_resids = np.zeros(shape=(horizon))
        forward_asym_resids = np.zeros(shape=(horizon))
        forward_sigma2 = np.zeros(shape=(horizon))
        forward_sigma2[0] = self.parameters[1] + self.parameters[2] * self.resids[-1] ** 2 + \
                self.parameters[3] * self.resids[-1] ** 2 * (self.resids[-1]<0) + \
                self.parameters[4] * self.conditional_volatility[-1] ** 2

        forward_resids[0] = np.sqrt(forward_sigma2[0])
        forward_asym_resids[0] = np.sqrt(0.5 * forward_sigma2[0])
        for t in range(1, horizon):
            forward_sigma2[t] = self.parameters[1] + self.parameters[2] * forward_resids[t - 1] ** 2 + \
                self.parameters[3] * forward_asym_resids[t - 1] ** 2 + \
                self.parameters[4] * forward_sigma2[t - 1]

            forward_resids[t] = np.sqrt(forward_sigma2[t])
            forward_asym_resids[t] = np.sqrt(forward_sigma2[t] * 0.5)

        return forward_sigma2