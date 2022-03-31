cimport numpy
cimport cython
import numpy
from libc cimport pow
numpy.import_array()


@cython.boundscheck(False)
def garch_recursion(
    double[::1] params,
    double[::1] returns
):
    cdef Py_ssize_t t
    cdef int T = returns.shape[0]

    cdef double mu = params[0]
    cdef double omega = params[1]
    cdef double alpha = params[2]
    cdef double beta = params[3]

    cdef double[::1] resids = numpy.subtract(numpy.array(returns), mu)
    cdef double[::1] sigma2 = numpy.zeros(shape=(T,))

    cdef int tau = min(75, T)
    cdef numpy.ndarray[double, ndim=1] w = 0.94 ** numpy.arange(tau)
    w = numpy.divide(w, numpy.sum(w))
    cdef double backcast = numpy.sum((pow(numpy.abs(resids[:tau]), 2)) * w)
    sigma2[0] = backcast

    # Compute the variance recursively...
    for t in range(1, T):
        sigma2[t] = omega + alpha * resids[t - 1] ** 2 + beta * sigma2[t - 1]
    return numpy.asarray(sigma2), numpy.asarray(resids)


@cython.boundscheck(False)
def gjrgarch_recursive(
    double[::1] params,
    double[::1] returns
):
    cdef Py_ssize_t t
    cdef int T = returns.shape[0]

    cdef double mu = params[0]
    cdef double omega = params[1]
    cdef double alpha = params[2]
    cdef double theta = params[3]
    cdef double beta = params[4]

    cdef double[::1] resids = numpy.subtract(numpy.array(returns), mu)
    cdef double[::1] sigma2 = numpy.zeros(shape=(T,))

    cdef int tau = min(75, T)
    cdef numpy.ndarray[double, ndim=1] w = 0.94 ** numpy.arange(tau)
    w = numpy.divide(w, numpy.sum(w))
    cdef double backcast = numpy.sum((pow(numpy.abs(resids[:tau]), 2)) * w)
    sigma2[0] = backcast

    # Compute the variance recursively...
    for t in range(1, T):
        sigma2[t] = omega + alpha * resids[t - 1] ** 2 + beta * sigma2[t - 1] + \
            theta * resids[t - 1] ** 2 * (resids[t - 1]<0)
    return numpy.asarray(sigma2), numpy.asarray(resids)