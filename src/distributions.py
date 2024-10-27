import numpy as np
from numba import jit, njit
from scipy.special import gammaln
from scipy.stats import beta as scipy_beta
from scipy.stats import uniform as scipy_uniform


class uniform:
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale
        self._dist = scipy_uniform(loc=loc, scale=scale)

    def pdf(self, x):
        return self._dist.pdf(x)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(size=size, random_state=random_state)

    @staticmethod
    @njit
    def _fast_pdf(x, loc, scale):
        return np.where((x >= loc) & (x <= loc + scale), 1.0 / scale, 0.0)

    def fast_pdf(self, x):
        x = np.asarray(x)
        return self._fast_pdf(x, self.loc, self.scale)


class beta:
    def __init__(self, a=1.0, b=1.0, loc=0.0, scale=1.0):
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale
        self._dist = scipy_beta(a=a, b=b, loc=loc, scale=scale)

    def pdf(self, x):
        return self._dist.pdf(x)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(size=size, random_state=random_state)

    @staticmethod
    @njit
    def _fast_pdf_scalar(x, a, b, loc, scale):
        a = float(a)
        b = float(b)
        z = (x - loc) / scale
        if 0.0 <= z <= 1.0:
            log_pdf = (a - 1.0) * np.log(z) + (b - 1.0) * np.log(1.0 - z)
            log_pdf -= gammaln(a) + gammaln(b) - gammaln(a + b) + np.log(scale)
            return np.exp(log_pdf)
        else:
            return 0.0

    @staticmethod
    @njit
    def _fast_pdf_array(x, a, b, loc, scale):
        a = float(a)
        b = float(b)
        z = (x - loc) / scale
        pdf = np.zeros_like(z)
        for i in range(z.size):
            zi = z.flat[i]
            if 0.0 <= zi <= 1.0:
                log_pdf = (a - 1.0) * np.log(zi) + (b - 1.0) * np.log(1.0 - zi)
                log_pdf -= gammaln(a) + gammaln(b) - gammaln(a + b) + np.log(scale)
                pdf.flat[i] = np.exp(log_pdf)
            else:
                pdf.flat[i] = 0.0
        return pdf

    def fast_pdf(self, x):
        if np.isscalar(x):
            return self._fast_pdf_scalar(x, self.a, self.b, self.loc, self.scale)
        else:
            x = np.asarray(x)
            return self._fast_pdf_array(x, self.a, self.b, self.loc, self.scale)
