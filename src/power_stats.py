import numpy as np
from numpy.ma.core import concatenate
from scipy.stats import ncx2, chi2

def n_for_power(w, df=3, alpha=0.05, power=0.80):
    crit = chi2.ppf(1-alpha, df)
    lo, hi = 1, 1_000_000
    for _ in range(60):
        mid = (lo+hi)//2
        nc = mid*(w**2)
        p = 1 - ncx2.cdf(crit, df, nc)
        if p >= power: hi = mid
        else: lo = mid+1
    return lo