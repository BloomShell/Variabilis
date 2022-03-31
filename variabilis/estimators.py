import numpy as np
from utils import moving_average


def Parkinson(df, window= 7, ddof=0):
    n = df.shape[0]
    hl = np.array(np.log(df.High/df.Low))
    vol = np.zeros(shape=(n))
    for i in range(0, n-window+1):
        vol[window+i-1] = np.sqrt(np.sum(hl[i:(window+i)]**2)/((window-ddof)*np.log(2)*4))
    return vol


def RogersSatchell(df, window= 7):
    n = df.shape[0]
    u = np.array(np.log(df.High/df.Open))
    d = np.array(np.log(df.Low/df.Open))
    c = np.array(np.log(df.Close/df.Open))
    rs =  u * (u-c) + d * (d-c)
    return np.sqrt(moving_average(rs, window= window, ddof=1))


def YangZhang(df, window= 7):
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    u = np.array(np.log(df.High/df.Open))
    d = np.array(np.log(df.Low/df.Open))
    c = np.array(np.log(df.Close/df.Open))
    o = np.array(np.log(df.Open/df.Close.shift(1)))
    cc = np.array(np.log(df.Close/df.Close.shift(1)))
    rs =  u * (u-c) + d * (d-c)
    cc_sq = np.nan_to_num(cc**2)
    o_sq = np.nan_to_num(o**2)
    cc_rv = moving_average(cc_sq, window= window, ddof=1)
    o_rv = moving_average(o_sq, window= window, ddof=1)
    rs_rv = moving_average(rs, window= window, ddof=1)
    return np.sqrt(o_rv + k * cc_rv + (1 - k) * rs_rv)


def GarmanKlass(df):
    c = np.array(np.log(df.Close/df.Open))
    hl = np.array(np.log(df.High/df.Low))
    return np.sqrt(0.5 * hl**2 - (2*np.log(2)-1) * c**2)