import numpy as np
from scipy.signal import savgol_filter
import talib
#@@@@@@@@@@@@@@@@@@@@@@@@
#    Custom indicators
#@@@@@@@@@@@@@@@@@@@@@@@@


#------------------------------------------------------------------------------------------

# HRV: Heart rate variability - RMSSD: Root mean square of the successive differences

def rmssd(data):
    delta = data.shift(0) - data.shift(1) 
    delta *= delta
    return np.sqrt(delta / 2)

def lnrmssd(data):
    delta = data.shift(0) - data.shift(1) 
    delta *= delta
    return np.log(np.sqrt(delta / 2))



#------------------------------------------------------------------------------------------

# Local Hurst Exponent Approximation
# https://www.youtube.com/watch?v=QpH-YPin01k&t=793s
# https://usethinkscript.com/threads/local-hurst-exponent-for-thinkorswim.2480/ 

def LHEA(high, low, length):
    tr = high - low
    atr = tr.rolling(length).mean()
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    H = (np.log(hh - ll) - np.log(atr)) / (np.log(length))
    return H



#------------------------------------------------------------------------------------------

def reverse_ema(y):

    y = y.rename("reverse_ema", inplace=True)
    
    aa = .1
    cc = 1 - aa

    ma = aa * y + cc * y.shift(1).interpolate(method='linear', limit_direction='both')
    r1 = cc * ma + ma.shift(1).interpolate(method='linear', limit_direction='both')
    r2 = np.power(cc, 2)   * r1 + r1.shift(1).interpolate(method='linear', limit_direction='both')
    r3 = np.power(cc, 4)   * r2 + r2.shift(1).interpolate(method='linear', limit_direction='both')
    r4 = np.power(cc, 8)   * r3 + r3.shift(1).interpolate(method='linear', limit_direction='both')
    r5 = np.power(cc, 16)  * r4 + r4.shift(1).interpolate(method='linear', limit_direction='both')
    r6 = np.power(cc, 32)  * r5 + r5.shift(1).interpolate(method='linear', limit_direction='both')
    r7 = np.power(cc, 64)  * r6 + r6.shift(1).interpolate(method='linear', limit_direction='both')
    r8 = np.power(cc, 128) * r7 + r7.shift(1).interpolate(method='linear', limit_direction='both')
    wa = ma - aa * r8

    # wa = wa.interpolate(method='linear', limit_direction='both', limit=1)
    
    return wa

#------------------------------------------------------------------------------------------

def gann_slope(high, low, close):
    c0 = (high + low + close) / 3
    a1 = c0.rolling(15).max()
    a2 = c0.rolling(15).min()
    a3 = a1 - a2
    gann_slope = ((c0 - a2) / a3).ewm(span=2, adjust=False).mean() * 100
    return gann_slope
#------------------------------------------------------------------------------------------
maxint = 9
def vortex(n):
    # return sum([int(x) for x in list(str(abs(int(float(str(n).replace(".",""))))))])
    return sum([int(x) for x in list(str(abs(int(float(n)))))])

def blackhole(n):
    if np.isnan(n):
        return np.nan
    if n > 0:
        if vortex(n) > maxint:
            return blackhole(vortex(n))
        if vortex(n) <= maxint:
            return vortex(n)
    else:
        if vortex(n) > maxint:
            return blackhole(vortex(n))*-1
        if vortex(n) <= maxint:
            return vortex(n)*-1

#------------------------------------------------------------------------------------------


def savgol(df):
    # copy the dataframe
    _df = df.copy()

    # apply log scaling
    for column in _df.columns:
        
        _df[column] = savgol_filter(_df[column], 9, 1)

#     _df = _df.dropna(axis='columns')
    _df = _df.fillna(0)
    _df = _df.astype(float)
    return _df

#------------------------------------------------------------------------------------------


def mtf_tsi(price, long_length=25, short_length=13, signal_length=13):
    
    # Calculate the TSI
    pc = price.diff()
    double_smoothed_pc = talib.EMA(talib.EMA(pc, long_length), short_length)
    double_smoothed_abs_pc = talib.EMA(talib.EMA(abs(pc), long_length), short_length)
    tsi_value = 100 * (double_smoothed_pc / double_smoothed_abs_pc)
    tsi_signal = talib.EMA(talib.EMA(tsi_value, signal_length), signal_length)

    tsi_value = tsi_value.interpolate(method='linear', limit_direction='both')
    tsi_signal = tsi_signal.interpolate(method='linear', limit_direction='both')
    return tsi_value, tsi_signal


#------------------------------------------------------------------------------------------
def fedNetLiquidity(fed_bal,tga,rev_repo):
    units = 1e6 # millions
    # units = 1e9 # billions
    # units = 1e12 # trillions

    net_liquidity_offset = 10 # 2-week offset for use with daily charts
    
    net_liquidity = (fed_bal - (tga + rev_repo)) / units

    net_liquidity = net_liquidity.shift(periods=net_liquidity_offset)

    # net_liquidity = net_liquidity.rename(columns={'close': 'net_liquidity'})
    net_liquidity.rename("net_liquidity", inplace=True)

    return net_liquidity
    