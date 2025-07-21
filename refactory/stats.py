import pandas as pd
import numpy as np
from scipy.stats import skew as scipy_skew


def min(net):
    return np.nanmin(net)


def max(net):
    return np.nanmax(net)


def mean(net):
    return np.nanmean(net)


def median(net):
    return np.nanmedian(net)


def std(net):
    return float(net.std())


def skew(net):
    return scipy_skew(net)


def avg_losses(net):
    x = net[net < 0]
    return np.mean(x)


def avg_gains(net):
    x = net[net > 0]
    return np.mean(x)


def gaintolossratio(net):
    return avg_gains(net) / -avg_losses(net)


def profitfactor(net):
    return np.sum(avg_gains(net)) / -np.sum(avg_losses(net))


def ann_mean(net):
    year_num = len(net) / 256
    return sum(net) / year_num


def ann_vol(net):
    return net.std() * 16


def sharpe(net):
    mean = ann_mean(net)
    vol = ann_vol(net)
    sharpe = mean / vol
    return sharpe


def drawdown(net):
    x = net.cumsum().ffill()
    maxx = x.expanding(min_periods=1).max()
    return x - maxx


def avg_drawdown(net):
    drawdown_ = drawdown(net)
    return np.nanmean(drawdown_)

def worst_drawdown(net):
    drawdown_ = drawdown(net)
    return np.nanmin(drawdown_)


def calmar(net):
    return ann_mean(net) / -worst_drawdown(net)


def avg_return_to_drawdown(net):
    return ann_mean(net) / -avg_drawdown(net)


