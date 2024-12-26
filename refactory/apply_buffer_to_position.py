import numpy as np
import pandas as pd


def apply_buffer(position_raw, volatility_scalar, buffer_size):
    buffer = volatility_scalar * buffer_size
    top_pos = (position_raw + buffer).round()
    bottom_pos = (position_raw - buffer).round()
    position_raw = position_raw.round()

    last = position_raw.values[0]
    if np.isnan(last):
        last = 0.0
    buffered_position_list = [last]
    for index in range(len(position_raw))[1:]:
        last = adjust_by_buffer(last, position_raw.values[index],
                                top_pos.values[index], bottom_pos.values[index])
        buffered_position_list.append(last)
    buffered_position = pd.Series(buffered_position_list, index=position_raw.index)
    # last = position_raw.shift(1).bfill()
    # df = pd.DataFrame({'last': last, 'current': position_raw, 'top': top_pos, 'bottom': bottom_pos})
    # buffered_position = df.apply(lambda x: adjust_by_buffer(x['last'], x['current'], x['top'], x['bottom']), axis=1)

    return buffered_position


def adjust_by_buffer(last, current, top, bottom, trade_to_edge=True):
    if np.isnan(top) or np.isnan(bottom) or np.isnan(current):
        return last

    if last > top:
        if trade_to_edge:
            return top
        else:
            return current
    elif last < bottom:
        if trade_to_edge:
            return bottom
        else:
            return current
    else:
        return last
