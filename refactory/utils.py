import pandas as pd


def align_time(df_multi):
    return (df_multi.unstack(level=0)
            .stack(dropna=False)
            .droplevel('instrument')
            .sort_index(ascending=True))


def bundle(func, instruments):
    # 纵向组装。将func返回的dataset组装成muliIndex的dataset
    return pd.concat((func(i) for i in instruments),
                     keys=instruments,
                     names=['instrument', 'datetime'])
