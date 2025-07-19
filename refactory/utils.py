import pandas as pd


def align_time(raw):
    return (raw.unstack(level=0)
            .stack(dropna=False)
            .droplevel('instrument')
            .sort_index(ascending=True))


def unstack_for_optimisation(multi_index_df):
    return (
        multi_index_df.unstack(level=0)
        .resample('1B').sum()
        .replace(0.0, pd.NA)
        .T
        .stack(dropna=False)
    )


def combine_multi(func, instruments):
    # 纵向组装。将func返回的dataset组装成muliIndex的dataset
    return pd.concat((func(i) for i in instruments),
                     keys=instruments,
                     names=['instrument', 'datetime'])
