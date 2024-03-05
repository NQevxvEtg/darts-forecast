import numpy as np
from scipy.special import boxcox1p, inv_boxcox1p

def log_transform(df):
    # copy the dataframe
    tdf = df.copy()
    # apply log scaling
    for column in tdf.columns:
        tdf[column] = np.log(tdf[column])

    tdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    tdf = tdf.astype(float)
    tdf = tdf.interpolate(method='linear', axis=0).ffill().bfill()
#     tdf = tdf.dropna(axis='columns')
    tdf = tdf.fillna(0)

    return tdf

def boxcox(df):
    # copy the dataframe
    _df = df.copy()

    # apply log scaling
    for column in _df.columns:

#         tdf[column] = stats.boxcox(tdf[column])
        _df[column] = boxcox1p(_df[column], 0.25)

#     _df = _df.dropna(axis='columns')
    _df = _df.fillna(0)
    _df = _df.astype(float)

    return _df

def min_max_scaling(df):
    # copy the dataframe
    tdf = df.copy()
    # apply min-max scaling
    for column in tdf.columns:
        tdf[column] = (tdf[column] - tdf[column].min()) / (tdf[column].max() - tdf[column].min())
#     tdf = tdf.dropna(axis='columns')
    tdf = tdf.fillna(0)
    return tdf

def unwrap_df(df):
    udf = df.copy()
    def unwrap_col(col):
        # udf[col] =  np.unwrap(2 * udf[col]) / 2
        udf[col] =  np.unwrap(udf[col])
    cols = udf.columns
    list(map(lambda col: unwrap_col(col), cols))
    return udf

