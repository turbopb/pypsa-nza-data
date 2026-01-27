# Here's an upgraded version of the function that:

# 1. **Handles missing values** (`NaN`) in the `x` or `y` columns.
# 2. **Computes the cumulative distance** between successive valid points.

# It safely skips over invalid or missing coordinates and ensures cumulative distance is correctly calculated only from valid data.
### ? Upgraded Function: `compute_cumulative_distance`

import numpy as np
import pandas as pd

def compute_cumulative_distance(df: pd.DataFrame, x_col: str = 'x', y_col: str = 'y') -> pd.Series:
    """
    Compute cumulative Euclidean distance between successive (x, y) points,
    handling missing values.

    Parameters:
    - df: pandas DataFrame containing x and y columns
    - x_col: column name for x-coordinates
    - y_col: column name for y-coordinates

    Returns:
    - A pandas Series of the same length as df with cumulative distances.
    """
    # Extract and clean coordinates
    valid_mask = df[[x_col, y_col]].notnull().all(axis=1)
    clean_df = df.loc[valid_mask].copy()
    x = clean_df[x_col].to_numpy()
    y = clean_df[y_col].to_numpy()

    # Compute deltas and distances
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    segment_distances = np.hypot(dx, dy)

    # Compute cumulative distance and align with original DataFrame
    cumulative_distance = np.insert(np.cumsum(segment_distances), 0, 0.0)
    result_series = pd.Series(index=clean_df.index, data=cumulative_distance)

    # Reindex to match original DataFrame shape, fill missing values with NaN
    full_series = pd.Series(index=df.index, dtype='float64')
    full_series.loc[result_series.index] = result_series

    return full_series

### ?? Example Usage:

# Sample DataFrame
df = pd.DataFrame({
    'x': [100, 105, 110, np.nan, 120],
    'y': [200, 205, 210, 215, 220]
})

df['cumulative_distance'] = compute_cumulative_distance(df)
print(df)


### ?? What This Does:

# * Skips rows where either `x` or `y` is missing.
# * Computes distance only between valid consecutive points.
# * Returns a column where invalid rows contain `NaN`, and valid rows contain the growing cumulative length.

# ---

# Let me know if you'd like:

# * 2D plotting of the segments
# * Line segment labels
# * Grouping by route/line name to compute multiple routes in one DataFrame
