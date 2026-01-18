import pandas as pd

def compute_load_duration_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Load Duration Curve (LDC) from a time series of load data.

    Workflow:
    1. Input: A pandas DataFrame with two columns:
       - First column: time-datestamps (not directly used in LDC computation, 
         but retained for traceability).
       - Second column: load values (e.g., energy demand at each timestamp).

    2. The load values are sorted in descending order to represent the 
       duration curve. This removes the time sequence but preserves 
       the statistical distribution of load magnitudes.

    3. A rank or duration index (from 1 to N) is assigned to each load value, 
       corresponding to the fraction of time the load is at or above that level.

    4. Output: A new DataFrame with:
       - "rank": position in the sorted order (1 = peak load).
       - "duration_fraction": fraction of total time for which the load 
         is exceeded (values between 0 and 1).
       - "load": the corresponding load value.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with two columns: [timestamp, load].

    Returns
    -------
    ldc_df : pandas.DataFrame
        DataFrame representing the Load Duration Curve, with columns:
        ["rank", "duration_fraction", "load"].
    """
    if df.shape[1] < 2:
        raise ValueError("Input DataFrame must contain at least two columns: [timestamp, load].")

    # Extract load column (assumed to be the second column)
    load = df.iloc[:, 1]

    # Sort load values in descending order
    sorted_load = load.sort_values(ascending=False).reset_index(drop=True)

    # Create rank and duration fraction
    n = len(sorted_load)
    rank = range(1, n + 1)
    duration_fraction = [(i / n) for i in rank]

    # Build LDC DataFrame
    ldc_df = pd.DataFrame({
        "rank": rank,
        "duration_fraction": duration_fraction,
        "load": sorted_load
    })

    return ldc_df


# Example data
data = {
    "timestamp": pd.date_range("2024-01-01", periods=5, freq="H"),
    "load": [120, 150, 100, 180, 160]
}
df = pd.DataFrame(data)

ldc_df = compute_load_duration_curve(df)
print(ldc_df)
