import pandas as pd

def aggregate_columns_by_prefix(csv_file_path):
    # Read the input CSV
    df = pd.read_csv(csv_file_path)

    # Separate the DATE column
    date_col = df[['DATE']].copy()
    
    # Select only the data columns
    data_cols = df.columns[1:]

    # Create a dictionary to group columns by 3-character prefix
    prefix_groups = {}
    for col in data_cols:
        prefix = col[:3]
        prefix_groups.setdefault(prefix, []).append(col)

    # Aggregate columns by summing those with the same prefix
    aggregated_data = {
        prefix: df[cols].sum(axis=1)
        for prefix, cols in prefix_groups.items()
    }

    # Create a new DataFrame from the aggregated data
    aggregated_df = pd.DataFrame(aggregated_data)

    # Concatenate the DATE column with the aggregated data
    result_df = pd.concat([date_col, aggregated_df], axis=1)

    return result_df


# Example file path
csv_path = "input_data.csv"

# Get the aggregated DataFrame
aggregated_df = aggregate_columns_by_prefix(csv_path)

# Save to a new CSV (optional)
aggregated_df.to_csv("aggregated_output.csv", index=False)
