# -*- coding: utf-8 -*-
"""
Created on Tue May 20 09:12:26 2025

@author: OEM
"""

import pandas as pd

def dataframe_to_latex_table(df, caption="", label=""):
    """
    Converts a Pandas DataFrame to a LaTeX table string.

    Args:
        df (pd.DataFrame): The DataFrame to convert.
        caption (str, optional): Table caption. Defaults to "".
        label (str, optional): Table label for referencing. Defaults to "".

    Returns:
        str: LaTeX table string.
    """
    latex_string = "\\begin{table}\n"
    if caption:
        latex_string += "\\caption{" + caption + "}\n"
    if label:
        latex_string += "\\label{" + label + "}\n"
    latex_string += "\\centering\n"
    latex_string += df.to_latex(index=False, escape=False)
    latex_string += "\\end{table}"
    return latex_string

if __name__ == '__main__':

    # Example usage with reading from CSV
    try:
        df_from_csv = pd.read_csv("Structures.csv", nrows = 10)
        latex_table_from_csv = dataframe_to_latex_table(df_from_csv, caption="Data from CSV", label="tab:csv_data")
        print(latex_table_from_csv)
    except FileNotFoundError:
        print("your_data.csv not found. Please create the file or change the file name")
        
        
        
# if __name__ == '__main__':
#     # Example usage with a DataFrame
#     data = {'Name': ['Alice', 'Bob', 'Charlie'],
#             'Age': [25, 30, 28],
#             'City': ['New York', 'London', 'Paris']}
#     df = pd.DataFrame(data)

#     latex_table = dataframe_to_latex_table(df, caption="Sample Data", label="tab:sample")
#     print(latex_table)

#     # Example usage with reading from CSV
#     try:
#         df_from_csv = pd.read_csv("your_data.csv")
#         latex_table_from_csv = dataframe_to_latex_table(df_from_csv, caption="Data from CSV", label="tab:csv_data")
#         print(latex_table_from_csv)
#     except FileNotFoundError:
#         print("your_data.csv not found. Please create the file or change the file name")