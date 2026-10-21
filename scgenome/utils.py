import pandas as pd
import numpy as np
import collections
import scipy.sparse

from . import refgenome


def union_categories(dfs, cat_cols=None):
    """ Recreate specified categoricals on the union of categories inplace.

    Args:
        dfs (list of pandas.DataFrame): pandas dataframes to unify categoricals in-place.
    
    KwArgs:
        cat_cols (list of str): columns to unify categoricals, default None for any categorical.
    """

    # Infer all categorical columns if not given
    if cat_cols is None:
        cat_cols = set()
        for df in dfs:
            for col in df:
                if df[col].dtype.name == 'category':
                    cat_cols.add(col)

    # Get a list of categories for each column
    col_categories = collections.defaultdict(set)
    for col in cat_cols:
        for df in dfs:
            if col in df:
                col_categories[col].update(df[col].values)

    # Remove None and nan as they cannot be in the list of categories
    for col in col_categories:
        col_categories[col] = col_categories[col] - set([None, np.nan])

    # Create a pandas index for each set of categories, sort for reproducibility
    for col, categories in col_categories.items():
        col_categories[col] = pd.Index(sorted(categories))

    # Set all categorical columns as having the same set of categories
    for col in cat_cols:
        for df in dfs:
            if col in df:
                df[col] = df[col].astype('category')
                df[col] = df[col].cat.set_categories(col_categories[col])


def concat_with_categories(dfs, **kwargs):
    """ Concatenate dataframes retaining categorical columns
    """
    # Infer all categorical columns
    cat_cols = set()
    for df in dfs:
        for col in df:
            if df[col].dtype.name == 'category':
                cat_cols.add(col)

    # Get a list of categories for each column
    col_categories = collections.defaultdict(set)
    for df in dfs:
        for col in cat_cols:
            col_categories[col].update(df[col].values)

    # Remove None and nan as they cannot be in the list of categories
    for col in col_categories:
        col_categories[col] = col_categories[col] - set([None, np.nan])

    # Create a pandas index for each set of categories, sort for reproducibility
    for col, categories in col_categories.items():
        col_categories[col] = pd.Index(sorted(categories))

    # Set all categorical columns as having the same set of categories
    for df in dfs:
        for col in cat_cols:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.set_categories(col_categories[col])

    return pd.concat(dfs, **kwargs)




def create_sparse_matrices(data, row, col, values):
    """ Create sparse matrices from a long form dataframe

    Rows and columns are ordered by the sorted unique values of the `row` and
    `col` columns. Explicit zero entries are removed from each matrix.

    Parameters
    ----------
    data : pandas.DataFrame
        long form data, one row per entry
    row : str
        column giving the row of each entry
    col : str
        column giving the column of each entry
    values : list of str
        columns giving the values, one matrix each

    Returns
    -------
    matrices : dict of scipy.sparse.csr_matrix
        matrix for each column in `values`
    index : pandas.Index
        row labels, named `row`
    columns : pandas.Index
        column labels, named `col`
    """
    row_values = pd.Categorical(data[row])
    col_values = pd.Categorical(data[col])

    index = row_values.categories.rename(row)
    columns = col_values.categories.rename(col)

    matrices = {}
    for value in values:
        matrices[value] = scipy.sparse.csr_matrix(
            (data[value], (row_values.codes, col_values.codes)),
            shape=(index.size, columns.size))
        matrices[value].eliminate_zeros()

    return matrices, index, columns
