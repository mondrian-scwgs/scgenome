import sklearn.decomposition
import sklearn.preprocessing
import pandas as pd
import anndata as ad
import numpy as np

import scgenome.preprocessing.transform

from anndata import AnnData


def pca_loadings(adata: AnnData, layer=None, n_components=None, random_state=100) -> AnnData:
    """ Compute PCA and store results in adata.

    Parameters
    ----------
    adata : AnnData
        Copy number or other cell matrix
    layer : str, optional
        layer to use, by default None, use .X
    n_components : int, optional
        sklearn.decomposition.PCA n_components parameter, by default None
    random_state : int, optional
        sklearn.decomposition.PCA random_state parameter, by default 100

    Returns
    -------
    AnnData
        input AnnData with PCA results stored in obsm, varm, and uns

    Reads
    -----
    adata.layers[layer] : matrix to decompose (or adata.X if layer is None)

    Modifies
    --------
    adata.obsm['X_pca'] : transformed coordinates (cells x components)
    adata.varm['PCs'] : PCA loadings (bins x components)
    adata.uns['pca'] : dict with variance, variance_ratio, params, etc.
    """

    if layer is None:
        data = adata.X
    else:
        data = adata.layers[layer]

    scaler = sklearn.preprocessing.StandardScaler()
    pca = sklearn.decomposition.PCA(n_components=n_components, random_state=random_state)
    transformed = pca.fit_transform(scaler.fit_transform(scgenome.preprocessing.transform.fill_missing(data)))

    # Store transformed coordinates (cells x components)
    adata.obsm['X_pca'] = transformed

    # Store loadings (bins x components)
    adata.varm['PCs'] = pca.components_.T

    # Store PCA metadata
    adata.uns['pca'] = {
        'params': {
            'layer': layer,
            'n_components': n_components,
            'random_state': random_state,
        },
        'variance': pca.explained_variance_,
        'variance_ratio': pca.explained_variance_ratio_,
        'singular_values': pca.singular_values_,
        'n_components': pca.n_components_,
        'n_samples': pca.n_samples_,
        'noise_variance': pca.noise_variance_,
        'n_features_in': pca.n_features_in_,
        'mean': pca.mean_,
    }

    return adata
