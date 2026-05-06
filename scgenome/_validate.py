"""Input validation utilities for scgenome functions.

Provides clear, actionable error messages when adata is missing
required fields. Designed to help agents diagnose pipeline issues.
"""

from anndata import AnnData


def validate_adata(
    adata: AnnData,
    require_layers=None,
    require_obs=None,
    require_var=None,
    require_uns=None,
    caller=None,
):
    """Validate that an AnnData object has required fields.

    Parameters
    ----------
    adata : AnnData
        object to validate
    require_layers : list of str, optional
        layer names that must exist
    require_obs : list of str, optional
        obs column names that must exist
    require_var : list of str, optional
        var column names that must exist
    require_uns : list of str, optional
        uns keys that must exist
    caller : str, optional
        name of calling function for error messages

    Raises
    ------
    TypeError
        if adata is not an AnnData object
    ValueError
        if required fields are missing
    """
    prefix = f"{caller}: " if caller else ""

    if not isinstance(adata, AnnData):
        raise TypeError(
            f"{prefix}expected AnnData, got {type(adata).__name__}. "
            f"Load data with scgenome.pp.read_dlp_hmmcopy() or scgenome.datasets.*()."
        )

    if require_layers:
        missing = [l for l in require_layers if l not in adata.layers]
        if missing:
            available = list(adata.layers.keys())
            raise ValueError(
                f"{prefix}missing required layers {missing}. "
                f"Available layers: {available}"
            )

    if require_obs:
        missing = [c for c in require_obs if c not in adata.obs.columns]
        if missing:
            raise ValueError(
                f"{prefix}missing required obs columns {missing}. "
                f"Available obs columns: {list(adata.obs.columns)}"
            )

    if require_var:
        missing = [c for c in require_var if c not in adata.var.columns]
        if missing:
            raise ValueError(
                f"{prefix}missing required var columns {missing}. "
                f"Available var columns: {list(adata.var.columns)}"
            )

    if require_uns:
        missing = [k for k in require_uns if k not in adata.uns]
        if missing:
            raise ValueError(
                f"{prefix}missing required uns keys {missing}. "
                f"Available uns keys: {list(adata.uns.keys())}"
            )
