""" Resolve the order in which cells and bins are drawn.

Ordering is a value here, not a side effect of plotting. These functions read
an AnnData and return an index; they never modify it. Plotting functions take
the result via ``cell_order`` / ``bin_order``, so one ordering can be computed
once and handed to several panels, which is what makes those panels line up.

A tree does not compete with a sort order, it constrains one. Any internal node
of a tree can swap its children without changing the topology or a single
branch length, so a binary tree over n cells admits 2**(n-1) equally valid leaf
orders. Sort fields choose among them. See :func:`align_tree_to_order`.
"""

import copy

import numpy as np
import pandas as pd

import scgenome.refgenome
from scgenome._validate import validate_adata


class OrderConflict(ValueError):
    """ A requested cell order cannot be drawn against a tree

    Raised when the leaves of some clade do not occupy a contiguous block of
    rows in the requested order, which means no rotation of the tree reproduces
    that order. Drawing the tree anyway would render cleanly while implying
    groupings that do not exist.

    Attributes
    ----------
    clade : Bio.Phylo.BaseTree.Clade
        the clade whose leaves are split
    lo, hi : int
        first and last row occupied by the clade's leaves
    n_leaves : int
        number of leaves in the clade, less than ``hi - lo + 1``
    """

    def __init__(self, clade, lo, hi, n_leaves, fields=None):
        self.clade = clade
        self.lo = lo
        self.hi = hi
        self.n_leaves = n_leaves
        self.fields = fields

        under = f' under order {list(fields)}' if fields else ''
        super().__init__(
            f'clade of {n_leaves} cells spans rows {lo}-{hi}{under}, so it cannot be '
            f'drawn as one contiguous block. Either:\n'
            f"  on_conflict='reorder'  let the tree drive row order, with the requested "
            f'order tie-breaking within clades\n'
            f'  sort by a field that does not split clades, or drop the tree')


def tree_leaf_order(tree):
    """ Cell ids of a tree's leaves, in the order the tree currently draws them

    Equivalent to ``[t.name for t in tree.get_terminals()]``, but walked with an
    explicit stack. Biopython's own traversal recurses, so it raises
    RecursionError on trees deeper than the interpreter's limit.

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        phylogenetic tree whose leaf names are cell ids

    Returns
    -------
    list of str
        leaf names top to bottom
    """
    order = []
    stack = [tree.root]

    while stack:
        clade = stack.pop()
        if clade.clades:
            stack.extend(reversed(clade.clades))
        else:
            order.append(clade.name)

    return order


def _copy_tree(tree):
    """ Copy a tree without recursing, which copy.deepcopy cannot do when deep

    Each clade is shallow copied and its children rewired, so attributes are
    carried across while the traversal stays iterative.
    """
    new_tree = copy.copy(tree)
    new_root = copy.copy(tree.root)

    stack = [(tree.root, new_root)]
    while stack:
        old, new = stack.pop()
        new.clades = [copy.copy(child) for child in old.clades]
        stack.extend(zip(old.clades, new.clades))

    new_tree.root = new_root

    return new_tree


def align_tree_to_order(tree, order, on_conflict='raise', fields=None):
    """ Rotate a tree so its leaves follow a requested cell order

    An ordering is realizable by a tree if and only if the leaves of every
    clade occupy a contiguous block of rows. That is necessary because a depth
    first traversal always gives a subtree a contiguous interval, and
    sufficient because each node can be rotated so its children appear in
    position order.

    A single post order pass computes each clade's span over the requested
    positions and rotates that clade, so the check and the rotation cost one
    traversal together. The pass, the copy and the leaf read are all iterative,
    so ladder shaped trees, which are deep in proportion to cell count, are
    ordered without hitting the recursion limit. Drawing such a tree still
    goes through Bio.Phylo.draw, which recurses.

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        tree to align, not modified
    order : iterable of str
        cell ids in the requested row order
    on_conflict : str, optional
        'raise' to refuse an order the tree cannot reproduce, 'reorder' to
        rotate as closely as possible and let the tree win, by default 'raise'
    fields : list, optional
        obs columns the order came from, used only in the error message

    Returns
    -------
    Bio.Phylo.BaseTree.Tree
        a rotated copy of the tree

    Raises
    ------
    OrderConflict
        if ``on_conflict`` is 'raise' and some clade's leaves are split

    Notes
    -----
    Cells present in ``order`` but absent from the tree are handled by the same
    contiguity test: a non leaf cell sitting between two members of a clade
    splits that clade, and is reported as a conflict.
    """
    if on_conflict not in ('raise', 'reorder'):
        raise ValueError(
            f"unknown on_conflict {on_conflict!r}, expected 'raise' or 'reorder'")

    tree = _copy_tree(tree)
    positions = {cell_id: i for i, cell_id in enumerate(order)}
    strict = (on_conflict == 'raise')

    # Post order traversal with an explicit stack. spans maps a clade to
    # (lo, hi, n_leaves) over the requested positions of its leaves.
    spans = {}
    stack = [(tree.root, False)]

    while stack:
        clade, expanded = stack.pop()

        if not expanded:
            stack.append((clade, True))
            stack.extend((child, False) for child in clade.clades)
            continue

        if not clade.clades:
            position = positions.get(clade.name)
            if position is not None:
                spans[id(clade)] = (position, position, 1)
            continue

        placed = [(c, spans[id(c)]) for c in clade.clades if id(c) in spans]
        unplaced = [c for c in clade.clades if id(c) not in spans]

        if not placed:
            continue

        lo = min(span[0] for _, span in placed)
        hi = max(span[1] for _, span in placed)
        n_leaves = sum(span[2] for _, span in placed)

        if strict and hi - lo + 1 != n_leaves:
            raise OrderConflict(clade, lo, hi, n_leaves, fields)

        # Rotate: children in the order their leaves are requested
        clade.clades = [c for c, _ in sorted(placed, key=lambda kv: kv[1][0])] + unplaced
        spans[id(clade)] = (lo, hi, n_leaves)

    return tree


def resolve_cell_order(adata, fields=None, tree=None, on_conflict='raise'):
    """ Resolve the order in which cells are drawn

    Parameters
    ----------
    adata : AnnData
        per cell data, not modified
    fields : list, optional
        obs columns to sort on, first is primary, by default None for the
        existing obs order
    tree : Bio.Phylo.BaseTree.Tree, optional
        tree constraining the order. With ``fields``, the fields order is used
        and checked against the tree. Without, the tree's leaf order is used.
    on_conflict : str, optional
        'raise' to refuse an order the tree cannot reproduce, 'reorder' to let
        the tree drive row order with ``fields`` tie breaking within clades,
        by default 'raise'

    Returns
    -------
    pandas.Index
        cell ids in plot order

    Raises
    ------
    OrderConflict
        if ``tree`` and ``fields`` disagree and ``on_conflict`` is 'raise'

    Reads
    -----
    adata.obs[fields] : sort keys

    Examples
    --------

    >>> import scgenome
    >>> adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    >>> adata = scgenome.tl.sort_cells(adata, layer_name='copy')
    >>> order = scgenome.tl.resolve_cell_order(adata, fields=['cell_order'])
    >>> len(order) == adata.shape[0]
    True

    Hand one order to several panels so they are guaranteed to agree::

        order = scgenome.tl.resolve_cell_order(adata, fields=['cluster_id', 'cell_order'])
        scgenome.pl.plot_cell_tcn_matrix(adata, cell_order=order, ax=axes[0])
        scgenome.pl.plot_cell_matrix(adata, layer_name='copy', cell_order=order, ax=axes[1])

    """
    validate_adata(adata, caller='resolve_cell_order')

    if on_conflict == 'tangle':
        raise NotImplementedError(
            "on_conflict='tangle' arrives with the dendrogram panel. Use 'reorder' "
            "to let the tree drive row order, or 'raise' to refuse the order")

    if on_conflict not in ('raise', 'reorder'):
        raise ValueError(
            f"unknown on_conflict {on_conflict!r}, expected 'raise' or 'reorder'")

    fields = list(fields) if fields is not None else []

    missing = [f for f in fields if f not in adata.obs.columns]
    if missing:
        raise ValueError(
            f'missing obs columns {missing} for cell ordering. '
            f'Available obs columns: {list(adata.obs.columns)}')

    if fields:
        # lexsort applies the last key first, so reverse to make fields[0] primary
        keys = adata.obs[list(reversed(fields))].values.transpose()
        order = adata.obs.index[np.lexsort(keys)]
    else:
        order = adata.obs.index

    if tree is None:
        return pd.Index(order)

    leaves = tree_leaf_order(tree)
    cells = set(adata.obs.index)

    absent = [name for name in leaves if name not in cells]
    if absent:
        raise ValueError(
            f'{len(absent)} tree leaves are not cells in adata, for instance '
            f'{absent[:3]}. Prune the tree with scgenome.tl.prune_leaves, or '
            f'subset adata to the tree leaves.')

    if not fields:
        extra = [c for c in adata.obs.index if c not in set(leaves)]
        if extra:
            raise ValueError(
                f'{len(extra)} cells are not leaves of the tree and no fields were '
                f'given to order them, for instance {extra[:3]}. Pass fields= to '
                f'order all cells, or subset adata to the tree leaves.')
        return pd.Index(leaves)

    aligned = align_tree_to_order(tree, order, on_conflict=on_conflict, fields=fields)

    if on_conflict == 'reorder':
        rotated = tree_leaf_order(aligned)
        if len(rotated) != len(order):
            raise ValueError(
                f"on_conflict='reorder' needs every cell to be a tree leaf, but "
                f'{len(order) - len(rotated)} of {len(order)} cells are not. Subset '
                f'adata to the tree leaves, or use the default on_conflict=\'raise\'.')
        return pd.Index(rotated)

    return pd.Index(order)


def resolve_bin_order(adata, genome=None):
    """ Resolve the order in which genomic bins are drawn

    Parameters
    ----------
    adata : AnnData
        data with var describing genomic bins, not modified
    genome : str or RefGenomeInfo, optional
        genome version, by default resolved from ``adata.uns['genome']`` then
        the global default

    Returns
    -------
    pandas.Index
        bin ids ordered by chromosome then start position

    Reads
    -----
    adata.var['chr'], adata.var['start'] : bin positions
    adata.uns['genome'] : genome version, if set

    Examples
    --------

    >>> import scgenome
    >>> adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    >>> bins = scgenome.tl.resolve_bin_order(adata)
    >>> len(bins) == adata.shape[1]
    True

    """
    validate_adata(adata, require_var=['chr', 'start'], caller='resolve_bin_order')

    genome_info = scgenome.refgenome.get_genome_info(adata, genome=genome)

    chr_index = (
        genome_info.chromosome_info
        .astype({'chr': str})
        .set_index('chr')['chr_index'])

    bin_chr_index = adata.var['chr'].astype(str).map(chr_index)

    if bin_chr_index.isnull().any():
        unknown = sorted(set(adata.var['chr'].astype(str)) - set(chr_index.index))
        raise ValueError(
            f'mismatching chromosomes {unknown} and {list(genome_info.chromosomes)}')

    ordering = np.lexsort((adata.var['start'].values, bin_chr_index.values))

    return adata.var.index[ordering]
