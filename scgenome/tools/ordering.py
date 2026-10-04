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
    clade : ``Bio.Phylo`` clade
        the clade whose leaves are split
    lo, hi : int
        first and last row occupied by the clade's leaves
    n_leaves : int
        number of leaves in the clade, less than ``hi - lo + 1``
    remedy : str, optional
        what to suggest in the message, since a tree and a dendrogram have
        different ways out
    """

    #: What to suggest when a tree cannot reproduce a requested order
    TREE_REMEDY = (
        "pass on_conflict='reorder' to let the tree drive the order instead, or "
        'align_tree_to_groups to gather a label as far as the tree allows')

    #: What to suggest when a linkage cannot be drawn against an order
    DENDROGRAM_REMEDY = (
        'order the rows by the clustering this linkage came from, for instance '
        "resolve_cell_order(adata, fields=['cell_order']), or drop the dendrogram")

    def __init__(self, clade, lo, hi, n_leaves, fields=None, remedy=None):
        self.clade = clade
        self.lo = lo
        self.hi = hi
        self.n_leaves = n_leaves
        self.fields = fields

        under = f' under order {list(fields)}' if fields else ''
        remedy = remedy if remedy is not None else self.TREE_REMEDY

        super().__init__(
            f'clade of {n_leaves} cells spans rows {lo}-{hi}{under}, so it cannot '
            f'be drawn as one contiguous block: {remedy}.')


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
    list
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
    order : pandas.Index or list
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
    scgenome.tl.OrderConflict
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



#: Above this many groups the exact rotation gets expensive, so 'auto' falls back
MAX_OPTIMAL_GROUPS = 12


def _group_ranks(groups, group_order):
    """ Map each group to an integer, in the order they should be laid out """
    if group_order is None:
        if isinstance(groups.dtype, pd.CategoricalDtype):
            group_order = list(groups.cat.categories)
        else:
            group_order = sorted(groups.dropna().unique())

    rank = {group: i for i, group in enumerate(group_order)}

    unknown = sorted(set(groups.dropna().unique()) - set(rank))
    if unknown:
        raise ValueError(
            f'groups {unknown} are not in group_order {list(group_order)}')

    return rank


def _rotate_greedy(tree, groups, rank):
    """ Rotate by sorting cells on their group, cheap but not always the fewest blocks """
    keys = np.array([rank.get(g, len(rank)) for g in groups])
    order = groups.index[np.argsort(keys, kind='stable')]

    return align_tree_to_order(tree, order, on_conflict='reorder')


def _gather_states(postorder, children_of, label_of, rank):
    """ Cheapest arrangement of every subtree, for each pair of end labels

    The shared core of gathering a label under a fixed topology. For each node
    it tabulates, for every (first label, last label) its subtree can end up
    with, the fewest label changes needed to get there. A node's cost is its
    children's costs plus one wherever two adjacent children meet on different
    labels, so one pass in post order solves the whole tree.

    Parameters
    ----------
    postorder : iterable
        node keys, children before parents
    children_of : callable
        node key to a list of child node keys, empty for a leaf
    label_of : callable
        leaf node key to its group label
    rank : dict
        group label to its position, used only to break ties

    Returns
    -------
    tuple
        ``(table, root_state)``, where table maps a node to
        ``{(first, last): (cost, payload)}`` and payload is
        ``(ordered_children, left_state, right_state)`` or None for a leaf
    """
    table = {}
    root = None

    for node in postorder:
        root = node
        kids = [c for c in children_of(node) if c in table]

        if not children_of(node):
            label = label_of(node)
            if label is not None:
                table[node] = {(label, label): (0, None)}
            continue

        extras = [c for c in children_of(node) if c not in table]

        if not kids:
            continue

        if len(kids) == 1:
            table[node] = {
                state: (cost, ([kids[0]] + extras, state, None))
                for state, (cost, _) in table[kids[0]].items()}
            continue

        # A multifurcation has too many arrangements to enumerate, so order the
        # extra children by group and solve the remaining pair exactly
        if len(kids) > 2:
            kids.sort(key=lambda c: min(
                (rank.get(first, len(rank)) for first, _ in table[c]), default=len(rank)))
            extras = kids[2:] + extras
            kids = kids[:2]

        best = {}
        for left, right in ((kids[0], kids[1]), (kids[1], kids[0])):
            for (first, left_last), (left_cost, _) in table[left].items():
                for (right_first, last), (right_cost, _) in table[right].items():
                    cost = left_cost + right_cost + (0 if left_last == right_first else 1)
                    key = (first, last)
                    if key not in best or cost < best[key][0]:
                        best[key] = (
                            cost,
                            ([left, right] + extras,
                             (first, left_last), (right_first, last)))

        table[node] = best

    if root is None or root not in table:
        return table, None

    # Fewest blocks wins; rank breaks ties, so group_order decides which way
    # round the groups go whenever that costs nothing
    states = table[root]
    root_state = min(states, key=lambda k: (states[k][0], rank.get(k[0], len(rank))))

    return table, root_state


def _rotate_optimal(tree, labels, rank):
    """ Rotate a tree's children to the arrangement with the fewest label blocks """
    tree = _copy_tree(tree)

    postorder = []
    stack = [(tree.root, False)]
    while stack:
        clade, expanded = stack.pop()
        if expanded:
            postorder.append(clade)
            continue
        stack.append((clade, True))
        stack.extend((child, False) for child in clade.clades)

    by_key = {id(c): c for c in postorder}
    table, root_state = _gather_states(
        [id(c) for c in postorder],
        lambda key: [id(c) for c in by_key[key].clades],
        lambda key: labels.get(by_key[key].name),
        rank)

    if root_state is None:
        return tree

    pending = [(id(tree.root), root_state)]
    while pending:
        key, state = pending.pop()
        payload = table[key][state][1]
        if payload is None:
            continue
        order, left_state, right_state = payload
        by_key[key].clades = [by_key[c] for c in order]
        pending.append((order[0], left_state))
        if right_state is not None:
            pending.append((order[1], right_state))

    return tree


def _linkage_order_optimal(linkage, ids, labels, rank):
    """ Leaf order of a linkage with the fewest label blocks """
    linkage = np.asarray(linkage)
    n = len(ids)

    children = {i: [] for i in range(n)}
    for k, row in enumerate(linkage):
        children[n + k] = [int(row[0]), int(row[1])]

    table, root_state = _gather_states(
        range(2 * n - 1),
        lambda key: children[key],
        lambda key: labels.get(ids[key]),
        rank)

    if root_state is None:
        return list(ids)

    order = []
    stack = [(2 * n - 2, root_state)]
    while stack:
        key, state = stack.pop()
        payload = table[key][state][1]
        if payload is None:
            order.append(ids[key])
            continue
        kids, left_state, right_state = payload
        stack.append((kids[1], right_state))
        stack.append((kids[0], left_state))

    return order


def align_tree_to_groups(tree, groups, group_order=None, method='auto'):
    """ Rotate a tree so cells sharing a group sit together

    A tree arrives with an arbitrary child order, since newick does not
    canonicalise one, and that order decides where each cell lands. Annotate
    such a tree with a per cell label and the label speckles: cells of one
    group are scattered down the plot rather than forming blocks. Rotating the
    tree to follow the label gathers them.

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        tree whose leaf names are cell ids, not modified
    groups : pandas.Series or dict
        group label per cell id, for instance ``adata.obs['cluster_id']``
    group_order : list, optional
        order to place the groups in, by default the categories of a
        categorical ``groups``, otherwise its sorted unique values. The optimal
        method minimises blocks first and uses this only to break ties, so it
        decides which way round the groups go but will not split one to do it.
    method : str, optional
        'optimal' for the fewest blocks achievable, found exactly; 'greedy' to
        sort cells by group and rotate to follow, which is cheaper but can
        leave a label more broken up than it needs to be; 'auto' for optimal up
        to ``MAX_OPTIMAL_GROUPS`` groups and greedy beyond. By default 'auto'.

    Returns
    -------
    Bio.Phylo.BaseTree.Tree
        a rotated copy of the tree

    Notes
    -----
    Rotation never changes the topology or a branch length, so this gathers a
    label only as far as the tree's own structure allows. A clade that genuinely
    mixes two groups cannot be unmixed by reordering it, and the speckle that
    survives is telling you the tree and the label disagree. For the stricter
    question of whether an order is reproducible at all, see
    :func:`align_tree_to_order`.

    The optimal method costs roughly the square of the group count per clade,
    which is why it is not used for an unbounded number of groups. Clades with
    more than two children have their extra children ordered greedily, since
    the arrangements of a multifurcation cannot be enumerated.

    Examples
    --------

    >>> import pandas as pd, io, Bio.Phylo, scgenome
    >>> tree = Bio.Phylo.read(io.StringIO('((a:1,c:1):1,(b:1,d:1):1);'), 'newick')
    >>> groups = pd.Series(['x', 'y', 'x', 'y'], index=['a', 'b', 'c', 'd'])
    >>> gathered = scgenome.tl.align_tree_to_groups(tree, groups)
    >>> [groups[leaf] for leaf in scgenome.tl.tree_leaf_order(gathered)]
    ['x', 'x', 'y', 'y']

    """
    if method not in ('auto', 'optimal', 'greedy'):
        raise ValueError(
            f"unknown method {method!r}, expected 'auto', 'optimal' or 'greedy'")

    groups = pd.Series(groups)
    rank = _group_ranks(groups, group_order)

    if method == 'auto':
        method = 'optimal' if len(rank) <= MAX_OPTIMAL_GROUPS else 'greedy'

    if method == 'greedy':
        return _rotate_greedy(tree, groups, rank)

    return _rotate_optimal(tree, groups.to_dict(), rank)


def _stored_linkage(adata, key):
    """ The linkage sort_cells recorded, and the ids its rows refer to """
    records = adata.uns.get('cell_order', {})
    if key not in records:
        raise ValueError(
            f'no stored linkage {key!r}, run scgenome.tl.sort_cells first. '
            f'Available: {sorted(records.keys())}')

    record = records[key]

    return np.asarray(record['linkage']), list(np.asarray(record['ids']))


def linkage_to_tree(adata, key='cell_order'):
    """ Build a tree from the clustering behind an ordering

    :func:`sort_cells` records the linkage it clustered with, which is a binary
    tree in array form. This returns it as a ``Bio.Phylo`` tree, so the cells
    can be drawn and manipulated as a phylogeny without reclustering.

    Parameters
    ----------
    adata : AnnData
        data sorted by :func:`sort_cells`, not modified
    key : str, optional
        which stored ordering to build from, by default 'cell_order'

    Returns
    -------
    Bio.Phylo.BaseTree.Tree
        tree whose leaf names are cell ids and whose branch lengths are the
        merge heights of the clustering

    Reads
    -----
    adata.uns['cell_order'][key] : linkage and ids

    Examples
    --------

    >>> import scgenome
    >>> adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    >>> adata = scgenome.tl.sort_cells(adata, layer_name='copy')
    >>> tree = scgenome.tl.linkage_to_tree(adata)
    >>> tree.count_terminals() == adata.shape[0]
    True

    """
    import Bio.Phylo.BaseTree

    linkage, ids = _stored_linkage(adata, key)
    n = len(ids)

    clades = [Bio.Phylo.BaseTree.Clade(name=cell_id) for cell_id in ids]

    for row in linkage:
        left, right, height = int(row[0]), int(row[1]), float(row[2])
        clade = Bio.Phylo.BaseTree.Clade(clades=[clades[left], clades[right]])
        for child, child_height in (
                (clades[left], _clade_height(linkage, n, left)),
                (clades[right], _clade_height(linkage, n, right))):
            child.branch_length = height - child_height
        clades.append(clade)

    return Bio.Phylo.BaseTree.Tree(root=clades[-1])


def _clade_height(linkage, n, node):
    """ Merge height of a linkage node, zero for a leaf """
    return 0. if node < n else float(linkage[node - n][2])


def order_cells_by_groups(adata, groups, key='cell_order', group_order=None,
                          method='auto'):
    """ Order cells so a label is gathered, without breaking the dendrogram

    A clustering admits many leaf orders: swapping the two children of any
    merge leaves the clustering itself untouched. :func:`sort_cells` picks one
    of them arbitrarily, so a label drawn beside the dendrogram tends to
    speckle. This returns the member of that family which gathers ``groups``
    into as few blocks as the clustering allows.

    The result is always drawable against the dendrogram, since it only ever
    reorders whole merges.

    Parameters
    ----------
    adata : AnnData
        data sorted by :func:`sort_cells`, not modified
    groups : str, pandas.Series or dict
        obs column name, or a group label per cell id
    key : str, optional
        which stored ordering to reorder, by default 'cell_order'
    group_order : list, optional
        order to place the groups in, used to break ties
    method : str, optional
        'optimal', 'greedy' or 'auto', see :func:`align_tree_to_groups`

    Returns
    -------
    pandas.Index
        cell ids in plot order

    Reads
    -----
    adata.uns['cell_order'][key] : linkage and ids
    adata.obs[groups] : group labels, when given as a column name

    Examples
    --------

    >>> import scgenome
    >>> adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    >>> adata = scgenome.tl.sort_cells(adata, layer_name='copy')
    >>> order = scgenome.tl.order_cells_by_groups(adata, 'cluster_id')
    >>> len(order) == adata.shape[0]
    True

    """
    if method not in ('auto', 'optimal', 'greedy'):
        raise ValueError(
            f"unknown method {method!r}, expected 'auto', 'optimal' or 'greedy'")

    if isinstance(groups, str):
        if groups not in adata.obs.columns:
            raise ValueError(
                f'missing obs column {groups!r}. '
                f'Available obs columns: {list(adata.obs.columns)}')
        groups = adata.obs[groups]

    groups = pd.Series(groups)
    rank = _group_ranks(groups, group_order)

    linkage, ids = _stored_linkage(adata, key)

    missing = [i for i in ids if i not in groups.index]
    if missing:
        raise ValueError(
            f'{len(missing)} cells in the stored linkage have no group, for '
            f'instance {missing[:3]}')

    if method == 'auto':
        method = 'optimal' if len(rank) <= MAX_OPTIMAL_GROUPS else 'greedy'

    if method == 'greedy':
        tree = _rotate_greedy(linkage_to_tree(adata, key=key), groups, rank)
        return pd.Index(tree_leaf_order(tree))

    return pd.Index(
        _linkage_order_optimal(linkage, ids, groups.to_dict(), rank))


def resolve_cell_order(adata, fields=None):
    """ Resolve the order in which cells are drawn

    Parameters
    ----------
    adata : AnnData
        per cell data, not modified
    fields : list, optional
        obs columns to sort on, first is primary, by default None for the
        existing obs order

    Returns
    -------
    pandas.Index
        cell ids in plot order

    Reads
    -----
    adata.obs[fields] : sort keys

    Notes
    -----
    To order cells by a tree instead, use :func:`tree_leaf_order`. To make a
    tree agree with an order resolved here, rotate it first with
    :func:`align_tree_to_order`; plotting functions take a tree or an order,
    not both, so that reconciliation is always something you asked for.

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
        scgenome.pl.plot_heatmap(adata, layer_name='state', ax=axes[0],
                                 palette='cn', cell_order=order)
        scgenome.pl.plot_heatmap(adata, layer_name='copy', ax=axes[1],
                                 cell_order=order)

    """
    validate_adata(adata, caller='resolve_cell_order')

    fields = list(fields) if fields is not None else []

    missing = [f for f in fields if f not in adata.obs.columns]
    if missing:
        raise ValueError(
            f'missing obs columns {missing} for cell ordering. '
            f'Available obs columns: {list(adata.obs.columns)}')

    if not fields:
        return pd.Index(adata.obs.index)

    # lexsort applies the last key first, so reverse to make fields[0] primary
    keys = adata.obs[list(reversed(fields))].values.transpose()

    return pd.Index(adata.obs.index[np.lexsort(keys)])


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


def linkage_order_conflict(linkage, ids, order):
    """ Find a merge in a linkage whose leaves are split by a row order

    The dendrogram equivalent of the check in :func:`align_tree_to_order`. A
    linkage can be drawn against a row order without its brackets crossing if
    and only if every merge's leaves occupy a contiguous block of rows.

    Parameters
    ----------
    linkage : numpy.ndarray
        scipy linkage matrix over ``len(ids)`` observations
    ids : numpy.ndarray or list
        labels the linkage rows refer to, in linkage observation order
    order : pandas.Index or list
        cell ids in the requested row order

    Returns
    -------
    tuple or None
        ``(lo, hi, n_leaves)`` of the first merge whose leaves are split, or
        None if the whole linkage is drawable against this order

    Examples
    --------

    >>> import scgenome
    >>> adata = scgenome.datasets.OV2295_HMMCopy_reduced()
    >>> adata = scgenome.tl.sort_cells(adata, layer_name='copy')
    >>> record = adata.uns['cell_order']['cell_order']
    >>> order = adata.obs.sort_values('cell_order').index
    >>> scgenome.tl.linkage_order_conflict(record['linkage'], record['ids'], order) is None
    True

    """
    positions = {cell_id: i for i, cell_id in enumerate(order)}

    n = len(ids)
    spans = {}
    for i, label in enumerate(ids):
        position = positions.get(label)
        if position is not None:
            spans[i] = (position, position, 1)

    for k, row in enumerate(np.asarray(linkage)):
        left, right = int(row[0]), int(row[1])
        placed = [spans[c] for c in (left, right) if c in spans]

        if not placed:
            continue

        lo = min(span[0] for span in placed)
        hi = max(span[1] for span in placed)
        n_leaves = sum(span[2] for span in placed)

        if hi - lo + 1 != n_leaves:
            return lo, hi, n_leaves

        spans[n + k] = (lo, hi, n_leaves)

    return None
