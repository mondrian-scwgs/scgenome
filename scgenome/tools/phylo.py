import Bio.Phylo
import copy


def prune_leaves(tree, f):
    """ Prune leaves from a tree based on a predicate

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        Tree to prune
    f : function
        Predicate function that takes a clade and returns a boolean
    """

    # Generate new clade object
    clade = prune_leaves_recursive(tree.clade, f)

    # Create a new Tree with the pruned clade as its root
    pruned_tree = Bio.Phylo.BaseTree.Tree(root=clade)

    return pruned_tree


def prune_leaves_recursive(clade, f):
    """ Prune leaves from a tree based on a predicate
    """
    # Create a copy of the clade
    clade = copy.deepcopy(clade)

    # If the clade is a leaf, check the predicate
    if clade.is_terminal():
        if f(clade):
            return None
        else:
            return clade

    # If the clade is not a leaf, recurse on its children
    else:
        new_clades = []
        for a in clade.clades:
            new_a = prune_leaves_recursive(a, f)
            if new_a is not None:
                new_clades.append(new_a)

        if len(new_clades) == 0:
            return None
        else:
            clade.clades = new_clades
            return clade


def aggregate_tree_branches(tree, f_merge=None):
    """ Merge redundant branches of a tree

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        Tree to merge
    f_merge : function
        Function to specify how to merge branches. If None, the default of summming branch lengths is used

    Returns
    -------
    Bio.Phylo.BaseTree.Tree
        Tree with merged branches
    """
    if f_merge is None:
        f_merge = lambda parent, child: {'branch_length': parent.branch_length + child.branch_length}

    # Generate new clade object
    clade = aggregate_tree_branches_recursive(tree.clade, f_merge)

    # Create a new Tree with the pruned clade as its root
    agg_tree = Bio.Phylo.BaseTree.Tree(root=clade)

    return agg_tree


def aggregate_tree_branches_recursive(clade, f_merge):
    """ Merge redundant branches of a tree

    Merge chains of branches into a single branch, aggregating properties of the branches.

    Parameters
    ----------
    clade : Bio.Phylo.BaseTree.Clade
        Clade to merge
    f_merge : function
        Function to specify how to merge branches

    Returns
    -------
    Bio.Phylo.BaseTree.Clade
        Clade with merged branches
    """

    # If the clade is not a leaf and has 1 child, merge with child
    if len(clade.clades) == 1:
        # Recurse on child
        new_child_clade = aggregate_tree_branches_recursive(clade.clades[0], f_merge)

        # Merge clade and child properties
        properties = f_merge(clade, new_child_clade)

        # Create new clade with given properties
        new_merged_clade = Bio.Phylo.BaseTree.Clade(
            clades=new_child_clade.clades,
            branch_length=properties.get('branch_length'),
            name=properties.get('name'),
            confidence=properties.get('confidence'),
            color=properties.get('color'),
            width=properties.get('width'))

        for key, value in properties.items():
            if key not in ['branch_length', 'name', 'confidence', 'color', 'width']:
                setattr(new_merged_clade, key, value)

        return new_merged_clade

    else:
        # If the clade is not a leaf and has more than 1 child, recurse on each child
        new_clades = []
        for a in clade.clades:
            new_a = aggregate_tree_branches_recursive(a, f_merge)
            new_clades.append(new_a)

        # Create a new clade with the new children
        clade = copy.deepcopy(clade)
        clade.clades = new_clades

        return clade


def align_cn_tree(tree, adata):
    """ Align a tree's leaves with an AnnData object's observations

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        Tree to align
    adata : anndata.AnnData
        AnnData object to align tree with

    Returns
    -------
    Bio.Phylo.BaseTree.Tree
        Aligned tree
    anndata.AnnData
        Aligned AnnData object
    """
    tree = prune_leaves(tree, lambda a: a.name not in adata.obs.index)

    cell_ids = []
    for a in tree.get_terminals():
        cell_ids.append(a.name)

    adata = adata[cell_ids]

    return tree, adata
