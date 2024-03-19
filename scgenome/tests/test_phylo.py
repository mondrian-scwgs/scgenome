import Bio.Phylo
import io

import scgenome.tl


def test_prune_leaves():
    # Create a small biopython tree
    tree = Bio.Phylo.read(io.StringIO('(((A:1,B:1):1,C:2):1,(D:1,E:1):3);'), 'newick')

    # Prune the leaves
    tree = scgenome.tl.prune_leaves(tree, lambda a: a.name == 'D' or a.name == 'E')

    # Verify the tree structure
    assert tree.count_terminals() == 3
    assert len(tree.clade.clades) == 1
    assert tree.clade.clades[0].clades[0].clades[0].name == 'A'
    assert tree.clade.clades[0].clades[0].clades[1].name == 'B'
    assert tree.clade.clades[0].clades[1].name == 'C'


def test_aggregate_branches():
    # Create a small biopython tree with a chain of several branches
    tree = Bio.Phylo.read(io.StringIO('(((A:1,B:1):1,C:2):1,((D:1):3):2);'), 'newick')

    print(tree)
    tree = scgenome.tl.aggregate_tree_branches(tree)

    # Verify the tree structure
    assert tree.count_terminals() == 4
    assert tree.clade.clades[0].clades[0].clades[0].name == 'A'
    assert tree.clade.clades[0].clades[0].clades[1].name == 'B'
    assert tree.clade.clades[0].clades[1].name == 'C'
    assert tree.clade.clades[1].branch_length == 6.0


def test_aggregate_branches_f_1():
    # Create a small biopython tree with a chain of several branches
    tree = Bio.Phylo.read(io.StringIO('(((A:1,B:1):1,C:2):1,((D:1):3):2);'), 'newick')

    # Add boolean wgd attribute to the tree
    tree.clade.clades[1].wgd = False
    tree.clade.clades[1].clades[0].wgd = True
    tree.clade.clades[1].clades[0].clades[0].wgd = False

    def f_merge(parent, child):
        return {
            'name': child.name,
            'branch_length': parent.branch_length + child.branch_length,
            'wgd': parent.wgd or child.wgd,
        }
    
    tree = scgenome.tl.aggregate_tree_branches(tree, f_merge)

    # Verify the tree structure
    assert tree.count_terminals() == 4
    assert tree.clade.clades[0].clades[0].clades[0].name == 'A'
    assert tree.clade.clades[0].clades[0].clades[1].name == 'B'
    assert tree.clade.clades[0].clades[1].name == 'C'
    assert tree.clade.clades[1].branch_length == 6.0
    assert tree.clade.clades[1].name == 'D'
    assert tree.clade.clades[1].wgd == True


def test_aggregate_branches_f_2():
    # Create a small biopython tree with a chain of several branches
    tree = Bio.Phylo.read(io.StringIO('Z(Y(X(D:1):3):2):4;'), 'newick')

    # Add boolean wgd attribute to the tree
    tree.clade.wgd = False
    tree.clade.clades[0].wgd = False
    tree.clade.clades[0].clades[0].wgd = True
    tree.clade.clades[0].clades[0].clades[0].wgd = False

    def f_merge(parent, child):
        return {
            'name': child.name,
            'branch_length': parent.branch_length + child.branch_length,
            'wgd': parent.wgd or child.wgd,
        }

    tree = scgenome.tl.aggregate_tree_branches(tree, f_merge)

    # Verify the tree structure
    assert tree.count_terminals() == 1
    assert tree.clade.branch_length == 10.0
    assert tree.clade.name == 'D'
    assert tree.clade.wgd == True
