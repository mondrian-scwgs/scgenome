import numpy as np
import pandas as pd
import pkg_resources
import sys

this = sys.modules[__name__]

this.version = None
this.chromosomes = None
this.plot_chromosomes = None
this.cytobands = None
this.chromosome_info = None


def initialize(version='hg19', chromosomes=None):
    # initialize is being called with same params as current state. dont do anything
    if this.version == version and (chromosomes is None or chromosomes == this.chromosomes):
        return

    this.version = version
    this.chromosomes = get_chromosomes(this.version) if chromosomes is None else chromosomes
    this.plot_chromosomes = get_plot_chromosomes(this.chromosomes)
    this.cytobands = get_cytobands(this.version)
    this.chromosome_info = get_chromosome_info(this.version, this.chromosomes, this.plot_chromosomes)


def read_chromosome_lengths(genome_fasta_index):
    chromosome_lengths = pd.read_csv(
        genome_fasta_index,
        sep='\t',
        header=None,
        names=['chr', 'chromosome_length', 'V3', 'V4', 'V5']
    )
    chromosome_lengths = chromosome_lengths[['chr', 'chromosome_length']]
    return chromosome_lengths


def read_cytobands(cyto_filename, remove_chr_prefix=False):
    cytobands = pd.read_csv(
        cyto_filename,
        sep='\t',
        names=['chr', 'start', 'end', 'cyto_band_name', 'cyto_band_giemsa_stain']
    )
    if remove_chr_prefix:
        cytobands['chr'] = cytobands['chr'].str.replace('^chr', '', regex=True)
    return cytobands


def get_chromosomes(version):
    if version == 'hg19':
        chromosomes = [str(a) for a in range(1, 23)] + ['X', 'Y']
    elif version == 'grch38':
        chromosomes = [f'chr{a}' for a in range(1, 23)] + ['chrX', 'chrY']
    elif version == 'mm10':
        chromosomes = [str(a) for a in range(1, 20)] + ['X', 'Y']
    else:
        raise ValueError()
    return chromosomes


def get_plot_chromosomes(chroms):
    return [v.strip('chr') for v in chroms]


def get_cytobands(version):
    if version == 'hg19':
        cyto_filename = pkg_resources.resource_filename('scgenome', 'data/hg19_cytoBand.txt.gz')
        cytobands = read_cytobands(cyto_filename, remove_chr_prefix=True)
    elif version == 'grch38':
        cyto_filename = pkg_resources.resource_filename('scgenome', 'data/grch38_cytoBand.txt.gz')
        cytobands = read_cytobands(cyto_filename, remove_chr_prefix=False)
    elif version == 'mm10':
        cyto_filename = pkg_resources.resource_filename('scgenome', 'data/mm10_cytoBand.txt.gz')
        cytobands = read_cytobands(cyto_filename, remove_chr_prefix=True)
    else:
        raise ValueError()
    return cytobands


def get_chromosome_info(version, chromosomes, plot_chromosomes):
    if version == 'hg19':
        genome_fasta_index = pkg_resources.resource_filename('scgenome', 'data/hg19.fa.fai')
    elif version == 'grch38':
        genome_fasta_index = pkg_resources.resource_filename('scgenome', 'data/grch38.fa.fai')
    elif version == 'mm10':
        genome_fasta_index = pkg_resources.resource_filename('scgenome', 'data/mm10.fa.fai')
    else:
        raise ValueError()

    chromosome_info = read_chromosome_lengths(genome_fasta_index)

    # Subset and order according to list of chromosomes
    chromosome_info = chromosome_info.set_index('chr').loc[chromosomes].reset_index()
    chromosome_info['chr_index'] = range(chromosome_info.shape[0])

    # Add plotting names of chromosomes
    chromosome_info['chr_plot'] = plot_chromosomes

    # Add start end and mid
    chromosome_info['chromosome_end'] = np.cumsum(chromosome_info['chromosome_length'])
    chromosome_info['chromosome_start'] = chromosome_info['chromosome_end'].shift(1)
    chromosome_info.loc[chromosome_info.index[0], 'chromosome_start'] = 0
    chromosome_info['chromosome_start'] = chromosome_info['chromosome_start'].astype(int)
    chromosome_info['chromosome_mid'] = (chromosome_info['chromosome_start'] + chromosome_info['chromosome_end']) // 2

    return chromosome_info
