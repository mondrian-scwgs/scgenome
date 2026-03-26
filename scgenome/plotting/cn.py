import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.path import Path
from pandas import DataFrame
from scgenome import refgenome
from scipy.sparse import issparse
from scgenome.plotting import cn_colors
from scgenome.tools.getters import get_obs_data


def genome_axis_plot(data, plot_function, position_columns, **kwargs):
    data = data.merge(refgenome.info.chromosome_info)
    for columns in position_columns:
        data[columns] = data[columns] + data['chromosome_start']

    plot_function(data=data, **kwargs)


def setup_genome_xaxis_lims(ax, chromosome=None, start=None, end=None):
    if chromosome is not None:
        chromosome_start = refgenome.info.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_start']
        chromosome_end = refgenome.info.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_end']

        if start is not None:
            plot_start = chromosome_start + start
        else:
            plot_start = chromosome_start

        if end is not None:
            plot_end = chromosome_start + end
        else:
            plot_end = chromosome_end

    else:
        plot_start = 0
        plot_end = refgenome.info.chromosome_info['chromosome_end'].max()

    ax.set_xlim((plot_start-0.5, plot_end+0.5))


def setup_genome_xaxis_ticks(ax, chromosome=None, start=None, end=None, major_spacing=2e7, minor_spacing=1e6, chromosome_names=None):
    if chromosome_names is None:
        chromosome_names = refgenome.info.chromosome_info.set_index('chr')['chr_plot']
    
    if chromosome is not None:
        if major_spacing is None:
            major_spacing = 2e7

        if minor_spacing is None:
            minor_spacing = 1e6

        chromosome_length = refgenome.info.chromosome_info.set_index('chr').loc[
            chromosome, 'chromosome_length']
        chromosome_start = refgenome.info.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_start']
        chromosome_end = refgenome.info.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_end']

        xticks = np.arange(0, chromosome_length, major_spacing)
        xticklabels = ['{0:d}M'.format(int(x / 1e6)) for x in xticks]
        xminorticks = np.arange(0, chromosome_length, minor_spacing)

        ax.set_xticks(xticks + chromosome_start)
        ax.set_xticklabels(xticklabels)

        ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(xminorticks + chromosome_start))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        if start is not None and end is not None:
            ax.set_xlim(chromosome_start+start, chromosome_start+end)
        elif start is not None:
            ax.set_xlim(left=chromosome_start+start)
        elif end is not None:
            ax.set_xlim(right=chromosome_start+end)

    else:
        ax.set_xticks([0] + refgenome.info.chromosome_info['chromosome_end'].values.tolist())
        ax.set_xticklabels([])

        ax.xaxis.set_minor_locator(
            matplotlib.ticker.FixedLocator(refgenome.info.chromosome_info['chromosome_mid'])
        )
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter([chromosome_names.get(c, c) for c in refgenome.info.chromosomes]))


def setup_squash_yaxis(ax):
    squash_coeff = 0.15
    squash_fwd = lambda a: np.tanh(squash_coeff * a)
    squash_rev = lambda a: np.arctanh(a) / squash_coeff
    ax.set_yscale('function', functions=(squash_fwd, squash_rev))

    yticks = np.array([0, 2, 4, 7, 20])
    ax.set_yticks(yticks)

    # Matplotlib will set problematic ylim if there are large y values
    ylim = ax.get_ylim()
    ax.set_ylim(-0.25, ylim[1])
 

def plot_profile(
        data: DataFrame,
        y,
        hue=None,
        ax=None,
        palette=None,
        chromosome=None,
        start=None,
        end=None,
        squashy=False,
        tick_major_spacing=None,
        tick_minor_spacing=None,
        **kwargs
):
    """Plot scatter points of copy number across the genome or a chromosome.

    Parameters
    ----------
    data : pandas.DataFrame
        copy number data
        observation to plot
    y : str
        field with values for y axis
    hue : str, optional
        field by which to color points, None for no color, by default None
    ax : matplotlib.axes.Axes, optional
        existing axess to plot into, by default None
    palette : str, optional
        color palette passed to sns.scatterplot
    chromosome : str, optional
        single chromosome plot, by default None
    start : int, optional
        start of plotting region
    end : int, optional
        end of plotting region
    squashy : bool, optional
        compress y axis, by default False
    rawy : bool, optional
        raw data on y axis, by default False
    tick_major_spacing : int, optional
        major tick spacing, by default 
    tick_minor_spacing : int, optional
        minor tick spacing, by default 1e6
    **kwargs :
        kwargs for sns.scatterplot

    Returns
    -------
    matplotlib.axes.Axes
        Axes used for plotting

    Examples
    -------

    .. plot::
        :context: close-figs

        import scgenome
        adata = scgenome.datasets.OV2295_HMMCopy_reduced()
        scgenome.pl.plot_profile(adata[:, adata.var['gc'] > 0], 'gc')

    """

    if ax is None:
        ax = plt.gca()

    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 0

    if 's' not in kwargs:
        kwargs['s'] = 5

    if palette is None and hue is not None:
        palette = cn_colors.color_reference

    if chromosome is not None:
        data = data[data['chr'] == chromosome]

    if start is not None:
        data = data[data['start'] >= start]

    if end is not None:
        data = data[data['end'] <= end]

    genome_axis_plot(
        data,
        sns.scatterplot,
        ('start',),
        x='start',
        y=y,
        hue=hue,
        palette=palette,
        ax=ax,
        clip_on=False,
        **kwargs)

    setup_genome_xaxis_ticks(
        ax,
        chromosome=chromosome,
        start=start,
        end=end,
        major_spacing=tick_major_spacing,
        minor_spacing=tick_minor_spacing,
    )

    setup_genome_xaxis_lims(
        ax,
        chromosome=chromosome,
        start=start,
        end=end,
    )

    if squashy:
        setup_squash_yaxis(ax)

    if chromosome is not None:
        ax.set_xlabel(f'Chromosome {chromosome}')

    else:
        ax.set_xlabel('Chromosome')

    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    return ax


def plot_cn_profile(
        adata: AnnData,
        obs_id: str,
        value_layer_name=None,
        state_layer_name=None,
        ax=None,
        palette=None,
        chromosome=None,
        start=None,
        end=None,
        squashy=False,
        tick_major_spacing=None,
        tick_minor_spacing=None,
        **kwargs
):
    """Plot scatter points of copy number across the genome or a chromosome.

    Parameters
    ----------
    adata : AnnData
        copy number data
    obs_id : str
        observation to plot
    value_layer_name : str, optional
        layer with values for y axis, None for X, by default None
    state_layer_name : str, optional
        layer with states for colors, None for no color by state, by default None
    ax : matplotlib.axes.Axes, optional
        existing axess to plot into, by default None
    palette : str, optional
        color palette passed to sns.scatterplot
    chromosome : str, optional
        single chromosome plot, by default None
    start : int, optional
        start of plotting region
    end : int, optional
        end of plotting region
    squashy : bool, optional
        compress y axis, by default False
    rawy : bool, optional
        raw data on y axis, by default False
    tick_major_spacing : int, optional
        major tick spacing, by default 
    tick_minor_spacing : int, optional
        minor tick spacing, by default 1e6
    **kwargs :
        kwargs for sns.scatterplot

    Returns
    -------
    matplotlib.axes.Axes
        Axes used for plotting

    Examples
    -------

    .. plot::
        :context: close-figs

        import scgenome
        adata = scgenome.datasets.OV2295_HMMCopy_reduced()
        scgenome.pl.plot_cn_profile(adata, 'SA922-A90554B-R27-C43', value_layer_name='copy', state_layer_name='state')

    """

    if value_layer_name is None:
        value_layer_name = '_X'

    layers = {value_layer_name}

    if state_layer_name is not None:
        layers.add(state_layer_name)

    cn_data = get_obs_data(
        adata,
        obs_id,
        ['chr', 'start'],
        layer_names=layers)

    return plot_profile(
        cn_data,
        y=value_layer_name,
        hue=state_layer_name,
        ax=ax,
        palette=palette,
        chromosome=chromosome,
        start=start,
        end=end,
        squashy=squashy,
        tick_major_spacing=tick_major_spacing,
        tick_minor_spacing=tick_minor_spacing,
        **kwargs)


def plot_rearrangement_arcs(
    ax,
    breakpoints,
    chromosome=None,
    start=None,
    end=None,
    height_diff_strand=0.10,
    height_same_strand=0.20,
    height_out_of_view=0.30,
    line_bottom=0.0,
    arc_height_min=0.1,
    arc_height_scale=1.0,
    diagonal_length=0.05,
    strand_colors=None,
    linewidth=0.5,
    connector_linewidth=0.5,
    show_rail_lines=True,
    show_rail_labels=True,
    rail_linewidth=0.5,
    rail_color='gray',
    rail_alpha=0.5,
    label_fontsize=7,
    alpha=1.0,
    zorder=10,
):
    """Plot rearrangement arcs linking breakpoint pairs.
    
    Draws vertical lines at breakpoint positions extending above the axes,
    connected by curved arcs. Rail height determined by strand combination:
    - +/- or -/+ (different strands): lowest rail
    - +/+ or -/- (same strands): middle rail
    - out of view partner: highest rail (diagonal line instead of arc)
    
    Arc direction determined by the strand at the leftmost position:
    - strand_left == '+' : arc curves upward (above rail)
    - strand_left == '-' : arc curves downward (below rail)
    
    When a breakend's partner is out of view (different chromosome or outside
    the start/end range), a diagonal line is drawn to the highest rail to
    indicate the connection goes off-screen.
    
    Colors are determined by strand combination (strand_left, strand_right)
    where left/right refers to genome position.
    
    Arc curvature scales with distance between breakpoints.
    
    All y-coordinates are in axes fraction (0=bottom, 1=top of axes).
    Values > 1 extend above the axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    breakpoints : pandas.DataFrame
        DataFrame with columns:
        - chromosome_1, position_1, strand_1
        - chromosome_2, position_2, strand_2
    chromosome : str, optional
        If specified, only plot breakpoints on this chromosome.
        If None, uses whole-genome coordinates with chromosome offsets.
    start : int, optional
        Start position of view region (used with chromosome parameter).
    end : int, optional
        End position of view region (used with chromosome parameter).
    height_diff_strand : float, optional
        Rail height for +/- and -/+ (different strands) in axes fraction
        (default 0.10, i.e., 10% above the top of the axes). Lowest rail.
    height_same_strand : float, optional
        Rail height for +/+ and -/- (same strands) in axes fraction
        (default 0.20, i.e., 20% above the top of the axes). Middle rail.
    height_out_of_view : float, optional
        Rail height for out-of-view partners in axes fraction
        (default 0.30, i.e., 30% above the top of the axes). Highest rail.
    line_bottom : float, optional
        Bottom of vertical lines in axes fraction (default 0).
    arc_height_min : float, optional
        Minimum arc height as fraction of rail spacing
        (default 0.1, i.e., 10% of the space between rails).
    arc_height_scale : float, optional
        Scale factor for arc height (default 1.0).
    diagonal_length : float, optional
        Length of diagonal lines for out-of-view partners in axes fraction
        (default 0.05).
    strand_colors : dict, optional
        Dict mapping strand combinations to colors. Keys should be
        tuples like ('+', '-') where first element is the strand at
        the leftmost position and second is at the rightmost.
        Default colors: +/+ green, +/- blue, -/+ orange, -/- purple.
    linewidth : float, optional
        Width of vertical lines (default 0.5).
    connector_linewidth : float, optional
        Width of arc curves (default 0.5).
    show_rail_lines : bool, optional
        Whether to draw horizontal reference lines at the
        rail heights (default True).
    show_rail_labels : bool, optional
        Whether to draw strand combination labels on rails
        (default True).
    rail_linewidth : float, optional
        Width of rail lines (default 0.5).
    rail_color : str, optional
        Color of rail lines (default 'gray').
    rail_alpha : float, optional
        Transparency of rail lines (default 0.5).
    label_fontsize : float, optional
        Font size for rail labels (default 7).
    alpha : float, optional
        Transparency for breakpoint lines (default 1.0).
    zorder : int, optional
        Drawing order (default 10).
    
    Returns
    -------
    list
        List of artists added to the axes.
    
    Examples
    --------
    >>> brks = pd.DataFrame({
    ...     'chromosome_1': ['chr1', 'chr1'],
    ...     'position_1': [1e6, 5e6],
    ...     'strand_1': ['+', '-'],
    ...     'chromosome_2': ['chr1', 'chr1'],
    ...     'position_2': [2e6, 8e6],
    ...     'strand_2': ['-', '-'],
    ... })
    >>> fig, ax = plt.subplots()
    >>> ax.set_xlim(0, 10e6)
    >>> ax.set_ylim(0, 8)
    >>> plot_rearrangement_arcs(ax, brks, chromosome='chr1')
    """
    artists = []
    breakpoints = breakpoints.copy()
    
    # Normalize chromosome names
    for col in ['chromosome_1', 'chromosome_2']:
        if col in breakpoints.columns:
            breakpoints[col] = breakpoints[col].astype(str)
    
    # Blended transform: x in data coords, y in axes coords
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    
    # Default colors by strand combination (strand_left, strand_right)
    # '?' indicates out-of-view partner
    default_strand_colors = {
        ('+', '+'): '#31a354',  # green - inversion (same strand)
        ('+', '-'): '#3182bd',  # blue - deletion-like
        ('-', '+'): '#e6550d',  # orange - duplication-like
        ('-', '-'): '#756bb1',  # purple - inversion (same strand)
        ('+', '?'): '#74c476',  # light green - out of view, + strand visible
        ('-', '?'): '#9e9ac8',  # light purple - out of view, - strand visible
    }
    if strand_colors is not None:
        default_strand_colors.update(strand_colors)
    
    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]
    
    # Determine view region
    if start is not None:
        view_start = start
    else:
        view_start = xlim[0]
    if end is not None:
        view_end = end
    else:
        view_end = xlim[1]
    
    # Rail spacing (distance between the two rail lines)
    rail_spacing = abs(height_same_strand - height_diff_strand)
    
    # Diagonal x-offset in data coordinates
    diag_x_offset = diagonal_length * x_range
    
    for idx, row in breakpoints.iterrows():
        chrom1 = str(row['chromosome_1'])
        chrom2 = str(row['chromosome_2'])
        pos1 = row['position_1']
        pos2 = row['position_2']
        strand1 = row.get('strand_1', '+')
        strand2 = row.get('strand_2', '+')
        
        # Track if breakends are on the target chromosome
        on_chrom_1 = True
        on_chrom_2 = True
        
        # Filter by chromosome and determine positions
        if chromosome is not None:
            chrom_match = chromosome.replace('chr', '')
            c1_match = chrom1.replace('chr', '')
            c2_match = chrom2.replace('chr', '')
            
            on_chrom_1 = (c1_match == chrom_match)
            on_chrom_2 = (c2_match == chrom_match)
            
            if not on_chrom_1 and not on_chrom_2:
                continue
        else:
            try:
                chromosome_info = refgenome.info.chromosome_info.set_index('chr')
                if chrom1 in chromosome_info.index:
                    pos1 = pos1 + chromosome_info.loc[chrom1, 'chromosome_start']
                if chrom2 in chromosome_info.index:
                    pos2 = pos2 + chromosome_info.loc[chrom2, 'chromosome_start']
            except:
                pass
        
        # Check if positions are within view region
        in_view_1 = on_chrom_1 and (view_start <= pos1 <= view_end)
        in_view_2 = on_chrom_2 and (view_start <= pos2 <= view_end)
        
        if not in_view_1 and not in_view_2:
            continue
        
        # Determine strand_left and strand_right based on genome position
        if pos1 <= pos2:
            strand_left, strand_right = strand1, strand2
            pos_left, pos_right = pos1, pos2
            in_view_left, in_view_right = in_view_1, in_view_2
        else:
            strand_left, strand_right = strand2, strand1
            pos_left, pos_right = pos2, pos1
            in_view_left, in_view_right = in_view_2, in_view_1
        
        # Determine color based on strand combination
        strand_combo = (strand_left, strand_right)
        color = default_strand_colors.get(strand_combo, '#000000')
        
        # Determine rail height based on strand combination
        same_strand = (strand_left == strand_right)
        rail_height = 1.0 + (height_same_strand if same_strand else height_diff_strand)
        
        # Determine arc direction: left strand + curves up, - curves down
        arc_direction = 1 if strand_left == '+' else -1
        
        # Both breakends in view - draw arc
        if in_view_left and in_view_right:
            # Draw vertical lines
            line1 = mlines.Line2D(
                [pos_left, pos_left], [line_bottom, rail_height],
                color=color, linewidth=linewidth, alpha=alpha,
                zorder=zorder, clip_on=False, transform=trans
            )
            ax.add_line(line1)
            artists.append(line1)
            
            line2 = mlines.Line2D(
                [pos_right, pos_right], [line_bottom, rail_height],
                color=color, linewidth=linewidth, alpha=alpha,
                zorder=zorder, clip_on=False, transform=trans
            )
            ax.add_line(line2)
            artists.append(line2)
            
            # Calculate arc height based on distance between breakpoints
            x_dist_frac = abs(pos_right - pos_left) / x_range
            arc_height = max(arc_height_min, x_dist_frac) * rail_spacing * arc_height_scale
            arc_peak = rail_height + arc_direction * arc_height
            
            # Draw curved arc
            xmid = (pos_left + pos_right) / 2
            control_y = 2 * arc_peak - rail_height
            
            verts = [
                (pos_left, rail_height),
                (xmid, control_y),
                (pos_right, rail_height),
            ]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            
            path = Path(verts, codes)
            patch = mpatches.PathPatch(
                path, facecolor='none', edgecolor=color,
                linewidth=connector_linewidth, alpha=alpha,
                zorder=zorder, clip_on=False, transform=trans
            )
            ax.add_patch(patch)
            artists.append(patch)
        
        # Only left breakend in view - partner is out of view
        elif in_view_left and not in_view_right:
            oov_rail_height = 1.0 + height_out_of_view
            
            # Color based on visible breakend's strand
            oov_color = default_strand_colors.get((strand_left, '?'), '#636363')
            
            # Direction based on visible breakend's strand: + goes up, - goes down
            visible_strand_direction = 1 if strand_left == '+' else -1
            
            # Draw vertical line up to out-of-view rail
            line = mlines.Line2D(
                [pos_left, pos_left], [line_bottom, oov_rail_height],
                color=oov_color, linewidth=linewidth, alpha=alpha,
                zorder=zorder, clip_on=False, transform=trans
            )
            ax.add_line(line)
            artists.append(line)
            
            # Draw diagonal line: left-to-right, up/down based on visible strand
            diag_y = diagonal_length * visible_strand_direction
            diag = mlines.Line2D(
                [pos_left, pos_left + diag_x_offset],
                [oov_rail_height, oov_rail_height + diag_y],
                color=oov_color, linewidth=connector_linewidth, alpha=alpha,
                zorder=zorder, clip_on=False, transform=trans
            )
            ax.add_line(diag)
            artists.append(diag)
        
        # Only right breakend in view - partner is out of view
        elif not in_view_left and in_view_right:
            oov_rail_height = 1.0 + height_out_of_view
            
            # Color based on visible breakend's strand
            oov_color = default_strand_colors.get((strand_right, '?'), '#636363')
            
            # Direction based on visible breakend's strand: + goes up, - goes down
            visible_strand_direction = 1 if strand_right == '+' else -1
            
            # Draw vertical line up to out-of-view rail
            line = mlines.Line2D(
                [pos_right, pos_right], [line_bottom, oov_rail_height],
                color=oov_color, linewidth=linewidth, alpha=alpha,
                zorder=zorder, clip_on=False, transform=trans
            )
            ax.add_line(line)
            artists.append(line)
            
            # Draw diagonal line: left-to-right, up/down based on visible strand
            diag_y = diagonal_length * visible_strand_direction
            diag = mlines.Line2D(
                [pos_right, pos_right + diag_x_offset],
                [oov_rail_height, oov_rail_height + diag_y],
                color=oov_color, linewidth=connector_linewidth, alpha=alpha,
                zorder=zorder, clip_on=False, transform=trans
            )
            ax.add_line(diag)
            artists.append(diag)
    
    # Draw rail lines at all three heights (always)
    if show_rail_lines:
        rail_heights = [
            1.0 + height_diff_strand,
            1.0 + height_same_strand,
            1.0 + height_out_of_view,
        ]
        for h in rail_heights:
            rail = mlines.Line2D(
                [xlim[0], xlim[1]], [h, h],
                color=rail_color, linewidth=rail_linewidth, alpha=rail_alpha,
                zorder=zorder - 1, clip_on=False, transform=trans,
                linestyle='--'
            )
            ax.add_line(rail)
            artists.append(rail)
    
    # Draw rail labels with strand combination text
    if show_rail_labels:
        label_x = xlim[1]
        label_offset = 0.015
        
        # Lower rail (different strands): +/- and -/+
        h_diff = 1.0 + height_diff_strand
        txt_plus_minus = ax.text(
            label_x, h_diff + label_offset, '+/−',
            color=default_strand_colors[('+', '-')],
            fontsize=label_fontsize, ha='right', va='bottom',
            transform=trans, clip_on=False, zorder=zorder
        )
        artists.append(txt_plus_minus)
        txt_minus_plus = ax.text(
            label_x, h_diff - label_offset, '−/+',
            color=default_strand_colors[('-', '+')],
            fontsize=label_fontsize, ha='right', va='top',
            transform=trans, clip_on=False, zorder=zorder
        )
        artists.append(txt_minus_plus)
        
        # Middle rail (same strands): +/+ and -/-
        h_same = 1.0 + height_same_strand
        txt_plus_plus = ax.text(
            label_x, h_same + label_offset, '+/+',
            color=default_strand_colors[('+', '+')],
            fontsize=label_fontsize, ha='right', va='bottom',
            transform=trans, clip_on=False, zorder=zorder
        )
        artists.append(txt_plus_plus)
        txt_minus_minus = ax.text(
            label_x, h_same - label_offset, '−/−',
            color=default_strand_colors[('-', '-')],
            fontsize=label_fontsize, ha='right', va='top',
            transform=trans, clip_on=False, zorder=zorder
        )
        artists.append(txt_minus_minus)
        
        # Upper rail (out of view): +/? and -/?
        h_oov = 1.0 + height_out_of_view
        txt_plus_oov = ax.text(
            label_x, h_oov + label_offset, '+/?',
            color=default_strand_colors[('+', '?')],
            fontsize=label_fontsize, ha='right', va='bottom',
            transform=trans, clip_on=False, zorder=zorder
        )
        artists.append(txt_plus_oov)
        txt_minus_oov = ax.text(
            label_x, h_oov - label_offset, '−/?',
            color=default_strand_colors[('-', '?')],
            fontsize=label_fontsize, ha='right', va='top',
            transform=trans, clip_on=False, zorder=zorder
        )
        artists.append(txt_minus_oov)
    
    return artists
