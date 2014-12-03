"""
contains configuration parameters for matplotlib
"""

textwidth_pt = 504.0
columnwidth_pt = 240.0
pt2inch = 1.0/72.27
lat_textwidth_in = textwidth_pt*pt2inch
lat_columnwidth_in = columnwidth_pt*pt2inch

style_gji = {'font.size'       : 7,
             'axes.labelsize'  : 'medium',
             'axes.titlesize'  : 'medium',
             'xtick.labelsize' : 'medium',
             'ytick.labelsize' : 'medium',
             'legend.fontsize' : 'medium',
             'figure.dpi'      : 160,
             'savefig.dpi'     : 160,
             'font.family'     : 'serif',
             'font.serif'      : ['Computer Modern Roman'],
             'text.usetex'     : True,
             'figure.figsize'  : (lat_columnwidth_in, lat_columnwidth_in/2)}

def label_axes(fig, axes, labels, xoff=12,yoff=12):
    if not len(axes):
        return
    for ax,label in zip(axes,labels):
        xy = -xoff,-yoff
        ax.annotate(label,xy=xy,xytext=xy,xycoords='axes points', va='top',ha='right',
                              fontsize='large',textcoords='axes points')
