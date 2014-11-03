#!/usr/bin/env python
"""
This script provides the function get_partitionfuncs that subdivides
an interval in subintervals with smooth variable costaper edges.
"""

import numpy as np
import matplotlib.pyplot as plt

def make_partition(dlocs,dtrans,radii):
    nr = len(radii)
    nregions = len(dlocs)-1
    tfuncs   = np.zeros( (nregions,nr) )

    for iregion in range(nregions):
        c1  = float(dlocs[iregion]-dtrans[iregion])
        c2  = float(dlocs[iregion]+dtrans[iregion])
        c3  = float(dlocs[iregion+1]-dtrans[iregion+1])
        c4  = float(dlocs[iregion+1]+dtrans[iregion+1])

        conditions = [radii<c1,
                     (c1<=radii)&(radii<c2),
                     (c2<=radii)&(radii<c3),
                     (c3<=radii)&(radii<c4),
                     radii>=c4]

        functions  = [lambda r: 0.,
                      lambda r: np.sin(np.pi*(r-c1)/2./(c2-c1))**2,
                      lambda r: 1.,
                      lambda r:  np.cos(np.pi*(r-c3)/2./(c4-c3))**2,
                      lambda r: 0.]
        tfuncs[iregion] = np.piecewise(radii.astype(np.float),conditions,functions)

    return tfuncs

#==== MAIN FUNCTION ====
def main():
    npoints = 100
    locations = np.linspace(0.,1.,npoints)
    dlocs  = [-1,0.20,0.8,1.2]
    dtrans = [0.0,0.05,0.2,0.0]
    tfuncs = make_partition(dlocs,dtrans,locations)

    fig,ax = plt.subplots(1,1)
    ax.set_title('transfer functions')
    for tfunc in tfuncs:
        ax.plot(locations,tfunc)
    plt.show()

#==== EXECUTE SCRIPTS ====
if __name__ == "__main__":
    main()
