#!/usr/bin/env python
"""
converts spherical power spectrum to cartesian and back
"""

import sys,os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import sph_jn

#---- custom imports ----
sys.path.append(os.path.join(os.path.dirname(__file__),'../CommonModules'))
from CovarianceFunctions import power_gaussian3d
from CVMatrix import LayeredCovarianceMatrix
from FigStyle import style_gji
mpl.rcParams.update(style_gji)

PI = np.pi

def main():
    nr,nl = 200,180
    ls = np.arange(nl)
    radii = np.linspace(0.,1,nr)
    rhos  = np.arange(nr/2)
    power = power_gaussian3d(2*PI*rhos,20)
    CVMatrix = LayeredCovarianceMatrix(ls,radii)
    CVMatrix.fill_isospherical(rhos,power)
    CVMatrix.plot_cvmatrix(5,loglog=True,totpower=True)

    hpower = CVMatrix.get_hpower()
    fig,ax = plt.subplots(1,1)
    ax.set_title('hpower at different radii')
    for ir in [20,40,80,160]:
        ax.plot(ls,hpower[:,ir]*(2*ls+1),label='r=%2.2f'%radii[ir])
        ax.plot()
    ax.set_xscale('log')
    ax.legend()

    ls2,hpower2 = CVMatrix.get_hpower_from_3dpower(rhos,power,radii[40])
    rhos2,power2 = CVMatrix.get_3dpower_from_hpower(hpower[:,80],radii[80])
    fig,ax = plt.subplots(1,2)
    ax[0].plot(rhos,power,label='original Fourier power')
    ax[0].plot(rhos2,power2,label='recovered Fourier power')
    ax[1].plot(ls,hpower[:,40],label='hpower from cvmatrix')
    ax[1].plot(ls2,hpower2,label='direct hpower')
    for axis in ax:
        axis.legend()

    plt.show()

if __name__ == "__main__":
    main()
