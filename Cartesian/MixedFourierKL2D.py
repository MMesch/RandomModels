#!/usr/bin/env python
"""
This script creates a nonstationary Gaussian 3D random model using a mixed
Fourier/Karhunen-Loeve transform.
"""

#---- imports ----
import os,sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from evtk.hl import gridToVTK

#---- custom imports from this package ----
sys.path.append(os.path.join(os.path.dirname(__file__),'../../CommonModules'))
from CovarianceFunctions import power_exponential2d
from PartitionFunctions import make_partition

#---- matplotlib style----
from FigStyle import style_gji
mpl.rcParams.update(style_gji)

PI = np.pi

def main():
    #---- initial parameters ----
    nx,ny,nz = 100,100,100
    sizex,sizey,sizez = 1.,1.,1.

    #---- derived parameters ----
    ntot2d = nx*ny
    nkz = nz/2+1
    print('total number of non stationary points: {}'.format(ntot2d))
    print('memory size of covariance matrix: {:1.3f}Mb'.format(ntot2d**2*8/1024./1024.))

    #---- setup frequency space ----
    kzs = 2*PI/sizex*np.arange(-nkz/2,nkz/2)

    #---- create grid (reordered to allow fast xy plane access) ----
    xcoords, ycoords,zcoords  = np.meshgrid(np.linspace(-sizex/2.,sizex/2.-1/nx,nx),
                                            np.linspace(-sizey/2.,sizey/2.-1/ny,ny),
                                            np.linspace(-sizez/2.,sizez/2.-1/nz,nz),
                                            indexing='ij')

    #---- fill horizontal coefficients with the right covariance one by one ----
    coeffs = np.zeros( (nx,ny,nkz) )

    #---- Fourier transform every xy layer ---
    model = np.empty( (nx,ny,nz) )
    for ix in range(nx):
        for iy in range(ny):
            model[ix,iy] = np.fft.irfft(np.fft.fftshift(coeffs))
    model = np.fft.ifftshift(model,axes=2)

    #---- output VTK model ----
    gridToVTK('../vtk/mixedKL2D',zcoords,xcoords,ycoords,pointData= {'random field': model})

    #---- plot random model section ----
    fig,axes = plt.subplots(1,2)
    axes[0].imshow(model[0,:,:],aspect='equal')
    axes[0].set_title('xy-plane')
    axes[1].imshow(model[:,0,:],aspect='equal')
    axes[1].set_title('zy-plane')
    plt.title('model section')

    plt.show()

#==== EXECUTE SCRIPT ====
if __name__ == "__main__":
    main()
