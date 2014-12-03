#!/usr/bin/env python
"""
This script creates a stationary Gaussian random model using the """

#---- imports ----
import os,sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from time import time

#---- custom imports ----
sys.path.append(os.path.join(os.path.dirname(__file__),'../CommonModules'))
from CovarianceFunctions import covar_exponential
from FigStyle import style_gji
mpl.rcParams.update(style_gji)

def main():
    #---- initial parameters ----
    scale    = 0.01
    size     = 1.
    nx,ny,nz = 9,9,9

    #---- derived parameters ----
    ntot = nx*ny*nz
    print('total number of points: {}'.format(ntot))
    print('memory size of covariance matrix: {}Mb'.format(ntot**2*8/1024./1024.))

    #---- create grid ----
    xcoords, ycoords, zcoords = np.meshgrid(np.linspace(-size/2.,size/2.-1/nx,nx),
                                            np.linspace(-size/2.,size/2.-1/ny,ny),
                                            np.linspace(-size/2.,size/2.-1/nz,nz),
                                            indexing='ij')

    #--- create covariance matrix ---
    print('now filling covariance matrix (may take a while) ...')
    cvmatrix = np.zeros( (nx*ny*nz,nx*ny*nz) )
    for ix1 in range(nx):
        for iy1 in range(ny):
            for iz1 in range(nz):
                i1 = ix1*ny*nz+iy1*nz+iz1
                if i1%200==0: 
                    tstart = time()
                    print('point {:d}/{:d}'.format(i1,ntot)),
                for ix2 in range(nx):
                    for iy2 in range(ny):
                        for iz2 in range(nz):
                            i2 = ix2*ny*nz+iy2*nz+iz2
                            dist = np.sqrt((ix1-ix2)**2+(iy1-iy2)**2+(iz1-iz2)**2)
                            cvmatrix[i1,i2] = covar_exponential(dist,scale/size*nx)
                if i1%200==0: 
                    tend = time()
                    print('{:2.1f}s to go'.format((tend-tstart)*(ntot-i1)))

    #---- compute eigenvectors and values ----
    print 'computing eigenvectors and values with cholesky method'
    evals,evecs = eigh(cvmatrix)

    #---- plot covariance matrix ----
    plt.figure()
    plt.imshow(cvmatrix)
    plt.title('Covariance Matrix')

    #---- plot eigenvectors ----
    plt.figure()
    plt.imshow(evecs)
    plt.title('Eigenvectors')

    #---- compute random model ----
    coeffs = np.random.normal(loc=0.,scale=1.,size=ntot)
    values = np.dot(np.sqrt(evals).reshape(1,ntot)*evecs,coeffs)
    values = values.reshape(nx,ny,nz)

    #---- plot random model section ----
    plt.figure()
    plt.imshow(values[0,:,:])
    plt.title('model section')

    plt.show()


#==== EXECUTE SCRIPT ====
if __name__ == "__main__":
    main()
