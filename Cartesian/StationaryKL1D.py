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
import mydebug

#---- custom imports ----
sys.path.append(os.path.join(os.path.dirname(__file__),'../CommonModules'))
from CovarianceFunctions import covar_gaussian
from FigStyle import style_gji
mpl.rcParams.update(style_gji)

def main():
    #---- initial parameters ----
    scale    = 50
    size     = 1.
    nx = 200

    #---- derived parameters ----
    print('total number of points: {}'.format(nx))
    print('memory size of covariance matrix: {}Mb'.format(nx**2*8/1024./1024.))

    #---- create grid ----
    xcoords = np.linspace(-size/2.,size/2.-1/nx,nx)

    #--- create covariance matrix ---
    print('now filling covariance matrix (may take a while) ...')
    cvmatrix = np.zeros( (nx,nx) )
    for ix1 in range(nx):
        for ix2 in range(nx):
            dist = np.sqrt((xcoords[ix1]-xcoords[ix2])**2)
            cvmatrix[ix1,ix2] = covar_gaussian(dist,scale)

    #---- compute eigenvectors and values ----
    print 'computing eigenvectors and values with cholesky method'
    evals,evecs = eigh(cvmatrix)
    mask = evals>evals.max()*1e-2
    evecs_masked = evecs[:,mask]
    evals_masked = evals[mask]
    npoints,nvecs = evecs_masked.shape

    #---- plot covariance matrix ----
    defFigsize = mpl.rcParams['figure.figsize']
    cubesize = defFigsize[0],defFigsize[0]
    plt.figure(figsize=cubesize)
    plt.imshow(cvmatrix)
    plt.title('Covariance Matrix')

    #---- plot eigenvectors ----
    plt.figure()
    plt.imshow(evecs[:,::-1])
    plt.title('Eigenvectors')

    plt.figure()
    plt.plot( (evecs_masked*np.sqrt(evals_masked).reshape(1,nvecs))[:,-7:] )
    plt.title('Eigenvectors')

    #---- plot eigenvalues ----
    plt.figure()
    plt.plot(evals[::-1])
    plt.title('Eigenvalues')

    #---- compute random model ----
    coeffs = np.random.normal(loc=0.,scale=1.,size=nvecs)
    values = np.dot(np.sqrt(evals_masked).reshape(1,nvecs)*evecs_masked,coeffs)
    values = values.reshape(nx)

    #---- plot random model section ----
    plt.figure()
    plt.plot(values)
    plt.title('model section')

    plt.show()


#==== EXECUTE SCRIPT ====
if __name__ == "__main__":
    main()
