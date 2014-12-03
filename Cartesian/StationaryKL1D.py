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
    scale    = 0.0001
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
            dist = np.sqrt((ix1-ix2)**2)
            cvmatrix[ix1,ix2] = covar_exponential(dist,scale/size*nx)

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

    #---- plot eigenvalues ----
    plt.figure()
    plt.plot(evals[::-1])
    plt.title('Eigenvalues')

    #---- compute random model ----
    coeffs = np.random.normal(loc=0.,scale=1.,size=nx)
    values = np.dot(np.sqrt(evals).reshape(1,nx)*evecs,coeffs)
    values = values.reshape(nx)

    #---- plot random model section ----
    plt.figure()
    plt.plot(values)
    plt.title('model section')

    plt.show()


#==== EXECUTE SCRIPT ====
if __name__ == "__main__":
    main()
