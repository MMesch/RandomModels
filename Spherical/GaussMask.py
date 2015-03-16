#!/usr/bin/env python

#---- common imports ----
import sys,os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#---- custom imports ----
sys.path.append(os.path.join(os.path.dirname(__file__),'../CommonModules'))
from CovarianceFunctions import power_exponential3d, power_gaussian3d
from PartitionFunctions import make_partition
from CVMatrix import LayeredCovarianceMatrix, LayeredSphere
from FigStyle import style_gji
mpl.rcParams.update(style_gji)

#---- import constants ----
PI = np.pi

def main():
    """
    this function creates a nonisotropic model with different covariance
    functions in different depth regions.
    """

    #---- input parameters ----
    nr,nl = 200,180
    ls = np.arange(nl)
    radii = np.linspace(0.,1,nr)
    rhos  = np.arange(nr/2)

    np.random.seed(0)

    power = power_gaussian3d(2*PI*rhos,10)

    #---- create and show covariance matrix ----
    #I. fill each region with its covariance matrix
    cvmatrix_sphere = LayeredCovarianceMatrix(ls,radii)
    cvmatrix_sphere.fill_isospherical(rhos,power)

    cvmatrix_sphere.plot_cvmatrix(5,loglog=True,totpower=True)

    #---- create a model realization ----
    model_sphere = LayeredSphere(cvmatrix_sphere)
    model_sphere.plot_section()
    #model_sphere.write_vtk()

    plt.show()

if __name__ == "__main__":
    main()
