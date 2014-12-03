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

    #---- partition the mesh into a set of layers with smooth transition ----
    power1 = power_gaussian3d(2*PI*rhos,50)
    power2 = 0.6*power_gaussian3d(2*PI*rhos,18)
    power3 = power_exponential3d(2*PI*rhos,10)

    dlocs  = [-1,0.30,0.85,1.4]
    dtrans = [0.0,0.2,0.15,0.0]
    tfuncs = make_partition(dlocs,dtrans,radii)

    fig,ax = plt.subplots(1,1)
    ax.set_title('transfer functions')
    for tfunc in tfuncs:
        ax.plot(radii,tfunc)

    #---- create and show covariance matrix ----
    #I. fill each region with its covariance matrix
    cvmatrix_sphere = LayeredCovarianceMatrix(ls,radii)
    mat1 = cvmatrix_sphere.get_isospherical(rhos,power1)

    tfuncs_ani = make_partition([-1,10.,22.,nl],[0.,2.,5.,0.],ls)
    stretch2 = tfuncs_ani[0]*0.1 + tfuncs_ani[1]*0.1+tfuncs_ani[1]*1.0
    mat2 = cvmatrix_sphere.get_anisospherical(rhos,power2,stretch2)

    tfuncs_ani = make_partition([-1,7.,20.,nl],[0.,4.,7.,0.],ls)
    stretch3 = tfuncs_ani[0]*0.1 + tfuncs_ani[1]*1.4 + tfuncs_ani[2]*1.0
    mat3 = cvmatrix_sphere.get_anisospherical(rhos,power3,stretch3)

    #II. combine cvmatrices
    partition = np.outer(tfuncs[0],tfuncs[0])
    cvmatrix_sphere.cvmatrix += mat1*partition

    partition = np.outer(tfuncs[1],tfuncs[1])
    cvmatrix_sphere.cvmatrix += mat2*partition

    partition = np.outer(tfuncs[2],tfuncs[2])
    cvmatrix_sphere.cvmatrix += mat3*partition

    partition = np.outer(tfuncs[1],tfuncs[2])
    cvmatrix_sphere.cvmatrix += 0.5*(mat2+mat3)*(partition+partition.transpose())

    partition = np.outer(tfuncs[0],tfuncs[1])
    cvmatrix_sphere.cvmatrix += 0.5*(mat1+mat2)*(partition+partition.transpose())

    cvmatrix_sphere.plot_cvmatrix(5,loglog=True,totpower=True)

    del partition,mat1,mat2,mat3

    #---- create a model realization ----
    #model_sphere = LayeredSphere(cvmatrix_sphere)
    #model_sphere.plot_section()
    #model_sphere.write_vtk()

    plt.show()

if __name__ == "__main__":
    main()
