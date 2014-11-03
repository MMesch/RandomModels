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

    np.random.seed(1)

    #---- partition the mesh into a set of layers with smooth transition ----
    power1 = power_gaussian3d(2*PI*rhos,50)
    power2 = 0.7*power_gaussian3d(2*PI*rhos,20)
    power3 = power_exponential3d(2*PI*rhos,10)

    dlocs  = [-1,0.30,0.8,1.4]
    dtrans = [0.0,0.1,0.2,0.0]
    tfuncs = make_partition(dlocs,dtrans,radii)

    fig,ax = plt.subplots(1,1)
    ax.set_title('transfer functions')
    for tfunc in tfuncs:
        ax.plot(radii,tfunc)

    #---- create and show covariance matrix ----
    cvmatrix_sphere = LayeredCovarianceMatrix(ls,radii)
    cvmatrix_sphere.fill_isospherical(rhos,power1,rtransfer=tfuncs[0])

    tfuncs_ani = make_partition([-1,10.,22.,nl],[0.,2.,2.,0.],ls)
    stretch = tfuncs_ani[0]*0.05 + tfuncs_ani[1]*0.05+tfuncs_ani[1]*1.0
    cvmatrix_sphere.fill_anisospherical(rhos,power2,stretch,rtransfer=tfuncs[1])

    tfuncs_ani = make_partition([-1,5.,15.,nl],[0.,2.,2.,0.],ls)
    stretch = tfuncs_ani[0]*0.1 + tfuncs_ani[1]*1.4 + tfuncs_ani[2]*1.0
    cvmatrix_sphere.fill_anisospherical(rhos,power3,stretch,rtransfer=tfuncs[2])

    cvmatrix_sphere.plot_cvmatrix(5,loglog=True,totpower=True)

    #---- create a model realization ----
    model_sphere = LayeredSphere(cvmatrix_sphere)
    model_sphere.plot_section()
    model_sphere.write_vtk()

    plt.show()

if __name__ == "__main__":
    main()
