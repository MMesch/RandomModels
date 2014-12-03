#!/usr/bin/env python
import os,sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__),'../../Spherical'))
from CVMatrix import LayeredCovarianceMatrix

def main():
    #create irregular radial grid
    dr_lowermantle = 40.
    dr_uppermantle = 10.
    depth_cmb = 2885.
    depth_tz  = 660.
    depth_moho = 25.
    rearth = 6371.
    rcmb = rearth - depth_cmb
    rtz   = rearth - depth_tz
    rmoho = rearth - depth_moho
    nr_lowermantle = np.ceil((rtz-dr_lowermantle-rcmb)/dr_lowermantle)
    nr_uppermantle = np.ceil((rmoho-rtz)/dr_uppermantle)

    radiis_lm   = np.linspace(rcmb,rtz-dr_lowermantle,nr_lowermantle)
    radiis_um   = np.linspace(rtz,rmoho,nr_uppermantle)
    radiis      = np.append(radiis_lm,radiis_um)

    #degrees
    lmax = 150
    ls = np.arange(lmax+1)

    #fill covariance matrix with medium defined in the .mod file
    cvmatrix = LayeredCovarianceMatrix(ls,radiis)
    cvmatrix.fillfromfile(sys.argv[-1])
    cvmatrix.plot_cvmatrix(25,totpower=True)
    hpower = cvmatrix.get_hpower()

    plt.figure()
    plt.plot(ls,hpower[:,50]*ls)

    plt.show()

if __name__ == "__main__":
    main()
