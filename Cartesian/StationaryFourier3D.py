#!/usr/bin/env python
"""
This script creates a stationary Gaussian random model with the classical
Fourier method, as for example described by Frankel & Clayton 1986.
"""

#---- imports ----
import os,sys
import numpy as np
from evtk.hl import gridToVTK

#---- custom imports ----
sys.path.append(os.path.join(os.path.dirname(__file__),'../CommonModules'))
from CovarianceFunctions import power_gaussian3d, power_exponential3d

#---- constants ----
from numpy import pi

def main():
    #---- initial parameters ----
    scale    = 20.
    nx,ny,nz = 256,256,256

    #--- set up frequency space for real fft (z dimension is symmetric) ---
    kx,ky,kz = nx,ny,nz/2+1
    size = 1.
    kxs = 2*pi/size*np.arange(-kx/2,kx/2)
    kys = 2*pi/size*np.arange(-ky/2,ky/2)
    kzs = 2*pi/size*np.arange(kz)
    kx_grid,ky_grid,kz_grid = np.meshgrid(kxs,kys,kzs,indexing='ij')

    #--- test normalization of cv function ---
    power_test = power_exponential3d(kzs,scale)
    power_tot = np.sum(4.*pi*kzs**2*(size/2./pi)**2*power_test) #divide by size/2PI due to fft normalization
    print 'integrated power is:',power_tot

    #--- create complex coefficients with power one ---
    coeffs = np.zeros( (kx,ky,kz),dtype=np.complex)
    coeffs.real = np.random.normal(loc=0,scale=1,size=kx*ky*kz).reshape(kx,ky,kz)/np.sqrt(2)
    coeffs.imag = np.random.normal(loc=0,scale=1,size=kx*ky*kz).reshape(kx,ky,kz)/np.sqrt(2)
    coeffs *= np.sqrt(power_exponential3d(np.sqrt(kx_grid**2+ky_grid**2+kz_grid**2), scale))

    #--- compute model via (real) inverse Fourier transform ---
    model = np.fft.irfftn(np.fft.fftshift(coeffs,axes=(0,1)),s=(nx,ny,nz))
    model = np.fft.ifftshift(model,axes=(0,1,2))
    model *= nx*ny*nz
    xcoords, ycoords, zcoords = np.meshgrid(np.linspace(-size/2.,size/2.-1/nx,nx),
                                            np.linspace(-size/2.,size/2.-1/ny,ny),
                                            np.linspace(-size/2.,size/2.-1/nz,nz),
                                            indexing='ij')

    #--- output model to vtk ---
    gridToVTK('vtk/stationary_exponential',xcoords,ycoords,zcoords,pointData= {'random field': model})
    print 'everything done...'


#==== EXECUTE SCRIPT ====
if __name__ == "__main__":
    main()
