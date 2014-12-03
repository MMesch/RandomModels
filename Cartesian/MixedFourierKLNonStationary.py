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
sys.path.append(os.path.join(os.path.dirname(__file__),'../CommonModules'))
from CovarianceFunctions import power_exponential2d
from PartitionFunctions import make_partition

#---- matplotlib style----
from FigStyle import style_gji
mpl.rcParams.update(style_gji)

PI = np.pi

def main():
    #---- initial parameters ----
    nx,ny,nz = 100,100,200
    sizex,sizey,sizez = 1.,1.,2.

    #---- derived parameters ----
    ntot1d = nz
    nkx,nky = nx,ny/2+1
    print('total number of non stationary points: {}'.format(ntot1d))
    print('memory size of covariance matrix: {:1.3f}Mb'.format(ntot1d**2*8/1024./1024.))

    #---- setup frequency space ----
    kxs = 2*PI/sizex*np.arange(-nkx/2,nkx/2)
    kys = 2*PI/sizey*np.arange(nky)

    #---- create grid (reordered to allow fast xy plane access) ----
    zcoords, xcoords, ycoords = np.meshgrid(np.linspace(-sizez/2.,sizez/2.-1/nz,nz),
                                            np.linspace(-sizex/2.,sizex/2.-1/nx,nx),
                                            np.linspace(-sizey/2.,sizey/2.-1/ny,ny),
                                            indexing='ij')

    #---- fill horizontal coefficients with the right covariance one by one ----
    hcoeffs =    np.zeros( (nz,nkx,nky), dtype=np.complex)
    cvmatrix =   np.empty( (nz,nz) )
    coeffs_buf = np.empty( nz, dtype=np.complex)

    #define horizontal power and vertical correlation function
    dlocs = [-1.,nz/6.,2*nz/3.,nz+1.]
    dtrans = [0.,nz/18.,nz/9.,0.]
    partition = make_partition(dlocs,dtrans,np.arange(nz))
    hscales = np.zeros( (nz) )
    vscales = np.zeros( (nz) )
    hscales += 2.*PI*1./sizez*partition[0]
    hscales += 2.*PI*5./sizez*partition[1]
    hscales += 2.*PI*1./sizez*partition[2]
    plt.figure()
    plt.plot(hscales)
    vscales += 2.*PI*3./sizez*partition[0]
    vscales += 2.*PI*1./sizez*partition[1]
    vscales += 2.*PI*3./sizez*partition[2]
    plt.figure()
    plt.plot(hscales)
    def corr_vertical1(dist,rho,scale):
        rhoscale = scale*np.sqrt(1+(rho/scale)**2)
        return np.exp(-rhoscale*dist)*(rhoscale*dist+1.)

    #compute covariance matrix, its eigenbasis and the correlated coefficients
    rhomax = 10*max(hscales)
    for ikx,kx in enumerate(kxs):
        for iky,ky in enumerate(kys):
            rho = np.sqrt(kx**2+ky**2)
            if rho>rhomax: continue
            print 'point {:d}/{:d}'.format(ikx*nky+iky,nkx*nky)
            #create uncorrelated Fourier coefficients with unit power for every layer:
            coeffs_buf.real = np.random.normal(loc=0,scale=1,size=nz)
            coeffs_buf.imag = np.random.normal(loc=0,scale=1,size=nz)
            #fill covariance matrix:
            for z1 in range(nz):
                hpower1 = power_exponential2d(rho,hscales[z1])
                for z2 in range(z1,nz):
                    hpower2 = power_exponential2d(rho,hscales[z2])
                    hpower = np.sqrt(hpower1*hpower2)
                    dist = np.abs(z1-z2)/float(nz)
                    zcorrelation = corr_vertical1(dist,rho,np.sqrt(vscales[z1]*vscales[z2]))
                    covariance = hpower*zcorrelation
                    cvmatrix[z1,z2] = covariance
                    cvmatrix[z2,z1] = covariance
            #get eigenbasis of covariance matrix for a single coefficient
            w,E = eigh(cvmatrix)
            w[w<0] = 0.
            L = np.sqrt(w).reshape(1,len(w))*E
            #write coefficients to matrix:
            hcoeffs[:,ikx,iky] = np.dot(L,coeffs_buf)

    #---- Fourier transform every xy layer ---
    model = np.empty( (nz,nx,ny) )
    for iz in range(nz):
        coeffs = np.fft.fftshift(hcoeffs[iz],axes=0)
        model[iz] = np.fft.irfft2(coeffs,s=(nx,ny))
    model = np.fft.ifftshift(model,axes=(1,2))

    #---- output VTK model ----
    gridToVTK('vtk/mixedKL_nonstationary',zcoords,xcoords,ycoords,pointData= {'random field': model})

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
