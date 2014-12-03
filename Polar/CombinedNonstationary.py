#!/usr/bin/env python
"""
This script compares random models created in cartesian and polar coordinates
with Fourier and Fourier-Bessel transforms.
"""

#---- imports ----
from __future__ import division
import sys,os
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import eigh,cholesky
import numpy as np
import time

#---- custom imports ----
sys.path.append(os.path.join(os.path.dirname(__file__),'../CommonModules'))
from FigStyle import style_gji
mpl.rcParams.update(style_gji)
from CovarianceFunctions import power_exponential2d

PI = np.pi

#==== MAIN FUNCTION ====
def main():
    #---- initial parameters ----
    nr,nphi = 150,400

    #---- derived parameters ----
    nkphi = int(nphi/2)+1
    print('total number of non stationary points: {}'.format(nr))
    print('memory size of covariance matrix: {:1.3f}Mb'.format(nr**2*8/1024./1024.))

    #---- setup frequency space (real fft) and model grid ----
    kphis = np.arange(nkphi)
    rs    = np.linspace(0.25,0.5,nr)
    phis  = np.linspace(0.,2*PI,nphi+1)
    radgrid,phigrid = np.meshgrid(rs,phis)

    #---- fill horizontal coefficients with the right covariance one by one ----
    hcoeffs    =   np.empty( (nr,nkphi), dtype=np.complex)
    coeffs_buf =   np.empty(  nr, dtype=np.complex)
    hspectrum  =   np.empty( (nr,nkphi) )
    cvmatrix   =   np.empty( (nr,nr) )

    #define horizontal power and vertical correlation function
    hscales = 20*np.ones( nr )*rs
    vscales = np.ones( nr )*10.*2*PI
    def corr_vertical1(dist,rho,scale):
        rhoscale = scale*np.sqrt(1+(rho/scale)**2)
        return np.exp(-rhoscale*dist)*(rhoscale*dist+1.)

    #compute/plot covariance matrix, its eigenbasis and the correlated coefficients
    nplot_cvmatrices = 5
    fig,ax = plt.subplots(1,nplot_cvmatrices,sharex=True,sharey=True)
    ax[0].set_ylabel('radius 2')
    for kphi,ikphi in enumerate(kphis):
        print 'point {:d}/{:d}'.format(ikphi,nkphi)
        #create uncorrelated Fourier coefficients with unit power for every layer:
        coeffs_buf.real = np.random.normal(loc=0,scale=1,size=nr)
        coeffs_buf.imag = np.random.normal(loc=0,scale=1,size=nr)
        #fill covariance matrix:
        hpowers = np.sqrt(power_exponential2d(kphi,hscales))
        cvmatrix = np.outer(hpowers,hpowers)
        for ir1 in range(nr):
            for ir2 in range(ir1,nr):
                dist = abs(rs[ir1] - rs[ir2])
                zcorrelation = corr_vertical1(dist,kphi,np.sqrt(vscales[ir1]*vscales[ir2])/2.)
                cvmatrix[ir1,ir2]*=zcorrelation
                cvmatrix[ir2,ir1]*=zcorrelation

        #plot cvmatrix and save horizontal power
        if ikphi<5:
            ax[ikphi].imshow(np.copy(cvmatrix),aspect='equal')
            ax[ikphi].set(aspect=1,adjustable='box-forced', xticks=[], yticks=[], xlabel='radius 1',
                          title='$k_\phi = %d$'%kphi)

        hspectrum[:,ikphi] = np.diagonal(cvmatrix)*kphi

        #get eigenbasis of covariance matrix for a single coefficient
        w,E = eigh(cvmatrix)
        w[w<w.max()*1e-3] = 0.
        L = np.sqrt(w).reshape(1,len(w))*E
        hcoeffs[:,ikphi] = np.dot(L,coeffs_buf)

        #create correlated random variables using the cholesky decomposition
        #L = cholesky(cvmatrix)
        #hcoeffs[:,ikphi] = np.dot(coeffs_buf,L)


    fig.tight_layout(pad=0.1)

    #---- Fourier transform every radial layer ---
    model = np.empty( (nr,nphi) )
    for ir in range(nr):
        model[ir] = np.fft.irfft(hcoeffs[ir])
    model = np.fft.ifftshift(model,axes=1)
    model = np.hstack( (model,model[:,0].reshape(nr,1)) )

    #---- plot horizontal power spectrum ----
    fig,ax = plt.subplots(1,1) 
    kphigrid,rgrid = np.meshgrid(kphis,rs,indexing='ij')
    mesh = ax.pcolormesh(kphigrid,rgrid,hspectrum.transpose(),shading='gouraud')
    mesh.set_rasterized(True)
    ax.set(xlabel='$k_\phi$', ylabel='radius')
    ax.set_xscale('log',basex=2)
    ax.set_yscale('log',basey=2)
    ax.set_xlim( 1,nkphi )
    ax.set_ylim( rs[0],rs[-1] )
    fig.tight_layout(pad=0.1)

    #---- plot random model section ----
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    cm = ax.pcolormesh(phigrid, radgrid, model.transpose() ,shading='gouraud')
    cm.set_rasterized(True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0.0,0.5)
    ax.plot(phis,np.ones_like(phis)*0.25,c='black',lw=1)
    fig.tight_layout(pad=0.1)


    #---- plot in cartesian coordinates ----
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(model,aspect='equal')
    ax.set_xlabel('angle')
    ax.set_ylabel('radius')

    plt.show()

#==== EXECUTION ====
if __name__ == "__main__":
    main()
