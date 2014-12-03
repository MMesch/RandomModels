#!/usr/bin/env python
"""
This script compares random models created in cartesian and polar coordinates
with Fourier and Fourier-Bessel transforms.
"""

from __future__ import division
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import jn
from configs import style_gji
mpl.rcParams.update(style_gji)
import numpy as np
import mydebug
import ipdb


#==== MAIN FUNCTION ====
def main():
    nx,ny = 512,512
    kx,ky = nx,ny/2+1
    nr,nphi = 300,500
    kr,kphi = 250,250

    niter = 1
    for it in range(niter):
      print 'filling cartesian coefficients with random numbers'
      kxs = np.arange(-kx/2,kx/2)
      kys = np.arange(ky)
      coeffs_carte = np.zeros( (kx,ky)   ,dtype=np.complex)
      #coeffs_carte[:,:] = 1.
      coeffs_carte.real = np.random.normal(loc=0,scale=1,size=kx*ky).reshape(kx,ky)
      coeffs_carte.imag = np.random.normal(loc=0,scale=1,size=kx*ky).reshape(kx,ky)
  
      print 'filling polar coefficients with random numbers'
      rhos = np.arange(kr)
      coeffs_polar = np.zeros( (kphi,kr) ,dtype=np.complex)
      #coeffs_polar[0,:] = rhos #this is the total variance per ring
      coeffs_polar.real = np.random.normal(loc=0,scale=1,size=kphi*kr).reshape(kphi,kr)
      coeffs_polar.imag = np.random.normal(loc=0,scale=1,size=kphi*kr).reshape(kphi,kr)
      coeffs_polar *= np.sqrt(rhos).reshape(1,kr)/np.sqrt(2*np.pi) #distribute power over all coefficients
  
      #==== FILTER AND PLOT COEFFICIENTS ====
      print 'filtering coefficients to exponential random medium'
      scale = 15
  
      print 'cartesian...'
      kx_grid,ky_grid = np.meshgrid(kxs,kys,indexing='ij')
      coeffs_carte *= power_exponential(np.sqrt(kx_grid**2+ky_grid**2), scale)
      coeffs_carte *= gaussian(np.sqrt(kx_grid**2+ky_grid**2), 50)
      #coeffs_carte *= ring(np.sqrt(kx_grid**2+ky_grid**2), scale, rho0=50)
  
      print 'polar...'
      #coeffs_polar *= gaussian(rhos,scale).reshape(1,kr)
      coeffs_polar *= power_exponential(rhos,scale).reshape(1,kr)
      coeffs_polar *= gaussian(rhos, 50).reshape(1,kr)
      #coeffs_polar *= ring(rhos,scale,rho0=50).reshape(1,kr)
      rho_grid, theta_grid = np.meshgrid(np.arange(kr),np.arange(kphi))
  
  
      #==== TRANSFORMS ====
      print 'making transforms'
      image_carte = np.fft.irfft2(np.fft.fftshift(coeffs_carte,axes=0),s=(nx,ny))
      image_carte = np.fft.ifftshift(image_carte,axes=(0,1))
      xcoords, ycoords = np.meshgrid(np.linspace(-0.5,0.5-1/nx,nx),np.linspace(-0.5,0.5-1/ny,ny),indexing='ij')
      image_carte *= nx*ny/(2*np.pi)**2 #continous fourier convention
  
      image_polar = fourier_bessel(coeffs_polar, rhos, nphi, nr)
      dphi = 2*np.pi/nphi
      krs,kps = np.meshgrid(np.linspace(0,0.5,nr),np.linspace(0,2*np.pi-dphi,nphi))

      max_carte = np.abs(image_carte).max()
      max_polar = np.abs(image_polar).max()
      max_all   = max(max_carte,max_polar)
      bins = np.linspace(-max_all,max_all,30)
      std_carte,hist_carte,dummy = cartesian_distribution(image_carte,bins)
      std_polar,hist_polar,dummy = polar_distribution(krs,kps,image_polar,bins)
      if it==0:
        stdav_carte=std_carte
        stdav_polar=std_polar
        histav_carte=hist_carte
        histav_polar=hist_polar
      else:
        stdav_carte+=std_carte
        stdav_polar+=std_polar
        histav_carte+=hist_carte
        histav_polar+=hist_polar

    fig,axes = plt.subplots(1,2)
    axes[0].set_title('cartesian coefficients')
    axes[0].pcolormesh(kx_grid,ky_grid,np.abs(coeffs_carte))
    axes[0].set_xlabel('x order')
    axes[0].set_ylabel('y order')
    axes[1].set_title('polar coefficients')
    axes[1].pcolormesh(theta_grid,rho_grid, np.abs(coeffs_polar))
    axes[1].set_xlabel('angular order')
    axes[1].set_ylabel('radial order')
    fig.tight_layout(pad=0.1,w_pad=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    norm = image_carte.max()/image_polar.max()
    ax.plot(xcoords[:,0],image_carte[:,ny/2],label='cartesian')
    ax.plot(krs[0,:],image_polar[0,:],label='polar')
    ax.set_title('norm: %g'%norm)
    ax.legend()

    fig = plt.figure()
    norm_carte = plt.Normalize(-max_all,max_all)
    norm_polar = plt.Normalize(-max_all,max_all)
    ax1 = fig.add_subplot(121, polar=False)
    ax2 = fig.add_subplot(122, polar=True)

    ax1.set_title('cartesian model')
    cm=ax1.pcolormesh(xcoords,ycoords,image_carte,norm=norm_carte)
    cm.set_rasterized(True)
    ax1.set_xlim(-0.5,0.5)
    ax1.set_ylim(-0.5,0.5)
    ax1.set_aspect(1.)

    ax2.set_title('polar model')
    #ax2.pcolormesh(krs*np.cos(kps),krs*np.sin(kps),image_polar,norm=norm_polar)
    cm = ax2.pcolormesh(kps,krs,image_polar,norm=norm_polar)
    cm.set_rasterized(True)
    ax2.set_xticks([])
    #ax2.set_xlim(-0.2,0.2)
    #ax2.set_ylim(-0.2,0.2)
    ax2.set_aspect(1.)
    fig.tight_layout(pad=0.1)

    fig,ax = plt.subplots(1,1)
    center = (bins[:-1]+bins[1:])/2
    width  = (bins[1]-bins[0])
    ax.set_title('std factor: %f'%(stdav_carte/stdav_polar))
    ax.bar(center,histav_carte,align='center',color='red',width=0.8*width,alpha=0.5,
            label='cartesian, std=%f'%stdav_carte)
    ax.bar(center,histav_polar,align='center',color='blue',width=0.8*width,alpha=0.5,
            label='polar, std=%f'%stdav_polar)
    ax.set_ylim(0.,max(histav_polar.max(),histav_carte.max())*1.4)
    ax.legend()

    plt.show()

#---- different spectra and correlation functions ----
def power_exponential(rho,scale):
    return 1/(1 + (rho/scale)**2)**(3/4)

def gaussian(rho,scale):
    return np.exp(-(rho/scale)**2)

def ring(rho,scale,rho0=5):
    return np.exp(-((rho-rho0)/scale)**2)

def cartesian_distribution(image,bins):
    hist, bins = np.histogram(image.flatten(), bins=bins, density=True)
    std = np.std(image.flatten())
    return std,hist,bins

def polar_distribution(rs,phis,image,bins):
    weights = rs*2*np.pi*phis
    hist, bins = np.histogram(image.flatten(), bins=bins, weights=weights.flatten(),density=True)
    mean = np.average(image.flatten(),weights=weights.flatten())
    variance = np.average((image.flatten()-mean)**2, weights=weights.flatten())
    std = np.sqrt(variance)
    return std,hist,bins

#---- get fourier bessel expansion
def fourier_bessel(coeffs, rhos, nphi, nr):
    kpmax,krmax = coeffs.shape
    if nphi < 2*kpmax-1:
        print 'nphi not large enough'

    #compute integrand
    radii = np.linspace(0,0.5,nr)
    bessel = np.zeros( (kpmax,krmax,nr) ,dtype=np.complex)
    for n in range(kpmax):
        bessel[n] = coeffs[n].reshape(krmax,1)*jn(n,2*np.pi*np.outer(rhos,radii))

    #integrate (sum) over rho (drho=1)
    coeffs_new = np.zeros( (kpmax,nr), dtype=np.complex)
    coeffs_new = 1/(2*np.pi)*np.sum(bessel,axis=1)

    #Fourier transform in phi direction (interpolate to nphi values)
    image = np.zeros( (nphi,nr) )
    dphi_in = 2*np.pi/(2*(kpmax-1))
    phis_in = np.linspace(0.,2.*np.pi-dphi_in, 2*(kpmax-1))
    dphi_out = 2*np.pi/nphi
    phis_out = np.linspace(0.,2.*np.pi-dphi_out, nphi)
    for k in range(nr):
        fft = np.fft.irfft(coeffs_new[:,k])*2*(kpmax-1)
        image[:,k] = np.interp(phis_out, phis_in, fft)
        #image[:,k] = coeffs_new[0,k]
    return image

#==== EXECUTION ====
if __name__ == "__main__":
    main()
