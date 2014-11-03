#!/usr/bin/env python
"""
This script containts and plots some common well-tested covariance functions
with their analytical Fourier transforms.
"""

#---- common imports ----
import numpy as np
import matplotlib.pyplot as plt

#---- import constants ----
PI = np.pi

#==== DIFFERENT SPECTRA AND CORRELATION FUNCTIONS ====
#---- exponential ----
def covar_exponential(dist,scale,sigma=1.):
    return sigma**2*np.exp(-dist*scale)

def power_exponential1d(rho,scale,sigma=1.):
    a = 1./scale
    kappa2 = 2.*sigma**2*a
    return kappa2/(1+(rho*a)**2)

def power_exponential2d(rho,scale,sigma=1.):
    a = 1./scale
    kappa2 = 2.*PI*sigma**2*a**2
    return kappa2/(1+(rho*a)**2)**(3./2.)

def power_exponential3d(rho,scale,sigma=1.):
    a = 1./scale
    kappa2 = 8.*PI*sigma**2*a**3
    return kappa2/(1+(rho*a)**2)**2

#---- gaussian ----
def covar_gaussian(dist,scale,sigma=1.):
    return sigma**2*np.exp(-dist**2*scale**2)

def power_gaussian1d(rho,scale,sigma=1.):
    a = 1./scale
    kappa2 = sigma**2*a*PI**(1./2.)
    return kappa2*np.exp(-(rho/scale)**2/4.)

def power_gaussian2d(rho,scale,sigma=1.):
    a = 1./scale
    kappa2 = sigma**2*a**2*PI
    return kappa2*np.exp(-(rho/scale)**2/4.)

def power_gaussian3d(rho,scale,sigma=1.):
    a = 1./scale
    kappa2 = sigma**2*a**3*PI**(3./2.)
    return kappa2*np.exp(-(rho/scale)**2/4.)

#==== MAIN AND TEST FUNCTIONS ====
def main():
    test_normalizations()

def test_normalizations():
    scale = 30

    #--- set up radial frequency space  ---
    krho = 128
    size = 1.0
    krhos = 2*PI/size*np.arange(krho)

    #--- testing exponential covariance functions ---
    #3d:
    power = power_exponential3d(krhos,scale)
    power_tot = np.sum(4.*PI*krhos**2*(size/2./PI)**2*power) 
    print 'integrated power in 3d is:',power_tot
    #2d:
    power = power_exponential2d(krhos,scale)
    power_tot = np.sum(2.*PI*krhos*(size/2./PI)*power) 
    print 'integrated power in 2d is:',power_tot
    #1d:
    power = power_exponential1d(krhos,scale)
    power_tot = np.sum(2*power[1:])+power[0]
    print 'integrated power in 1d is:',power_tot

    #--- testing Gaussian covariance functions ---
    #3d:
    power = power_gaussian3d(krhos,scale)
    power_tot = np.sum(4.*PI*krhos**2*(size/2./PI)**2*power)
    print 'integrated power in 3d is:',power_tot
    #2d:
    power = power_gaussian2d(krhos,scale) 
    power_tot = np.sum(2.*PI*krhos*(size/2./PI)*power)
    print 'integrated power in 2d is:',power_tot
    #1d:
    power = power_gaussian1d(krhos,scale)
    power_tot = np.sum(2.*power[1:])+power[0]
    print 'integrated power in 1d is:',power_tot

def test_3d():
    """plots fft of covariance functions against their analytical spectrum"""
    #---- fast computation of covariance matrix ----
    from ctypes import cdll,c_int
    covlib = cdll.LoadLibrary('./covariance.so')
    get_covariance = covlib.covariance
    get_covariance.restype = None
    from numpy.ctypeslib import ndpointer
    get_covariance.argtypes = [ndpointer(np.double),
                               ndpointer(np.double),
                               ndpointer(np.double),
                               c_int,c_int,c_int]

    #---- start function ----
    scale    = 30.
    nx,ny,nz = 96,96,96

    #--- set up frequency space for real fft (z dimension is symmetric) ---
    kx,ky,kz = nx,ny,nz/2+1
    size = 1.
    kxs = 2*PI/size*np.arange(-kx/2,kx/2)
    kys = 2*PI/size*np.arange(-ky/2,ky/2)
    kzs = 2*PI/size*np.arange(kz)
    kx_grid,ky_grid,kz_grid = np.meshgrid(kxs,kys,kzs,indexing='ij')

    #--- test normalization of cv functions ---
    power_test = power_exponential3d(kzs,scale) #evaluate at 2PI multiples
    power_tot = np.sum(4.*PI*kzs**2*(size/2./PI)**2*power_test) #divide by size/2PI due to fft normalization
    print 'integrated power is:',power_tot

    #create complex coefficients with power one
    coeffs = np.zeros( (kx,ky,kz),dtype=np.complex)
    coeffs.real = np.random.normal(loc=0,scale=1,size=kx*ky*kz).reshape(kx,ky,kz)/np.sqrt(2)
    coeffs.imag = np.random.normal(loc=0,scale=1,size=kx*ky*kz).reshape(kx,ky,kz)/np.sqrt(2)
    coeffs *= np.sqrt(power_exponential3d(np.sqrt(kx_grid**2+ky_grid**2+kz_grid**2), scale))

    #compute model via (real) inverse Fourier transform
    model = np.fft.irfftn(np.fft.fftshift(coeffs,axes=(0,1)),s=(nx,ny,nz))
    model = np.fft.ifftshift(model,axes=(0,1,2))
    model *= nx*ny*nz
    xcoords, ycoords, zcoords = np.meshgrid(np.linspace(-size/2.,size/2.-1/nx,nx),
                                            np.linspace(-size/2.,size/2.-1/ny,ny),
                                            np.linspace(-size/2.,size/2.-1/nz,nz),
                                            indexing='ij')

    #compute all point to point distances (periodic domain)
    #distance_matrix = np.sqrt((xcoords+0.5)**2+(ycoords+0.5)**2+(zcoords+0.5)**2)
    covariance_matrix = np.zeros( (nx,ny,nz) )
    distance_matrix = np.zeros( (nx,ny,nz) )
    get_covariance(model,distance_matrix,covariance_matrix,nx,ny,nz)
    distance_matrix *= size/nx

    #remove values that were not set
    mask = np.abs(covariance_matrix.flatten()) > 0.
    distance_matrix = distance_matrix.flatten()[mask]
    covariance_matrix = covariance_matrix.flatten()[mask]

    #bin data
    bins = np.arange(-size/nx,size/2.,size/2./nx)
    countsPerBin,bins = np.histogram(distance_matrix,bins=bins)
    sumsPerBin,  bins = np.histogram(distance_matrix,bins=bins,weights=covariance_matrix)
    binned_covariance = sumsPerBin / countsPerBin 
    centers = 0.5*(bins[1:] + bins[:-1])

    #plot covariance in dependence of distance
    plt.plot(distance_matrix,covariance_matrix,'o')
    plt.plot(centers,binned_covariance,'-')
    plt.plot(centers,covar_exponential3d(centers,scale))

    #compute and plot histograms
    modelmax = np.abs(model).max()
    bins = np.linspace(-modelmax,modelmax,30)
    hist, bins = np.histogram(model.flatten(), bins=bins, density=True)
    std = np.std(model.flatten())

    fig,ax = plt.subplots(1,1)
    center = (bins[:-1]+bins[1:])/2.
    width  = (bins[1]-bins[0])
    ax.bar(center,hist,align='center',color='red',width=0.8*width,alpha=0.5,
            label='cartesian, std=%f'%std)
    ax.set_ylim(0.,max(hist.max(),hist.max())*1.4)
    ax.legend()

#==== EXECUTE SCRIPT ====
if __name__ == "__main__":
    main()
