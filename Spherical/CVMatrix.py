#!/usr/bin/env python
"""
This script provides a class that can store, manipulate and plot the covariance
matrix of a spherically symmetric, Gaussian random model.
"""

#common library imports
from __future__ import division
import sys,os
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import sph_jn
from scipy.linalg import eigh
from scipy.ndimage import map_coordinates
from ConfigParser import SafeConfigParser
import numpy as np
from evtk.hl import gridToVTK

#custom modules
sys.path.append(os.path.join(os.path.dirname(__file__),'../CommonModules'))
from CovarianceFunctions import power_exponential3d, power_gaussian3d, power_scalefree3d
from PartitionFunctions import make_partition
from FigStyle import style_gji
mpl.rcParams.update(style_gji)

#spherical harmonics transforms from shtools library
from shtools import pymakegriddh as shmakegrid

#debugging
import mydebug

#---- import constants ----
PI = np.pi

#==== MAIN FUNCTION ====
def main():
    """
    this function creates a nonisotropic model with different covariance
    functions in different depth regions.
    """
    test2()

def test1():
    #---- input parameters ----
    nr,nl = 200,180
    ls = np.arange(nl)
    radii = np.linspace(0.,1,nr)
    rhos  = np.arange(nr/2)

    #---- partition the mesh into a set of layers with smooth transition ----
    power1 = power_gaussian3d(2*PI*rhos,50)
    power2 = power_gaussian3d(2*PI*rhos,16)
    power3 = power_exponential3d(2*PI*rhos,10)

    dlocs  = [-1,0.30,0.8,1.4]
    dtrans = [0.0,0.2,0.2,0.0]
    tfuncs = make_partition(dlocs,dtrans,radii)

    fig,ax = plt.subplots(1,1)
    ax.set_title('transfer functions')
    for tfunc in tfuncs:
        ax.plot(radii,tfunc)

    #---- create and show covariance matrix ----
    cvmatrix_sphere = LayeredCovarianceMatrix(ls,radii)
    cvmatrix_sphere.fill_isospherical(rhos,power1,rtransfer=tfuncs[0])

    stretch = 0.01 * (np.tanh((ls - 20.)/3. * np.pi/2)+1)/2.
    cvmatrix_sphere.fill_anisospherical(rhos,power2,stretch,rtransfer=tfuncs[1])

    tfuncs_ani = make_partition([-1,14.,nl],[0.,5.,0.],ls)
    stretch = tfuncs_ani[0]*0.4 + tfuncs_ani[1]*1.0
    cvmatrix_sphere.fill_anisospherical(rhos,power3,stretch,rtransfer=tfuncs[2])

    cvmatrix_sphere.plot_cvmatrix(5,loglog=True,totpower=True)
    plt.show()

def test2():
    #---- input parameters ----
    nr,nl = 200,180
    ls = np.arange(nl)
    radii = np.linspace(0.,1,nr)
    rhos  = np.arange(nr/2)

    #---- partition the mesh into a set of layers with smooth transition ----
    power1 = power_gaussian3d(2*PI*rhos,50)
    #---- create and show covariance matrix ----
    cvmatrix_sphere1 = LayeredCovarianceMatrix(ls,radii)
    cvmatrix_sphere1.fill_isospherical(rhos,power1)
    cvmatrix_sphere1.plot_cvmatrix(5,loglog=True,totpower=True)

    stretch = 0.5*np.ones_like(ls)
    cvmatrix_sphere2 = LayeredCovarianceMatrix(ls,radii)
    cvmatrix_sphere2.fill_anisospherical(rhos,power1,stretch)
    cvmatrix_sphere2.plot_cvmatrix(5,loglog=True,totpower=True)

    plt.show()

#==== SPHERICAL MODEL ====
class LayeredSphere(object):
    def __init__(s, cvmatrix):
        """
        This function creates the layered sphere. The random model is fully
        defined by the covariance matrices for each angular order l.
        """
        #compute array sizes in space and frequency
        s.cvmatrix = cvmatrix
        s.nl   = cvmatrix.nl
        s.lmax,s.nr, = cvmatrix.lmax,cvmatrix.nr
        s.nlat,s.nlon = s.nl*2,s.nl*4
        s.dlat = np.pi/s.nlat
        s.nrho = s.nr/2.

        #create coordinate arrays in space and frequency
        s.radii  = cvmatrix.radii
        s.rmin,s.rmax = s.radii[0],s.radii[-1]
        s.thetas = np.linspace(0+s.dlat/2,   np.pi-s.dlat/2,   s.nlat)
        s.phis   = np.linspace(0+s.dlat/2, 2*np.pi-s.dlat/2,   s.nlon)
        s.rhos   = np.arange(s.nrho)
        s.ls     = np.arange(s.lmax)

    def get_model(s):
        data   = np.zeros( (s.nr,s.nlat,s.nlon) )
        coeffs = np.zeros( (s.nr,2,s.nl,s.nl) )
        for l in s.ls:
            evals,evecs = s.cvmatrix.get_basis(l,thresh=0.)
            nbasis = len(evals)
            #create 'nbasis' random coefficients with l+1 entries
            alm = np.random.normal(loc=0.,scale=1.,size=(l+1)*nbasis).reshape(nbasis,l+1)
            blm = np.random.normal(loc=0.,scale=1.,size=(l+1)*nbasis).reshape(nbasis,l+1)
            #multiply with eigenvector basis to get the correct covariance matrix
            coeffs[:,0,l,:l+1] = np.dot(np.sqrt(evals).reshape(1,nbasis)*evecs,alm)
            coeffs[:,1,l,:l+1] = np.dot(np.sqrt(evals).reshape(1,nbasis)*evecs,blm)

        #make spherical harmonics transform for all layers
        for ir in range(s.nr):
            data[ir] = shmakegrid(coeffs[ir])
        return data

    #---- plot a spherical section ----
    def plot_section(s):
        fig = plt.figure()
        ax  = fig.add_subplot(111,polar=True)
        ax.set_title('spherical model slice')
        data = s.get_model()
        norm = plt.Normalize(-data.max(),data.max())
        rgrid,tgrid,pgrid = np.meshgrid(s.radii,s.thetas,s.phis,indexing='ij')
        cm = ax.pcolormesh(pgrid[:,s.nlat/2,:],rgrid[:,s.nlat/2,:],data[:,s.nlat/2,:],norm=norm,shading='gouraud')
        cm.set_rasterized(True)
        ax.set_xticks([])
        fig.tight_layout(pad=0.1)

    def write_vtk(s):    
        data = s.get_model()
        rgrid,tgrid,pgrid = np.meshgrid(s.radii,s.thetas,s.phis,indexing='ij')
        xs = rgrid*np.sin(tgrid)*np.cos(pgrid)
        ys = rgrid*np.sin(tgrid)*np.sin(pgrid)
        zs = rgrid*np.cos(tgrid)
        gridToVTK('sphere_noniso',xs,ys,zs,pointData= {'random field': data})

#==== COVARIANCE MATRIX CLASS ====
class LayeredCovarianceMatrix(object):
    """
    This class stores and fills a covariance matrix. A layered model has
    effectively a single dimension in which it can be nonisotropic and
    nonstationary. The other (isotropic) dimensions can be collapsed and
    described by a single 'order' parameter, that can correspond to: sqrt(k_x^2
    + k_y^2), to the spherical harmonics order or to the angular fourier order
    k_phi in polar coordinates. This class does not know anything about the geometry
    of the problem. This comes in the model class.
    """
    def __init__(s,ls,radii):
        """provide a list of orders and of radii"""
        #setup parameters
        s.ls = ls
        s.radii = radii
        s.nl = len(ls)
        s.nr = len(radii)
        s.rmin,s.rmax = radii[0],radii[-1]
        s.lmin,s.lmax = ls[0],ls[-1]

        #this is the main data array
        s.cvmatrix = np.zeros( (s.nl,s.nr,s.nr) )


    def fillfromfile(s,fname):
        """
        fills covariance matrix from a layer file (.mod)
        """
        #---- read model configuration from parameter file ----
        config = SafeConfigParser()
        config.read(fname)

        rposition  = 6371.-np.fromstring(config.get('layers','rposition'),sep=' ')[::-1]
        rtransit   = np.fromstring(config.get('layers','rtransit'),sep=' ')[::-1]

        mtype      = config.get('medium','type').split()[::-1]
        power_degs = np.fromstring(config.get('medium','power_degs'),sep=' ')[::-1]
        power_vals = np.fromstring(config.get('medium','power_vals'),sep=' ')[::-1]

        aniso      = np.fromstring(config.get('anisotropy','aniso'),sep=' ')[::-1]
        liso       = np.fromstring(config.get('anisotropy','liso'),sep=' ')[::-1]
        diso       = np.fromstring(config.get('anisotropy','diso'),sep=' ')[::-1]

        #---- compute some derived parameters ----
        nlayers = len(mtype)
        assert len(rposition) == nlayers+1
        assert len(rtransit) == nlayers+1
        npowers = len(power_degs)
        assert len(aniso) == nlayers*npowers
        assert len(liso)  == nlayers*npowers
        assert len(diso)  == nlayers*npowers

        print '---- building model from configuration file ----'
        print fname

        #---- input parameters ----
        nrhos = int(s.nl/2.)
        rhos  = np.arange(nrhos)

        #---- partition the mesh into a set of layers with smooth transition ----
        tfuncs = make_partition(rposition,rtransit,s.radii)
        pspectra = np.zeros( (nlayers,nrhos) )
        for imedium,medium in enumerate(mtype):
            if medium == 'sf':
                pspectra[imedium] = power_scalefree3d(2*PI*rhos,1.4*power_degs[0])*power_vals[imedium]

        #fig,ax = plt.subplots(1,2)
        #ax[0].set_title('transfer functions')
        #ax[1].set_title('power spectra')
        #for tfunc in tfuncs:
        #    ax[0].plot(s.radii,tfunc)
        #for power in pspectra:
        #    ax[1].plot(power)

        #---- create and show covariance matrix ----
        matrices = []
        for ilayer in range(nlayers):
            tfuncs_ani = make_partition([-1,liso[ilayer],s.nl],[0.,diso[ilayer],0.],s.ls)
            stretch = aniso[ilayer]*tfuncs_ani[0] + tfuncs_ani[1]*1.0
            matrices.append(s.get_anisospherical(rhos,pspectra[ilayer],stretch))

        for ilayer1 in range(nlayers):
            #fill layer:
            partition = np.outer(tfuncs[ilayer1],tfuncs[ilayer1])
            s.cvmatrix += matrices[ilayer1]*partition
            for ilayer2 in range(ilayer1+1,nlayers):
                #fill crossterms:
                partition = np.outer(tfuncs[ilayer1],tfuncs[ilayer2])
                s.cvmatrix += 0.5*(matrices[ilayer1]+matrices[ilayer2])\
                                 *(partition+partition.transpose())

    #---- different covariance functions ----
    def get_isospherical(s, rhos, power, rtransfer=None,ltransfer=None):
        print "computing cv matrix from cartesian power spectrum"
        nrho = len(rhos)

        #compute integrand
        bessel = np.zeros( (s.nl,nrho,s.nr) )
        for ir in range(s.nr):
            for irho in range(nrho):#the factor 1/(2*s.rmax) ensures good sampling
                bessel[:,irho,ir] = np.sqrt(4*PI*rhos[irho]**2*power[irho]) *\
                                    sph_jn(s.lmax,2*np.pi*rhos[irho]*s.radii[ir]/(2*s.rmax))[0]
    
        #compute and sum product of bessel functions and coefficients
        cvmatrix = np.zeros_like(s.cvmatrix)
        for n in range(s.nl):
            for irho in range(nrho):
                cvmatrix[n] += np.outer(bessel[n,irho],bessel[n,irho])

        #explicitely free bessel function memory and damp with transfer function
        del bessel

        #if rtransfer is not None:
        #    cvmatrix[:]*=np.outer(np.sqrt(rtransfer),np.sqrt(rtransfer))
        if rtransfer is not None:
            cvmatrix *= rtransfer
        if ltransfer is not None:
            cvmatrix   *=ltransfer.reshape(s.nl,1,1)
        return cvmatrix

    def get_anisospherical(s, rhos, power, stretch, rtransfer=None):
        print "computing anisotropic cv matrix from cartesian power spectrum"
        nrho = len(rhos)
        #compute integrand
        bessel = np.zeros( (s.nl,nrho,s.nr) )
        for ir in range(s.nr):
            for irho in range(nrho):#the factor 1/(2*s.rmax) ensures good sampling
                bessel[s.lmin:,irho,ir] = np.sqrt(4*PI*rhos[irho]**2*power[irho]) *\
                                    sph_jn(s.lmax,2*np.pi*rhos[irho]*s.radii[ir]/(2*s.rmax))[0]

        #directly compute cvmatrix
        cvmatrix_buf = np.zeros( (s.nr,s.nr) )
        cvmatrix = np.zeros( (s.nl,s.nr,s.nr) )

        #stretch matrix around diagonal!
        idx1,idx2 = np.meshgrid(np.arange(s.nr),
                                np.arange(s.nr),indexing='ij')
        for n in range(s.nl):
            cvmatrix_buf[:,:] = 0.
            idx_sampl1 = (1+stretch[n])/2.*idx1+(1-stretch[n])/2.*idx2
            idx_sampl2 = (1+stretch[n])/2.*idx2+(1-stretch[n])/2.*idx1
            ipoints = np.vstack((idx_sampl1.flatten(),idx_sampl2.flatten()))
            for irho in range(nrho):
                cvmatrix_buf += np.outer(bessel[n,irho],bessel[n,irho])
            cvmatrix[n] = map_coordinates(cvmatrix_buf,ipoints,mode='nearest').reshape(s.nr,s.nr)

        #explicitely free bessel function memory and damp with transfer function
        del bessel

        if rtransfer is not None:
            cvmatrix *= rtransfer

        return cvmatrix

    def get_hpower_from_3dpower(s,rhos,power,radius):
        print "computing horizontal power from 3D Fourier power"
        nrho = len(rhos)
        ls = np.arange(s.nl)

        #compute integrand
        bessel_matrix = np.zeros( (nrho,s.nl) )
        for irho in range(nrho):
            bessel_matrix[irho] = 4*PI*rhos[irho]**2*sph_jn(s.lmax,2*PI*rhos[irho]*radius/(2*s.rmax))[0]**2
        plt.figure()
        plt.imshow(bessel_matrix)

        hpower = np.dot(power,bessel_matrix)*(2*ls+1)
        return ls,hpower

    def get_3dpower_from_hpower(s,hpower,radius):
        print "computing 3d Fourier power from horizontal power spectrum"
        nrho = int(s.nl)
        rhos = np.arange(nrho)
        ls = np.arange(len(hpower))

        #compute integrand
        bessel_matrix = np.zeros( (nrho,s.nl) )
        for irho in range(nrho):
            bessel_matrix[irho] = 4*PI*rhos[irho]**2*sph_jn(s.lmax,2*np.pi*rhos[irho]*radius/(2*s.rmax))[0]**2

        inv_matrix = np.linalg.pinv(bessel_matrix,rcond=1e-2)
        power = np.dot(hpower/(2*ls+1),inv_matrix)
        return rhos,power

    def fill_iso_from_hpower(s,hpower,ir):
        s.cvmatrix += s.get_isocvmatrix_from_hpower(hpower,ir)

    def fill_isospherical(s,rhos,power,rtransfer=None):
        s.cvmatrix += s.get_isospherical(rhos,power,rtransfer=rtransfer)
        s.cvmatrix = 0.5*(s.cvmatrix + s.cvmatrix.transpose((0,2,1)))

    def fill_anisospherical(s,rhos,power,aniso,rtransfer=None):
        s.cvmatrix += s.get_anisospherical(rhos,power,aniso,rtransfer=rtransfer)
        s.cvmatrix = 0.5*(s.cvmatrix + s.cvmatrix.transpose((0,2,1)))

    def fill_horizpower(s,hpower):
        """hpower: nl x nr array with horizontal power spectrum"""
        s.cvmatrix[:,range(s.nr),range(s.nr)] += hpower
        s.cvmatrix = 0.5*(s.cvmatrix + s.cvmatrix.transpose((0,2,1)))

    def fill_crosspower(s,vcorr):
        """vcorr: function that takes arguments (distances,degree)"""
        cvmatrix_buf = np.empty( (s.nr,s.nr) )
        for il in range(s.nl):
            for ir1 in range(s.nr):
                hpower1 = s.cvmatrix[il,ir1,ir1]
                hpowers2 = np.diagonal(s.cvmatrix[il])
                distances = np.abs(s.radii[ir1] - s.radii)/s.radii[-1]
                hpowers = np.sqrt(hpower1*hpowers2)
                cvmatrix_buf[ir1] = hpowers*vcorr(distances,il)
            cvmatrix_buf[range(s.nr),range(s.nr)] = 0.
            s.cvmatrix[il] += cvmatrix_buf
        #s.cvmatrix = 0.5*(s.cvmatrix + s.cvmatrix.transpose((0,2,1)))

    #---- Eigenvector basis ----
    def get_basis(s,order,thresh=0.2):
        w,E = eigh(s.cvmatrix[order])
        valmax = w.max()
        ithresh = w>(valmax*thresh)
        return w[ithresh],E[:,ithresh]

    def plot_basis(s,order,thresh=0.2):
        w,E = s.get_basis(order,thresh=thresh) #get all eigenvalues
        valmax = w.max()
        ithresh = w>(valmax*thresh)
        fig,axes = plt.subplots(1,2)
        axes[1].plot(w)
        axes[1].plot([0,len(w)],[valmax*thresh,valmax*thresh])
        for evalue,evector in zip(w[ithresh],np.transpose(E)[ithresh]):
            color = mpl.cm.jet(evalue/valmax)
            axes[0].plot(s.radii,evector,c=color)

    def get_hpower(s):
        hpower = np.diagonal(s.cvmatrix,axis1=1,axis2=2)*(2*s.ls.reshape(s.nl,1)+1)
        return hpower

    #---- plotting covariance matrix and power spectrum (diagonal of cvmatrix) ----
    def plot_cvmatrix(s, order, loglog=False, totpower=False):
        fig,axes = plt.subplots(1,2)
        axes[0].set_title('cvmatrix (order %d)'%order)
        axes[1].set_title('log psd')

        #plot cvmatrix of a single order
        rgrid1,rgrid2 = np.meshgrid(s.radii,s.radii,indexing='ij')
        mesh = axes[0].pcolormesh(rgrid1,rgrid2,s.cvmatrix[order],shading='gouraud')
        mesh.set_rasterized(True)
        del rgrid1,rgrid2
        axes[0].set_xlabel('radius 1')
        axes[0].set_ylabel('radius 2')
        axes[0].set_xlim(s.rmin,s.rmax)
        axes[0].set_ylim(s.rmax,s.rmin)

        #plot cvmatrix diagonal (expected power spectrum)
        hpower = s.get_hpower()*s.ls.reshape(s.nl,1)
        lgrid,rgrid = np.meshgrid(s.ls,s.radii,indexing='ij')
        mesh = axes[1].pcolormesh(lgrid,rgrid,hpower,shading='gouraud')
        mesh.set_rasterized(True)
        del lgrid,rgrid
        axes[1].set_ylabel('radius')
        axes[1].set_xlabel('degree l')
        if loglog:
            axes[1].set_xscale('log',basex=2)
            axes[1].set_yscale('log',basex=2)
            axes[1].set_xlim(1,s.lmax)
            axes[1].set_ylim(s.rmax,0.01)
        else:
            axes[1].set_xlim(0,s.lmax)
            axes[1].set_ylim(s.rmax,0)

        fig.tight_layout(pad=0.1)
        #labels = ['a)','b)']
        #label_axes(fig,axes,labels)

        #plot total power (expected model variance) at each depth
        if totpower:
            fig,ax = plt.subplots(1,1)
            ax.set_title('total power (expected model variance)')
            hpower = np.sum(np.diagonal(s.cvmatrix,axis1=1,axis2=2)*(2*s.ls[:,None]+1),axis=0)
            ax.plot(s.radii,hpower)
            ax.set_xlabel('radius')
            ax.set_ylabel('total power')
            ax.set_ylim(0.,2.)

#==== EXECUTE SCRIPT ====
if __name__ == "__main__":
    main()
