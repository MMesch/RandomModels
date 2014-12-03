#!/usr/bin/env python
"""
plots a model spectrum
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
sys.path.append('/home/matthias/projects/python/tomographic_models/scripts/global_models')
from configs import lat_columnwidth_in
from global_models import get_global_model
from scipy.ndimage import map_coordinates
sys.path.append('/home/matthias/projects/gitprojects/SHTOOLS_dev/SHTOOLS')
from pyshtools import PLegendre

class spline_model():
    """
    This is a random model which is based on horizontal spectra that are parametrized via
    a cubic B-spline formulation
    """
    def __init__(s,fname,lmax=360):
        """
        takes a name of the file with horizontal and radial spline coefficients as input
        """
        #read input file
        infile = open(fname)
        lines = infile.readlines()
        iline = 1
        s.depths = np.fromstring(lines[iline], sep=' ')
        s.ndepths = len(s.depths)
        iline += 2
        s.lknots = np.fromstring(lines[iline], sep=' ')
        s.nlknots = len(s.lknots)
        iline += 2
        s.values = np.fromstring(' '.join(lines[iline:iline+s.ndepths]),sep=' ').reshape(s.ndepths,s.nlknots)
        s.values = np.transpose(s.values)
        iline += s.ndepths+1
        s.rfunc = lines[iline].strip()
        iline += 2
        s.rcorr = np.array([float(line.strip()) for line in lines[iline:iline+s.ndepths]])
        infile.close()
        #compute horizontal correlation functions at spline knots

        s.nphi_hcorrs = 256
        s.lmax_hcorrs = 150
        s.phis_hcorrs = np.linspace(0.,np.pi,s.nphi_hcorrs)
        s.hcorrs = np.zeros( (s.ndepths,s.nphi_hcorrs) )
        for idepth,depth in enumerate(s.depths):
            s.hcorrs[idepth] = s.get_hcorr(depth,s.phis_hcorrs,s.lmax_hcorrs)

    def get_hcorr(s,depth,phis,lmax):
        nphi = len(phis)
        power = s.get_power(depth,lreturn=lmax)
        hcorr = np.zeros(nphi)
        for iphi,cosphi in enumerate(np.cos(phis)):
            hcorr[iphi] = np.sum(PLegendre(lmax-1,cosphi)*power)
        hcorr/=np.sum(power)
        return hcorr

    def get_rcorr(s,depth,degree_max,rdists):
        nphi = 512
        lmax = 150
        phis = np.linspace(0.,np.pi,nphi)
        dphi = phis[1]-phis[0]
        hcorr = s.get_hcorr(depth,phis,lmax)
        power = s.get_power(depth,lreturn=lmax)

        #return radial correlations for all distances and degrees...
        nrdists = len(rdists)
        rcorr = np.zeros( (nrdists,degree_max) )
        for idist,rdist in enumerate(rdists):
            phis_new  = np.sqrt(rdist**2+phis**2)
            hcorr_new = np.interp(phis_new,phis,hcorr,right=0.)
            for iphi,phi in enumerate(phis):
                rcorr[idist] += PLegendre(degree_max-1,np.cos(phi))*hcorr_new[iphi]*np.sin(phi)
            rcorr[idist] *= (2*np.arange(degree_max)+1)/2.*dphi*np.sum(power)/(power+power.max()*1e-1)
        return rcorr

    def plot_rcorrs(s,depth,rdists):
        plt.figure()
        lmax = 150
        rcorr = s.get_rcorr(depth,lmax,rdists)
        plt.plot(rcorr[0])

    def plot_hcorrs(s):
        plt.figure()
        for idep in range(s.ndepths):
            plt.plot(s.hcorrs[idep])

    def get_power(s,depth,lreturn=150):
        ls = np.arange(lreturn)
        ideps = np.interp(depth,s.depths,np.arange(s.ndepths))
        ils = np.interp(ls,s.lknots,np.arange(s.nlknots))
        lls,dds = np.meshgrid( ils,ideps )
        nls,ndep = lls.shape
        points2d = np.vstack( (lls.flatten(),dds.flatten()) )
        power = map_coordinates(np.log(s.values),points2d,order=1).reshape(nls,ndep)
        power = np.exp(power)
        if power.shape[0] == 1:
            power = power.flatten()
        power[1:]=power[1:]/np.log(2)/ls[1:]
        return power

    def radial_correlation(s,degreel,depth1,depth2):
        middepth = 0.5*(depth1+depth2)
        dr = np.abs(depth2 - depth1)
        if s.rfunc == 'exponential':
            radius = (6371.-middepth)
            scalel = 2.*np.pi/float(degreel)
            #scaler = np.pi/np.interp(middepth,s.depths,s.rcorr)
            scale_new = radius*scalel
            return np.exp(-dr/scale_new)*(1.+(np.abs(dr)/scale_new))
        elif s.rfunc == 'gaussian':
            scale_max = np.interp(middepth,s.depths,s.rcorr)
            scale = scale_max
            return np.exp(-dr**2/scale**2)

    def plot_spectrum(s):
        #visualize
        ndep = 400
        dep = np.linspace(0.,3000.,ndep)
        power = s.get_power(dep,lreturn=300)
        plt.figure()
        plt.imshow(power)
        plt.show()

#==== MAIN FUNCTION ====
def main():
    ndepths = 200
    depths = np.linspace(50.,2850.,ndepths)
    lreturn =150
    ls_art = np.arange(lreturn)
    log2 = np.log(2)

    #reference model
    model_ref = get_global_model('semum2.1')
    #artificial model
    model_art = spline_model(sys.argv[-1])

    #---- plot horizontal correlation function ----
    model_art.plot_hcorrs()
    model_art.plot_rcorrs(200.,np.linspace(0.,0.1,200))

    #---- get and plot power and single depth ----
    depth = 700
    power_ref = model_ref.get_power(depth)
    ls_ref = np.arange(len(power_ref))
    power_art = model_art.get_power(depth,lreturn=lreturn)

    fig,ax = plt.subplots(1)
    ppo_ref = log2*ls_ref*power_ref
    ppo_art = log2*ls_art*power_art
    maximum = ppo_ref.max()
    ax.plot(ls_ref,ppo_ref)
    ax.plot(ls_art,ppo_art)
    ax.set_xscale('log',basex=2)
    ax.set_yscale('log',basey=2)
    ax.grid(True,which='major')
    ax.set_ylim(maximum*2**-8,maximum*2)
    formattery = matplotlib.ticker.FormatStrFormatter('%2.2e')
    ax.yaxis.set_major_formatter(formattery)

    #---- get and plot power over the whole mantle ----
    ppo_art_matrix = np.zeros( (ndepths,lreturn) )
    ppo_ref_matrix = np.zeros( (ndepths,lreturn) )
    for idepth,depth in enumerate(depths):
        ref_power = model_ref.get_power(depth)
        ls_ref = np.arange(len(power_ref))
        art_power = model_art.get_power(depth,lreturn=lreturn)
        ppo_ref_matrix[idepth,:len(ref_power)] = log2*ls_ref*ref_power
        ppo_art_matrix[idepth] = log2*ls_art*art_power

    #figure and axes configuration
    fig,axes = plt.subplots(1,2,figsize=(2*lat_columnwidth_in,lat_columnwidth_in),sharey=True)
    axes[0].set_ylabel('depth x 100km')
    formatter = matplotlib.ticker.FormatStrFormatter('%d')
    for ax in axes:
        ax.set_xlabel('degree l')
        ax.set_xscale('log',basex=2)
        #adjust formatting and axis limits
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlim(2**0,90)
        ax.set_ylim(depths[-1],depths[0])

    #contour plot and rasterize (insert function hack) matrices
    minval, maxval = 1e-8, 1e-3
    norm = matplotlib.colors.LogNorm(minval, maxval, clip=True)
    nlevels = 16
    levels = np.logspace(np.log10(minval),np.log10(maxval),nlevels)
    lgrid, depgrid = np.meshgrid(ls_art[1:],depths)
    for imatrix,matrix in enumerate([ppo_ref_matrix,ppo_art_matrix]):
        cs = axes[imatrix].contourf(lgrid,depgrid, matrix[:,1:], levels, norm=norm)
        insert(cs,axes[imatrix])

    #make some space and plot colorbar on the side
    fig.tight_layout(pad=0.1,h_pad=0.7)
    plt.subplots_adjust(wspace=0.05,right=0.8)
    cax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    bounds = np.linspace(0.,1.,nlevels)
    bcmap = matplotlib.colors.ListedColormap([cs.cmap(l) for l in np.linspace(0.,1.,nlevels)])
    bnorm = matplotlib.colors.BoundaryNorm(bounds, bcmap.N)
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=bcmap,
                                           norm=bnorm,
                                           extend='both',
                                           orientation='vertical')
    nticks = 4
    cells = [nlevels/nticks for i in range(nticks)]
    for i in range(nlevels%nticks):
        cells[i] += 1
    iticks = np.append([0],np.cumsum(cells)-1)
    ticks = levels[iticks]
    tbounds = bounds[iticks]
    cb.set_ticks(tbounds)
    cb.set_ticklabels(['%1.1e'%l for l in ticks])
    cb.set_label('power per octave')
    plt.show()

#-- rasterization hack --
from matplotlib.collections import Collection
from matplotlib.artist import allow_rasterization

class ListCollection(Collection):
     def __init__(self, collections, **kwargs):
         Collection.__init__(self, **kwargs)
         self.set_collections(collections)
     def set_collections(self, collections):
         self._collections = collections
     def get_collections(self):
         return self._collections
     @allow_rasterization
     def draw(self, renderer):
         for _c in self._collections:
             _c.draw(renderer)

def insert(c,ax):
     collections = c.collections
     for _c in collections:
         _c.remove()
     cc = ListCollection(collections, rasterized=True)
     ax.add_artist(cc)
     return cc 

#==== EXECUTION ====
if __name__ == "__main__":
    main()
