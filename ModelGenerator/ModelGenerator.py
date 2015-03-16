#!/usr/bin/env python

import sys,os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from geographiclib.geodesic import Geodesic
from scipy.ndimage import map_coordinates
from scipy.special import sph_jn
from scipy.linalg import eigh
from scipy import interpolate
sys.path.append('/home/matthias/projects/python/modules/spherical_section')
from section import section_factory
sys.path.append('/home/matthias/projects/gitprojects/SHTOOLS_dev/SHTOOLS')
import pyshtools as shtools
from seismoclasses import xcosTaper

#custom modules
sys.path.append(os.path.join(os.path.dirname(__file__),'../CommonModules'))
from FigStyle import style_gji
from CovarianceFunctions import power_gaussian3d
matplotlib.rcParams.update(style_gji)
sys.path.append(os.path.join(os.path.dirname(__file__),'../Spherical'))
from CVMatrix import LayeredCovarianceMatrix

#!/usr/bin/env python
class ModelGenerator3D:
    def __init__(s, nlat, dr_lowermantle, dr_uppermantle):
        """
        initializes a random model with white spectrum and Gaussian
        distribution function. The model is parametrized horizontally
        in spherical harmonics and radially on layers.

        nlat: horizontal resolution (e.g. 360) should be even number
        nr:   radial resolution     (e.g. 60)  should be even number
        """

        print '\n---- initializing 3d random model generator ----'
        #initialize parameters
        depth_cmb = 2885.
        depth_tz  = 660.
        depth_moho = 25.
        s.rearth = 6371.
        s.rcmb = s.rearth - depth_cmb
        s.rtz   = s.rearth - depth_tz
        s.rmoho = s.rearth - depth_moho
        nr_lowermantle = np.ceil((s.rtz-dr_lowermantle-s.rcmb)/dr_lowermantle)
        nr_uppermantle = np.ceil((s.rmoho-s.rtz)/dr_uppermantle)

        s.nlat = nlat
        s.nlon = 2*s.nlat
        s.lmax = s.nlat/2-1
        s.interval = np.pi*s.rearth/s.lmax
        s.ncoeffs = 2*(s.lmax+1)*(s.lmax+1)

        print '---- creating the grid ----'
        interval = np.pi/s.nlat
        thetas   = np.linspace(0.+interval/2.,    np.pi - interval/2., s.nlat)
        phis     = np.linspace(0.+interval/2., 2.*np.pi - interval/2., s.nlon)
        radiis_lm   = np.linspace(s.rcmb,s.rtz-dr_lowermantle,nr_lowermantle)
        radiis_um   = np.linspace(s.rtz,s.rmoho,nr_uppermantle)
        radiis = np.append(radiis_lm,radiis_um)
        s.ls_cutoff = radiis/s.rearth * s.lmax #even resolution throughout the mantle
        s.ls_cutoff[s.ls_cutoff>s.lmax-1] = s.lmax-1

        s.nr = len(radiis)
        print 'number of radial layers:',s.nr
        s.ntot = s.nlat * s.nlon * s.nr

        print 'maximum distance in angular direction is: {0}'.format(interval*6371)
        print 'radial resolution is: {0} km'.format(radiis[1] - radiis[0])

        s.rgrid = radiis
        s.tgrid = thetas
        s.pgrid = phis

        print '---- filling grid with white spectrum ----'
        s.point_data = np.empty( (s.nr, s.nlat, s.nlon) )
        s.coeff_data = np.zeros( (s.nr, 2, s.lmax+1, s.lmax+1) )
        s.power_spectrum = np.empty( (s.nr,s.lmax+1) )

        s.update_points()
        s.update_power()
        s.power_interpolator  = interpolate.interp1d(s.rgrid,s.power_spectrum,axis=0)
        s.coeffs_interpolator = interpolate.interp1d(s.rgrid,s.coeff_data,axis=0)
        print '---- model initialized ----'

    def update_power(s):
        for ir in range(s.nr):
            coeffs = s.coeff_data[ir]
            s.power_spectrum[ir] = shtools.SHPowerSpectrum(coeffs)

    def update_coeffs(s):
        """update harmonic coeffs from the point data"""
        for ir in range(s.nr):
            s.coeff_data[ir] = shtools.SHExpandDH(s.point_data[ir],sampling=2)
            s.power_spectrum[ir] = shtools.SHPowerSpectrum(s.coeff_data[ir])
        s.coeffs_interpolator = interpolate.interp1d(s.rgrid,s.coeff_data,axis=0)
        s.power_interpolator  = interpolate.interp1d(s.rgrid,s.power_spectrum,axis=0)

    def update_points(s):
        """update point data from harmonic coefficients"""
        for ir in range(s.nr):
            s.point_data[ir] = shtools.MakeGridDH(s.coeff_data[ir],sampling=2)

    def filter(s,tfunction):
        assert len(tfunction)==s.lmax+1
        s.coeff_data*=tfunction.reshape(1,1,s.lmax+1,1)
        s.coeffs_interpolator = interpolate.interp1d(s.rgrid,s.coeff_data,axis=0)
        s.update_points()

    def plot(s, depth, model=None, ax=None, vrange=None, lon0=0., colormap=None, projection='moll', label=None):
        if model is None:
            coeffs = s.get_coeffs(depth)
            grid = shtools.MakeGridDH(coeffs,sampling=2)
        else:
            coeffs = model.get_coeffs(depth)
            grid = shtools.MakeGridDH(coeffs,sampling=2)
        print 'plotting depth: %dkm'%s.rgrid[-5]
        if vrange is None:
            lim = np.max(np.abs(grid))
            vrange = (-lim,lim)
        nlat, nlon = grid.shape
        if colormap is None:
            colormap = matplotlib.cm.Spectral
        interval = 180./nlat
        latsin = np.arange(-90.+interval/2.,90.0,interval)
        lonsin = np.arange(0.+interval/2.,360.,interval)
        norm = matplotlib.colors.Normalize(vmin=vrange[0], vmax=vrange[1]) 
        if ax == None:
            fig = plt.figure()
            ax  = plt.subplot(111)
        else:
            fig = plt.gcf()
            plt.sca(ax)
        grid_rot,lons = shiftgrid(lon0+180.001,grid,lonsin,start=False)
        if projection=='moll':
            m = Basemap(lon_0 = lon0,resolution='c',area_thresh=10000.,projection='moll')
            x,y = m(*np.meshgrid(lons,latsin))
            p = m.pcolormesh(x,y,grid_rot[::-1,:],norm=norm,cmap=colormap)
            p.set_rasterized(True)
            cbar = m.colorbar(p)
            m.drawcoastlines()
            m.drawmapboundary()
        elif projection=='robin':
            m = Basemap(lon_0 = lon0,resolution='c',projection='robin')
            x,y = m(*np.meshgrid(lons,latsin))
            p = m.pcolormesh(x,y,grid_rot[::-1,:],norm=norm,cmap=colormap)
            p.set_rasterized(True)
            cbar = m.colorbar(p)
            m.drawmapboundary()
            m.drawcoastlines()
        cbar.set_label(r'Amplitude')
        plt.title(label)

    def random_mask(s,scale):
        print '---- multiplying with random Gaussian mask to sparsify ----'
        ls = np.arange(s.lmax+1)
        nrhos = int((s.lmax+1)/2.)
        rhos  = np.arange(nrhos)
        cvmatrix = LayeredCovarianceMatrix(ls,s.rgrid)
        power   = power_gaussian3d(2*np.pi*rhos,scale)
        cvmatrix.fill_isospherical(rhos,power)

        mask_grid   = np.zeros_like(s.point_data)
        mask_coeffs = np.zeros_like(s.coeff_data)
        mask_power  = np.zeros_like(s.power_spectrum)

        for l in ls:
            #multiply with Karhunen-Loeve basis
            w,E = cvmatrix.get_basis(l,thresh=1e-4)
            nbasis = len(w)
            L = np.sqrt(w).reshape(1,nbasis)*E

            #create uncorrelated random vectors
            alm = np.random.normal(loc=0.,scale=1.,size=(l+1)*nbasis).reshape(nbasis,l+1)
            blm = np.random.normal(loc=0.,scale=1.,size=(l+1)*nbasis).reshape(nbasis,l+1)
            alm = np.dot(L,alm)
            blm = np.dot(L,blm)

            #update coefficients
            mask_coeffs[:,0,l,:l+1] = np.real(alm)
            mask_coeffs[:,1,l,:l+1] = np.real(blm)
        mask_coeffs[:,:,0,:] = 0.

        for ir in range(s.nr):
            mask_grid[ir]  = shtools.MakeGridDH(mask_coeffs[ir],sampling=2)**2
            mask_power[ir] = shtools.SHPowerSpectrum(mask_coeffs[ir])

        for ir in range(s.nr):
            s.point_data[ir]     *= mask_grid[ir]/np.sum(mask_power[ir])
            s.coeff_data[ir]     = shtools.SHExpandDH(s.point_data[ir])
            s.power_spectrum[ir] = shtools.SHPowerSpectrum(s.coeff_data[ir])

        s.power_interpolator  = interpolate.interp1d(s.rgrid,s.power_spectrum,axis=0)
        s.coeffs_interpolator = interpolate.interp1d(s.rgrid,s.coeff_data,axis=0)

        fig,ax = plt.subplots(1,1)
        mask_power = mask_power*ls*np.log(2)
        minval, maxval = mask_power.max()*1e-3, mask_power.max()*2.
        norm = matplotlib.colors.LogNorm(minval, maxval)
        ax.imshow(mask_power,norm=norm)
        ax.set_xlim(2**0,s.lmax)

    def correlate_layers(s,layer_file):
        print '---- filling with layered random model ----'
        ls = np.arange(s.lmax+1)
        cvmatrix = LayeredCovarianceMatrix(ls,s.rgrid)
        cvmatrix.fillfromfile(layer_file)

        for l in ls:

            #multiply with Karhunen-Loeve basis
            w,E = cvmatrix.get_basis(l,thresh=1e-4)
            nbasis = len(w)
            L = np.sqrt(w).reshape(1,nbasis)*E

            #create uncorrelated random vectors
            alm = np.random.normal(loc=0.,scale=1.,size=(l+1)*nbasis).reshape(nbasis,l+1)
            blm = np.random.normal(loc=0.,scale=1.,size=(l+1)*nbasis).reshape(nbasis,l+1)
            alm = np.dot(L,alm)
            blm = np.dot(L,blm)

            #update coefficients
            s.coeff_data[:,0,l,:l+1] = np.real(alm)
            s.coeff_data[:,1,l,:l+1] = np.real(blm)
        s.coeff_data[:,:,0,:] = 0.

        for ir in range(s.nr):
            s.point_data[ir]     = shtools.MakeGridDH(s.coeff_data[ir],sampling=2)
            s.power_spectrum[ir] = shtools.SHPowerSpectrum(s.coeff_data[ir])

        s.power_interpolator  = interpolate.interp1d(s.rgrid,s.power_spectrum,axis=0)
        s.coeffs_interpolator = interpolate.interp1d(s.rgrid,s.coeff_data,axis=0)

    def get_points(s, points):
        rs = s.rgrid
        lats = np.degrees(s.tgrid)
        lons = np.degrees(s.pgrid)
        points[2,points[2]<0.] += 360.
        points[2,points[2]>360.] -= 360.

        ipoints = np.empty_like(points)
        ipoints[0,:] = np.interp(points[0,:],rs,np.arange(s.nr),left=-1,right=-1)
        ipoints[1,:] = np.interp(points[1,:],lats,np.arange(s.nlat),left=-1,right=-1)
        ipoints[2,:] = np.interp(points[2,:],lons,np.arange(s.nlon),left=-1,right=-1)
        #points3d shape: points3d.reshape(nradii,npoints,3)
        values = map_coordinates(s.point_data, ipoints,cval=np.NaN)
        return values

    def plot_profile(s, lat1,lon1, lat2,lon2, ax=None, model=None, colormap=None, vrange=None):
        if lon1>180.:lon1-=360.
        if lon2>180.:lon2-=360.
        mindepth = 25.
        maxdepth = 2885.
        #---- create geodesic ----
        npoints = 150
        ndepths = 150
        g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
        #>> g.viewkeys() ['lat1', 'a12', 's12', 'lat2', 'azi2', 'azi1', 'lon1', 'lon2']
        #>> dict_keys(['lat1', 'a12', 's12', 'lat2', 'azi2', 'azi1', 'lon1', 'lon2'])
        distance = float(g['s12'])
        print 'geodesic has length: %5.2fkm'%(distance/1000.)
        line = Geodesic.WGS84.Line(lat1, lon1, g['azi1'])
        latlons = np.array([(line.Position(dx)['lat2'],line.Position(dx)['lon2'])
                                            for dx in np.linspace(0.,distance,npoints)])
        depths = np.linspace(maxdepth,mindepth,ndepths)
        points = []
        for depth in depths:
            for lat,lon in latlons:
                points.append((6371.-depth,lat,lon))
        points = np.transpose(np.array(points))

        print 'creating vertical cut: % 3.2f, % 3.2f -> % 3.2f, % 3.2f'%(lat1,lon1,lat2,lon2)
        if model is None:
            values = s.get_points(points)
        else:
            values = model.get_points(points)
        values = np.reshape(values,(ndepths,npoints))
        thetas = np.linspace(0.,distance*1e-3/6371.,npoints)
        radii  = 6371. - depths
        ts,rs = np.meshgrid(thetas,radii)
        
        def forceAspect(ax,aspect=1):
            im = ax.get_images()
            extent =  im[0].get_extent()
            ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

        if vrange is None:
            lim = np.max(np.abs(values))
            vrange = -lim,lim
        if colormap is None:
            colormap = matplotlib.cm.Spectral
        norm = matplotlib.colors.Normalize(vmin=vrange[0], vmax=vrange[1]) 

        #fig = plt.figure(figsize=(lat_textwidth_in,lat_textwidth_in))
        if ax is None:
            fig = plt.figure()
            nrows,ncols,nplot = 1,1,1
        else:
            nrows,ncols,nplot = ax
            fig = plt.gcf()
           
        rticks = (6371.-np.array([depths[0],660.,220.,depths[-1]]))
        degrees = np.degrees(thetas[-1]-thetas[0])
        width   = (rticks[-1]-rticks[0])/rticks[-1]
        tticks = np.linspace(0.,degrees,5.)
        section_factory(degrees,width,rticks[1:-1],tticks)
        ax=fig.add_subplot(nrows,ncols,nplot,projection='spherical_section')

        ax.set_yticks(rticks[1:-1])
        #ax.set_yticklabels(['660','220'])
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        cm = ax.pcolormesh(ts,rs,values,cmap=colormap,norm=norm)
        cm.set_rasterized(True)
        plt.tick_params(labelsize=9)
        ax.set_ylim(0.,6371.-depths[-1])
        ax.grid(True)
        fig.tight_layout(pad=0.1)
        #bar = plt.colorbar(cm,shrink=.7)
        #bar.set_label(r'$dvs/vs_0$')

    def histogram(s, vrange, label=None, plot=True,plot_iso=True):
        data = s.point_data[0]
        if vrange is None:
            lim = np.abs(data).max()
            vrange = -lim, lim
        nlat,nlon = data.shape
        size = (5,3.5)
        interval = 180.0/nlat
        latsin = np.arange(-90.+interval/2., 90.,interval)
        lonsin = np.arange(0.+interval/2.,360.,interval)
        nbins = 31
        lons,lats = np.meshgrid(lonsin,latsin)
        weights = np.sin( np.radians(90.-lats.flatten()) )
        hist, bins = np.histogram(data.flatten(), bins=nbins, range=vrange,weights=weights,density=True)
        dbin = bins[1] - bins[0]
        #ist*= dbin
        width = 0.8*(bins[1]-bins[0])
        center = (bins[:-1]+bins[1:])/2.
        if plot:
            if plot_iso:
                sigma = s.lmax+1
                x = np.linspace(-10.*sigma,+10.*sigma,1000)
                def gaussian(pos,loc,sigma):
                    return np.exp(-(x-loc)**2/(2.*sigma**2))/np.sqrt(2.*np.pi*sigma**2)
                iso_dist = gaussian(x,0.,s.lmax+1)
            print np.sum(hist)
            fig = plt.figure(figsize=size)
            ax = fig.add_subplot(111)
            ax.bar(center, hist, align = 'center', width = width)
            ax.plot(x,iso_dist,c='red')
            ax.set_title(label)
            ax.set_xlim(*vrange)
            ax.set_xlabel(r'$dv_s$ shear wave velocity perturbation')
            ax.set_ylabel(r'probability density')
        return center, hist
    
    def point_data_interpolator(s,radius):
        coeffs = s.coeffs_interpolator(radius)
        power_old = shtools.SHPowerSpectrum(coeffs)
        power_new  = s.power_interpolator(radius)
        coeffs *= np.sqrt(power_new/(power_old+1e-10)).reshape(1,s.lmax+1,1)
        grid = shtools.MakeGridDH(coeffs,sampling=2)
        return grid 

    #---------- returns coefficients at a certain depth ----------
    def get_coeffs(s,depth):
        radius = 6371. - depth
        grid = s.point_data_interpolator(radius)
        coeffs = shtools.SHExpandDH(grid,sampling=2)
        return coeffs

    def get_power(s,depth):
        radius = 6371. - depth
        grid = s.point_data_interpolator(radius)
        coeffs = shtools.SHExpandDH(grid,sampling=2)
        return coeffs

    #======= OUTPUT ========
    def writecoefffile(s,fname,modelname='noname'):
        #write header
        print 'writing coefficient file %s'%fname
        outfile = open(fname,'w')
        header = 'nlayer lmax rmin rmax\n%d %d %.2f %.2f %s\n'%(s.nr, s.lmax, s.rgrid[0], s.rgrid[-1],modelname)
        outfile.write(header)
        for ir,l_cutoff in enumerate(s.ls_cutoff):
            coeffs = s.coeff_data[ir]*np.sqrt(4.*np.pi) #remove 4pi normalization
            coeffs[:,:,0]*=np.sqrt(2) #for complex spherical harmonics routine
            radius  = s.rgrid[ir]
            outfile.write('%d %d\n'%(radius,int(l_cutoff)))
            for il in range(int(l_cutoff)+1):
                al0 = coeffs[0,il,0]
                outfile.write(' % 6.4E'%al0)
                for im in range(1,il+1):
                    norm = (-1)**im  #renormalize coeffs as s40 for specfem routine
                    alm = coeffs[0,il,im]*norm
                    blm = coeffs[1,il,im]*norm
                    outfile.write(' % 6.4E % 6.4E'%(alm,blm))
                outfile.write('\n')
        outfile.close()

    def writevtkfile(s,fname,method='evtk'):
        """
        writes output to vtk files. Data is ordered as:
        starting from the deepest layer, north pole, goes in rings to the south pole,
        than to the upper layers. We should create a local2global array!
        """
        xgrid = s.rgrid*np.sin(s.tgrid)*np.cos(s.pgrid)
        ygrid = s.rgrid*np.sin(s.tgrid)*np.sin(s.pgrid)
        zgrid = s.rgrid*np.cos(s.tgrid)
        if method == 'vtk':
            import vtk
            print '---- creating binary vtk file using the vtk library ----'
            vtkpoints = vtk.vtkPoints()
            vtkpoints.SetNumberOfPoints(s.ntot)
            vtkdata = vtk.vtkFloatArray()
            vtkdata.SetNumberOfComponents(1)
            vtkdata.SetNumberOfTuples(s.ntot)
            vtkdata.SetName('random data')
        
            #create points. Save their position in the global array
            iglob = 0
            for ir in range(s.nr):
                for ilat in range(s.nlat):
                    for ilon in range(s.nlon):
                        x = xgrid[ir, ilat, ilon]
                        y = ygrid[ir, ilat, ilon]
                        z = zgrid[ir, ilat, ilon]
                        val   = s.point_data[ir,ilat,ilon]
                        vtkpoints.InsertPoint(iglob,x,y,z)
                        vtkdata.SetTuple1(iglob,val)
                        iglob += 1
        
            StructuredGrid = vtk.vtkStructuredGrid()
            StructuredGrid.SetDimensions(s.nlon, s.nlat, s.nr)
            StructuredGrid.SetPoints(vtkpoints)
            StructuredGrid.GetPointData().AddArray(vtkdata)
        
            writer = vtk.vtkStructuredGridWriter()
            writer.SetInputConnection(StructuredGrid.GetProducerPort())
            writer.SetFileTypeToBinary()
            writer.SetFileName(fname)
            writer.Write()
        elif method == 'evtk':
            from evtk.hl import gridToVTK
            print 'writing output'
            gridToVTK(fname,xgrid,ygrid,zgrid,pointData= {'random field': s.point_data})

    def insert_real_model(s,target_model,plot=False,save=False,model_only=False,gradient=False):
        """
        This function inserts a "real" tomographic model in the low degrees with a smooth
        transition to the random model in the higher degrees.
        """
        #some parameters
        s.radial_nodes = target_model.radial_nodes
        ls = np.arange(0.,s.lmax+1)
        coeffs_rand = np.zeros((2,s.lmax+1,s.lmax+1))
        coeffs_model = np.zeros( (2,s.lmax+1,s.lmax+1) )
        lmax_target = target_model.lmax
        crossover_l = 40
        crossover_dl = 10
        crossover_model_l = 60
        crossover_model_dl = 10

        print 'loop over layers and insert original coefficients:'
        for ir in range(s.nr):
            depth = 6371. - s.rgrid[ir]
            if depth < 660:
                crossover_l = 18
                crossover_dl = 10
                crossover_model_l = 70
                crossover_model_dl =10
            else:
                crossover_l = 15
                crossover_dl = 3
                crossover_model_l = 40
                crossover_model_dl = 3
            coeffs_rand = s.coeff_data[ir]
            coeffs_model[:,:,:] = 0.

            ltrim = min(s.lmax+1,lmax_target+1)
            coeffs_model[:,:ltrim,:ltrim] = target_model.get_coeffs(depth)[:,:ltrim,:ltrim]
            if gradient:
                grid_model = shtools.MakeGridDH(coeffs_model[:,:45,:45],lmax=s.lmax,sampling=2)
                grad_model = get_gradient(grid_model)
                interval = 180./s.nlat
                weights = np.cos(np.radians(np.linspace(-90.+interval/2.,90.-interval/2.,s.nlat)))
                norm = np.sum(grad_model**2*weights.reshape(s.nlat,1))/(weights.sum()*s.nlon)
                grad_model /= np.sqrt(norm)
                #filt = xcosTaper(np.arange(s.lmax+1),(10.,20.,70,200))
                #grid_rand = shtools.MakeGridDH(coeffs_rand*filt.reshape(1,s.lmax+1,1),sampling=2)*grad_model
                grid_rand = shtools.MakeGridDH(coeffs_rand,sampling=2)*grad_model
                coeffs_rand = shtools.SHExpandDH(grid_rand,sampling=2)

            if model_only:
                crossover_model = 0.0
                crossover = 0.0
            else:
                crossover = (np.tanh((ls - crossover_l)/float(crossover_dl) * np.pi/2.0)+1.0)/2.0
                crossover = crossover.reshape(1,s.lmax+1,1)
                crossover_model = (np.tanh((ls - crossover_model_l)/float(crossover_model_dl) * np.pi/2.0)+1.0)/2.0
                crossover_model = crossover_model.reshape(1,s.lmax+1,1)

            original_power = shtools.SHPowerSpectrum(coeffs_model)
            if gradient:
                crossover = 1.
            else:
                coeffs_model *= (1. - crossover_model)

            rand_power = shtools.SHPowerSpectrum(coeffs_rand)
            real_power = shtools.SHPowerSpectrum(coeffs_model)
            diff_power = (real_power/(rand_power+1e-19)).reshape(1,s.lmax+1,1)
            mask = np.array(np.ones_like(coeffs_rand) * (diff_power < 1.),np.bool)
            coeffs_combined = coeffs_model
            if gradient:
                coeffs_combined = coeffs_model + coeffs_rand
            else:
                coeffs_combined[mask]  += (np.sqrt(np.abs(1.-diff_power))*crossover*coeffs_rand)[mask]
            s.coeff_data[ir,:,:,:] = coeffs_combined
            s.point_data[ir,:,:] = shtools.MakeGridDH(coeffs_combined,lmax=s.lmax,sampling=2)

            if plot or save:
                fig = plt.figure(figsize = (5,5) )
                ax = fig.add_subplot(111)
                combined_power = shtools.SHPowerSpectrum(coeffs_combined) 
                ax.plot(ls,ls*rand_power,'-',label='random')
                ax.plot(ls,ls*original_power,'-',label='model')
                ax.plot(ls,ls*combined_power,lw=2,alpha=0.6,label='combined')
                ax.set_xscale('log',basex=2)
                ax.set_yscale('log',basey=2)
                ax.grid(True)
                ax.set_title('Harmonic Power Spectrum at depth {:.0f}km'.format(depth))
                ax.set_xlabel(r'harmonic degree l')
                ax.set_ylabel(r'power')
                ax.legend(loc=3)
                ax.set_ylim( (ls*original_power).max()*2**-4,(ls*original_power).max()*2**3)
                if save:
                    fig.savefig('figures/spectrum_{:04.0f}km.svg'.format(depth))
                    plt.close(fig)
        s.update_power()
        s.power_interpolator  = interpolate.interp1d(s.rgrid,s.power_spectrum,axis=0)
        s.coeffs_interpolator = interpolate.interp1d(s.rgrid,s.coeff_data,axis=0)

    def plot_harmonic_spectrum(s,ax=None, model=None, nr_image=200, peroctave=True, logx=True, 
                                 cbar=True, cutoff=False):
        """
        plots the harmonic spectrum for all layers as colormap
        """
        if ax == None:
            fig = plt.figure()
            ax  = plt.subplot(111)
        else:
            fig = plt.gcf()

        #get power spectrum at all depths
        power = np.zeros( (nr_image,s.lmax+1) )
        depths = np.linspace(50.,2885.,nr_image)
        for idep in range(nr_image):
            depth = depths[idep]
            if model:
                coeffs = np.zeros( (2,s.lmax+1,s.lmax+1) )
                coeffs_model = model.get_coeffs(depth)
                lmax_trim = coeffs_model.shape[1]
                coeffs[:,:lmax_trim,:lmax_trim] = coeffs_model
            else:
                coeffs = s.get_coeffs(depth)
            power[idep] = shtools.SHPowerSpectrum(coeffs)

        #plot power_spectrum
        ls = np.arange(s.lmax+1)
        if peroctave:
            power = power*ls*np.log(2)
            minval, maxval = 1e-6, 1e-3
            norm = matplotlib.colors.LogNorm(minval, maxval)
            levels = np.logspace(np.log10(minval),np.log10(maxval),16)
            cblabel = 'power per octave'
            lgrid, depgrid = np.meshgrid(ls[1:],depths)
            cs = plot_matrix(ax,lgrid,depgrid,np.clip(power,minval,maxval),levels,norm)
        else:
            minval, maxval = 1e-7, 1e-4
            norm = matplotlib.colors.LogNorm(minval, maxval, clip=True)
            levels = np.logspace(np.log10(minval),np.log10(maxval),32)
            cblabel = 'power per degree'
            cs=ax.imshow(np.clip(power,minval,maxval),norm=norm,interpolation='nearest',
                             aspect='auto',extent=(-0.5,s.lmax-0.5,s.depths[-1],s.depths[0]))
        if logx: ax.set_xscale('log', basex=2)

        formatter = matplotlib.ticker.FormatStrFormatter('%d')
        ax.xaxis.set_major_formatter(formatter)
        if peroctave:
            ax.set_xlim(2**0,s.lmax)
        else:
            ax.set_xlim(0.,s.lmax)

        ylims = np.array( [depths[-1],depths[0]] )
        ax.set_ylim(*tuple(ylims))
        if cutoff:
            ax.plot(s.ls_cutoff,6371.-s.rgrid,lw=2,c='black')
        #vertical lines for kilometer resolution
        def km2deg(km,dep):
            return (6371.-dep)*2.*np.pi/km/2.
        #ax.plot(km2deg(1000.,s.depths),s.depths,c='black')
        #make good looking colorbar
        #fig.tight_layout(pad=0.1)
        if cbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", "5%", pad="3%")
            box = ax.get_position()
            #cax = fig.add_axes([box.x1+0.1*box.width, box.y0, box.width*0.05, box.height])
            nlevels = len(levels)
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
            cb.set_label(cblabel)

def get_gradient(grid):
    nlat,nlon = grid.shape
    interval = 180./nlat
    lats = np.linspace(-90.+interval/2.,90.+interval/2.,nlat)
    pgradient = np.zeros( (nlat,nlon) )
    tgradient = np.zeros( (nlat,nlon) )
    for ilat in range(nlat):
        pgradient[ilat,:] = np.gradient(grid[ilat,:])/np.cos(np.radians(lats[ilat]))
    for ilon in range(nlon):
        tgradient[:,ilon] = np.gradient(grid[:,ilon])
    return np.sqrt(tgradient**2 + pgradient**2)

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

def plot_matrix(ax, lgrid, depgrid, matrix, levels, norm):
    cs = ax.contourf(lgrid,depgrid, matrix[:,1:], levels, norm=norm)
    ax.set_aspect(30)
    insert(cs,ax)
    return cs

