#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import sys,os

#custom modules
from ModelGenerator import ModelGenerator3D
sys.path.append('/home/matthias/projects/python/tomographic_models/scripts/global_models')
from global_models import get_global_model
sys.path.append(os.path.join(os.path.dirname(__file__),'../CommonModules'))
from FigStyle import style_gji
matplotlib.rcParams.update(style_gji)

def main():
    model = ModelGenerator3D(360,40,10)
    model.correlate_spline('model_spectra/semum2_pl_scalefree.spl')

    fig1,ax1 = plt.subplots(1,3, sharey=True)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)

    ax1[0].set_ylabel('depth in km')
    for ax in ax1:
        ax.set_xlabel('degree l')

    ax21 = fig2.add_subplot(311)
    ax22 = fig2.add_subplot(312)
    ax23 = fig2.add_subplot(313)

    depth = 100.
    semum2 = get_global_model('semum2.1')
    model.plot_harmonic_spectrum(ax=ax1[0],model=semum2 ,peroctave=True, cbar=False)
    model.plot(depth, ax=ax21, vrange=(-0.1,0.1), model=semum2, lon0=40, projection='robin',label='')
    plt.figure(3)
    model.plot_profile(20.,3.,30.,160.,ax=(3,1,1), model=semum2, vrange=(-0.03,0.03))

    model.plot_harmonic_spectrum(ax=ax1[1],peroctave=True, cbar=False)
    model.plot(depth, ax=ax22, vrange=(-0.1,0.1), lon0=40, projection='robin', label='' )
    plt.figure(3)
    model.plot_profile(20.,3.,30.,160.,ax=(3,1,2), vrange=(-0.03,0.03))

    model.insert_real_model(semum2,save=True, model_only=False, gradient=False)
    model.plot_harmonic_spectrum(ax=ax1[2],peroctave=True,cbar=False)
    model.plot(depth, ax=ax23, vrange=(-0.1,0.1), lon0=40, projection='robin', label='' )
    plt.figure(3)
    model.plot_profile(20.,3.,30.,160.,ax=(3,1,3), vrange=(-0.03,0.03))

    fig1.tight_layout(pad=0.1)
    fig2.tight_layout(pad=0.1)
    fig3.tight_layout(pad=0.1)
    fig3.subplots_adjust(left=-1.5,right=2.5,top=1.3,bottom=-0.1,hspace=-0.6)

    model.writecoefffile('random_coeffs_sf1.sph')

    model.histogram(vrange = None, label='Gaussian random model' )
    model.plot_profile(20.,3.,30.,160.,vrange=(-0.03,0.03))
    #model.writevtkfile('out')

    plt.show()

if __name__ == "__main__":
    main()
