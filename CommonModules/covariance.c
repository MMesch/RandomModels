#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int covariance(double* model, double* distances, double* covariance, int nx, int ny, int nz)
{
    int ix,iy,iz,ix1,iy1,iz1,ix2,iy2,iz2,ix3,iy3,iz3,ineigh;
    int count[nx*ny*nz];
    for(ix=0; ix<nx*ny*nz; ix++)
        count[ix] = 0;

    //neighborhoud size in points
    ineigh = 10;

    for(ix1=0; ix1<nx; ix1++){//outer loop
    printf("xlayer %d/%d\n",ix1,nx);
    for(iy1=0; iy1<ny; iy1++)
    for(iz1=0; iz1<nz; iz1++)
       for(ix2=ix1; ix2<ix1+ineigh; ix2++)//inner loop
       for(iy2=iy1; iy2<iy1+ineigh; iy2++)
       for(iz2=iz1; iz2<iz1+ineigh; iz2++)
       {   
           ix3 = ix2;
           iy3 = iy2;
           iz3 = iz2;
           if (ix2>nx-1) ix3 = ix2-nx;
           if (iy2>nx-1) iy3 = iy2-ny;
           if (iz2>nx-1) iz3 = iz2-nz;

           ix = abs(ix1-ix2);
           if (ix > nx/2) ix = ix - nx/2;
           iy = abs(iy1-iy2);
           if (iy > ny/2) iy = iy - ny/2;
           iz = abs(iz1-iz2);
           if (iz > nz/2) iz = iz - nz/2;
           distances[ix*ny*nz+iy*nz+iz]   = sqrt(ix*ix+iy*iy+iz*iz);
           covariance[ix*ny*nz+iy*nz+iz] += model[ix1*ny*nz+iy1*nz+iz1]*model[ix3*ny*nz+iy3*nz+iz3];
           count[ix*ny*nz+iy*nz+iz] += 1;
       }
    }

    for(ix=0; ix<nx*ny*nz; ix++)
        if(count[ix]>0)
           covariance[ix] /= (float)count[ix];
}
