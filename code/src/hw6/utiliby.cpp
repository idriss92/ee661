#include "utiliby.h"

#include <highgui.h>
#include <cv.h>

namespace CVisby
{

  double Utiliby::convolveMask(double* mask, int radius, IplImage* pImage, int pi, int pj, bool useNearest)
  {
    double res = 0;
    for (int i=-radius; i <= radius; i++)
    {
      for (int j=-radius; j <= radius; j++)
      {
        if ( pi+i >= 0  && pi+i < pImage->width && pj+j >=0 && pj+j <pImage->height )
        {
          CvScalar s = cvGet2D(pImage,pj+j,pi+i);
          res += mask[(i+radius)*(2*radius+1)+j+radius]*s.val[0];
        }
        else if (useNearest)
        {
          int tmpi =pi+i, tmpj = pj+j;
          if (pi+i < 0) tmpi = 0;
          if (pi+i >= pImage->width) tmpi = pImage->width-1;
          if (pj+j < 0) tmpj = 0;
          if (pj+j >= pImage->height) tmpj = pImage->height-1;
          CvScalar s = cvGet2D(pImage,tmpj,tmpi);
          res += mask[(i+radius)*(2*radius+1)+j+radius]*s.val[0];
        }
      }
    }
    return res;
  }

  void Utiliby::printMatrix(CvMat* mat)
  {
    for (int i=0; i<mat->rows; i++)
    {
      for (int j=0; j<mat->cols-1; j++)
      {
        printf("%.6f, ",cvmGet(mat, i, j));
      }
      printf("%.6f\n",cvmGet(mat,i,mat->cols-1));
    }
  }

  void Utiliby::printMatrix(double *m, int row, int col, int mode)
  {
    if (mode == 0)
    {
      for (int i=0; i<row; i++)
      {
        for (int j=0; j<col-1; j++)
        {
          printf("%.6f, ",m[i*col+j]);
        }
        printf("%.6f\n",m[i*col+col-1]);
      }
    }
    else
    {
      for (int i=0; i<row; i++)
      {
        for (int j=0; j<col-1; j++)
        {
          printf("%.6f, ",m[j*row+i]);
        }
        printf("%.6f\n",m[(col-1)*row+i]);
      }
    }
  }

  void Utiliby::transformImageToWorld(IplImage *pImage, 
    double *H, IplImage **returnImage, int mode, double* inRegion, double* inClip)
  {
    double *region = NULL;
    if ( inRegion == NULL )
    {
      region = new double [utilibyRecDim*utilibyCoordDim];
      for (int r=0; r<utilibyRecDim; r++)
      {
        for (int i=0; i<utilibyCoordDim-1;i++)
        {
          region[i*utilibyRecDim+r]=-1;
        }
        region[(utilibyCoordDim-1)*utilibyRecDim+r]=1;
      }
    }
    else 
      region = inRegion;
    if (region == NULL || region[0*utilibyRecDim+0]==-1)
    {
      region[0*utilibyRecDim+0]=0;
      region[1*utilibyRecDim+0]=0;
    }
    if (region == NULL || region[0*utilibyRecDim+1]==-1)
    {
      region[0*utilibyRecDim+1]=pImage->width-1;
      region[1*utilibyRecDim+1]=pImage->height-1;
    }
    if (region == NULL || region[0*utilibyRecDim+2]==-1)
    {
      region[0*utilibyRecDim+2]=0;
      region[1*utilibyRecDim+2]=pImage->height-1;
    }
    if (region == NULL || region[0*utilibyRecDim+3]==-1)
    {
      region[0*utilibyRecDim+3]=pImage->width-1;
      region[1*utilibyRecDim+3]=0;
    }

    double minWorldX = 1e10, minWorldY = 1e10, maxWorldX = -1e10, maxWorldY = -1e10;
    CvMat imageCoord;
    cvInitMatHeader(&imageCoord, utilibyCoordDim, utilibyRecDim, CV_64FC1, region);
    CvMat *pWorldCoord = cvCreateMat(utilibyCoordDim, utilibyRecDim, CV_64FC1);
    CvMat *pmH = cvCreateMat(utilibyCoordDim, utilibyCoordDim, CV_64FC1);
    CvMat *pmInvH = cvCreateMat(utilibyCoordDim, utilibyCoordDim, CV_64FC1);
    if (mode == 0)
    {
      cvInitMatHeader(pmInvH, utilibyCoordDim, utilibyCoordDim, CV_64FC1, H);
      cvInvert(pmInvH, pmH, CV_LU);
    }
    else
    {
      cvInitMatHeader(pmH, utilibyCoordDim, utilibyCoordDim, CV_64FC1, H);
      cvInvert(pmH, pmInvH, CV_LU);
    }
    cvMatMul(pmH, &imageCoord, pWorldCoord);

    //Check for the range for WorldCoord.
    for (int i=0; i<utilibyRecDim; i++)
    {
      double k = cvmGet(pWorldCoord, utilibyCoordDim-1, i);
      double worldX = cvmGet(pWorldCoord, 0, i)/k;
      double worldY = cvmGet(pWorldCoord, 1, i)/k;
      if (worldX > maxWorldX) maxWorldX = worldX;
      if (worldX < minWorldX) minWorldX = worldX;
      if (worldY > maxWorldY) maxWorldY = worldY;
      if (worldY < minWorldY) minWorldY = worldY;
    }

    if (inClip != NULL)
    {
      minWorldX = inClip[0];
      maxWorldX = inClip[1];
      minWorldY = inClip[2];
      maxWorldY = inClip[3];
    }

    double outImageWidth = pImage->width;
    double xRange = maxWorldX - minWorldX;
    double yRange = maxWorldY - minWorldY;
    double outImageHeight = outImageWidth * yRange / xRange;
    double stepSize = xRange / outImageWidth;
    worldImage = cvCreateImage(cvSize((int)outImageWidth, (int)outImageHeight), IPL_DEPTH_8U, 3);
    cvZero(worldImage);
    CvMat *piCoord = cvCreateMat(utilibyCoordDim, 1, CV_64FC1);
    CvMat *pwCoord = cvCreateMat(utilibyCoordDim, 1, CV_64FC1);
    cvmSet(pwCoord, 2, 0, 1.0);
    for (int j=0; j<worldImage->height; j++)
    {
      cvmSet(pwCoord, 1, 0, minWorldY + j*stepSize);
      for (int i=0; i<worldImage->width; i++)
      {
        cvmSet(pwCoord, 0, 0, minWorldX + i*stepSize);
        double ix, iy, six, siy;
        cvMatMul(pmInvH, pwCoord, piCoord);
        ix = cvmGet(piCoord,0,0) / cvmGet(piCoord,2,0);
        iy = cvmGet(piCoord,1,0) / cvmGet(piCoord,2,0);
        if ( ix < 0 || ix > pImage->width-2 || iy < 0 || iy > pImage->height-2)
          continue;
        six = ix - (int)ix;
        siy = iy - (int)iy;
        int x1 = (int)floor(ix);
        int x2 = x1+1;
        int y1 = (int)floor(iy);
        int y2 = y1+1;
        CvScalar s1 = cvGet2D(pImage, y1, x1);
        CvScalar s2 = cvGet2D(pImage, y2, x2);
        CvScalar s3 = cvGet2D(pImage, y1, x2);
        CvScalar s4 = cvGet2D(pImage, y2, x1);
        CvScalar merged;
        for (int t=0; t<4; t++)
        {
          merged.val[t]=0;
          merged.val[t]+=(1-six)*(1-siy)*s1.val[t];
          merged.val[t]+=(six)*(siy)*s2.val[t];
          merged.val[t]+=(six)*(1-siy)*s3.val[t];
          merged.val[t]+=(1-six)*siy*s4.val[t];
        }
        cvSet2D(worldImage, j, i, merged);
      }
    }
    cvReleaseMat(&pmInvH);
    //cvReleaseMat(mH);
    cvReleaseMat(&piCoord);
    cvReleaseMat(&pwCoord);
    cvReleaseMat(&pWorldCoord);
    *returnImage = worldImage;
    if (inRegion == NULL) delete region;
    return;
  }

  double Utiliby::getRotateThetaForVerticleLine(CvMat* H, double* p1, double* p2)
  {
    CvMat* point1 = cvCreateMat(3,1,CV_64FC1);
    CvMat* point2 = cvCreateMat(3,1,CV_64FC1);
    cvmSet(point1,0,0,p1[0]);
    cvmSet(point1,1,0,p1[1]);
    cvmSet(point1,2,0,1);
    cvmSet(point2,0,0,p2[0]);
    cvmSet(point2,1,0,p2[1]);
    cvmSet(point2,2,0,1);
    cvMatMul(H,point1,point1);
    cvMatMul(H,point2,point2);
    double p2y = cvmGet(point2,1,0)/cvmGet(point2,2,0);
    double p2x = cvmGet(point2,0,0)/cvmGet(point2,2,0);
    double p1y = cvmGet(point1,1,0)/cvmGet(point1,2,0);
    double p1x = cvmGet(point1,0,0)/cvmGet(point1,2,0);
    double theta = atan2(p2y-p1y,p2x-p1x);
    double angleToRotate = theta - pi/2;
    cvReleaseMat(&point1);
    cvReleaseMat(&point2);
    return angleToRotate;
  }

}