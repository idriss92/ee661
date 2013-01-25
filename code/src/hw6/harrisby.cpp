#include "harrisby.h"
#include "utiliby.h"
#include <iostream>
#include "time.h"
using namespace std;

#include <cv.h>
#include <highgui.h>
#include <math.h>

namespace CVisby{

  void Harrisby::detectHarrisCorner(IplImage *pImage, 
    vector<HarrisCorner>& corners, double threshold /* = 0.1 */, IplImage *pImageOut/* = NULL*/)
  {
    int w = pImage->width;
    int h = pImage->height;
    int imageSize = w*h;
    double *I = new double[imageSize*2];
    double *G = new double[imageSize*3];
    double *H = new double[imageSize*2];
    double *HSup = new double [imageSize*2];
    double sobelX[] = {-1,0,1,-2,0,2,-1,0,1};
    double sobelY[] = {-1,-2,-1,0,0,0,1,2,1};

    for (int i=0; i < w; i++)
    {
      for (int j=0; j < h; j++)
      {
        I[(i*h+j)*2+0] = Utiliby::convolveMask(sobelY, 1, pImage, i, j);
        I[(i*h+j)*2+1] = Utiliby::convolveMask(sobelX, 1, pImage, i, j);
      }
    }

    cv::Mat gaussianMat = cv::getGaussianKernel(HarrisLen,1);
    for (int i=0; i < w; i++)
    {
      for (int j=0; j < h; j++)
      {
        double g0=0,g1=0,g2=0;
        for (int ki=-HarrisRadius; ki <= HarrisRadius; ki++)
        {
          for (int kj=-HarrisRadius; kj <= HarrisRadius; kj++)
          {
            double ix=0,iy=0;
            if (i+ki >=0 && i+ki < w && j+kj >=0 && j+kj < h)
            {
              ix = I[((i+ki)*h+j+kj)*2+0];
              iy = I[((i+ki)*h+j+kj)*2+1];
            }
            double gfilteri = gaussianMat.at<double>(kj+HarrisRadius,0);
            double gfilterj = gaussianMat.at<double>(ki+HarrisRadius,0);
            g0+=gfilteri*gfilterj*ix*ix;
            g1+=gfilteri*gfilterj*ix*iy;
            g2+=gfilteri*gfilterj*iy*iy;
          }
        }
        //g0/=HarrisWindow;
        //g1/=HarrisWindow;
        //g2/=HarrisWindow;
        G[(i*h+j)*3+0]=g0;
        G[(i*h+j)*3+1]=g1;
        G[(i*h+j)*3+2]=g2;
      }
    }

    //Use CVSVD
    for (int i=0; i < w; i++)
    {
      for (int j=0; j < h; j++)
      {
        CvMat* gMat = cvCreateMat(2,2,CV_64FC1);
        cvmSet(gMat, 0,0,G[(i*h+j)*3+0]);
        cvmSet(gMat, 0,1,G[(i*h+j)*3+1]);
        cvmSet(gMat, 1,0,G[(i*h+j)*3+1]);
        cvmSet(gMat, 1,1,G[(i*h+j)*3+2]);
        CvMat* U = cvCreateMat(2,2,CV_64FC1);
        CvMat* D = cvCreateMat(2,2,CV_64FC1);
        CvMat* V = cvCreateMat(2,2,CV_64FC1);
        cvSVD( gMat, D, U ,V, CV_SVD_U_T|CV_SVD_V_T);
        //H[(i*h+j)*2+0] = cvmGet(D,0,0);
        //H[(i*h+j)*2+1] = cvmGet(D,1,1);
        double lambda1 = cvmGet(D,0,0);
        double lambda2 = cvmGet(D,1,1);
        H[(i*h+j)*2+0] = lambda1*lambda2 - 0.04 * (lambda1 + lambda2) * (lambda1 + lambda2);
        H[(i*h+j)*2+1] = 0;
        cvReleaseMat(&gMat);
        cvReleaseMat(&U);
        cvReleaseMat(&D);
        cvReleaseMat(&V);
      }
    }

    //Use Taylor
    /*for (int i=0; i < w; i++)
    {
      for (int j=0; j < h; j++)
      {
        H[(i*h+j)*2+0] = (G[(i*h+j)*3+0]*G[(i*h+j)*3+2]-G[(i*h+j)*3+1]*G[(i*h+j)*3+1])-0.04*(G[(i*h+j)*3+0]+G[(i*h+j)*3+2])*(G[(i*h+j)*3+0]+G[(i*h+j)*3+2]);
        H[(i*h+j)*2+1] = 0;
      }
    }*/

    double HMax = 0;
    int countmaxima=0;
    memset(HSup, 0, 2*imageSize*sizeof(double));
    int suppressRadius = 2*HarrisRadius;//was 5*HarrisRadius, for calibration, decreased to 2
    for (int pi=suppressRadius; pi < w-suppressRadius; pi++)
    {
      for (int pj=suppressRadius; pj < h-suppressRadius; pj++)
      {
        bool isLocalMaxima = true;
        
        for (int i = -suppressRadius; i <= suppressRadius; i++)
        {
          for (int j = -suppressRadius; j <= suppressRadius; j++)
          {
            if ( H[(pi*h+pj)*2] < H[((pi+i)*h+pj+j)*2] )
            {
              isLocalMaxima = false;
              break;
            }
          }
          if (!isLocalMaxima) break;
        }
        if (isLocalMaxima) 
        {
          countmaxima++;
          HSup[(pi*h+pj)*2+0] = H[(pi*h+pj)*2+0];
          HSup[(pi*h+pj)*2+1] = H[(pi*h+pj)*2+1];
          if (HSup[(pi*h+pj)*2+0] > HMax)
            HMax = HSup[(pi*h+pj)*2+0];
        }
      }
    }

    int lastX = -5;
    int lastY = -5;
    for (int i=0; i < w; i++)
    {
      for (int j=0; j < h; j++)
      {
        double h1 = HSup[(i*h+j)*2+0];
        double h2 = HSup[(i*h+j)*2+1];
        //Use CVSVD
        /*if ( h1 < 2000 )
          continue;
        if ( h2/h1 < threshold )
          continue;
        else
        {
          if (i-lastX >= HarrisLen && j-lastY >= HarrisLen 
            && i >= HarrisRadius && j>=HarrisRadius
            && i+HarrisRadius < w && j+HarrisRadius < h)
            corners.insert(corners.end(),HarrisCorner(i,j));
        }*/
        //Use Taylor
        if ( h1 < HMax * threshold )
          continue;
        else
        {
          if (i-lastX >= HarrisDist && j-lastY >= HarrisDist 
            && i >= HarrisRadius && j>=HarrisRadius
            && i+HarrisRadius < w && j+HarrisRadius < h)
            corners.insert(corners.end(),HarrisCorner(i,j));
        }
      }
    }

    delete[] I;
    delete[] G;
    delete[] H;
    delete[] HSup;

    if (pImageOut)
    for (unsigned int i = 0; i < corners.size(); i++)
    {
      cvCircle(pImageOut, cvPoint(corners[i].x, corners[i].y), HarrisRadius, cvScalar(255, 0, 0));
    }
  }

  void Harrisby::matchNCC(IplImage *pImage1, IplImage *pImage2, 
    const vector<HarrisCorner>& corners1, const vector<HarrisCorner>& corners2, 
    vector<std::pair<int,int> >& pairs, double distThres /* = 1 */)
  {
    for (unsigned int i=0; i < corners1.size(); i++)
    {
      double maxNCC = -100000000;
      double maxNCCSecond = -100000000;
      int matching1to2 = -1;
      int matching1to2Second = -1;
      for (unsigned int j=0; j < corners2.size(); j++)
      {
        int x1 = corners1[i].x;
        int y1 = corners1[i].y;
        int x2 = corners2[j].x;
        int y2 = corners2[j].y;
        if (abs(x1-x2) > (MatchingRange * pImage1->width) || abs(y1-y2) > (MatchingRange * pImage1->height))
        {
          continue;
        }
        if ( x1 >=MatchingLen && x1 < pImage1->width-MatchingLen && y1 >=MatchingLen && y1 < pImage1->height-MatchingLen &&
             x2 >=MatchingLen && x2 < pImage2->width-MatchingLen && y2 >=MatchingLen && y2 < pImage2->height-MatchingLen )
        {
          double ncc = 0;
          double nccX1 = 0;
          double nccX2 = 0;
          double nccX11 = 0;
          double nccX22 = 0;
          double nccX12 = 0;
          //cv::Mat gaussianMat = cv::getGaussianKernel(MatchingLen,-1);
          for (int hi = -MatchingRadius; hi <= MatchingRadius; hi++)
            for (int hj = -MatchingRadius; hj <= MatchingRadius; hj++)
            {
              double gfilter = 1.0/MatchingWindow;//;gaussianMat.at<double>(hi+MatchingRadius,0)*gaussianMat.at<double>(hj+MatchingRadius,0);
              CvScalar p1 = cvGet2D(pImage1,y1+hj,x1+hi);
              CvScalar p2 = cvGet2D(pImage2,y2+hj,x2+hi);
              double v1 = p1.val[0];
              double v2 = p2.val[0];
              nccX1 += gfilter * v1;
              nccX2 += gfilter * v2;
              /*nccX11 += gfilter * v1*v1;
              nccX22 += gfilter * v2*v2;
              nccX12 += gfilter * v1*v2;*/
            }
          for (int hi = -MatchingRadius; hi <= MatchingRadius; hi++)
            for (int hj = -MatchingRadius; hj <= MatchingRadius; hj++)
            {
              double gfilter = 1.0/MatchingWindow;//;gaussianMat.at<double>(hi+MatchingRadius,0)*gaussianMat.at<double>(hj+MatchingRadius,0);
              CvScalar p1 = cvGet2D(pImage1,y1+hj,x1+hi);
              CvScalar p2 = cvGet2D(pImage2,y2+hj,x2+hi);
              double v1 = p1.val[0] - nccX1;
              double v2 = p2.val[0] - nccX2;
              nccX11 += v1*v1;
              nccX22 += v2*v2;
              nccX12 += v1*v2;
            }
          /*nccX1=nccX1/MatchingWindow;
          nccX2=nccX2/MatchingWindow;
          nccX11=nccX11/MatchingWindow;
          nccX12=nccX12/MatchingWindow;
          nccX22=nccX22/MatchingWindow;*/
          //ncc = (nccX12 - nccX1*nccX2)/sqrt(nccX11-nccX1*nccX1)/sqrt(nccX22-nccX2*nccX2);
          ncc = (nccX12)/sqrt(nccX11*nccX22);
          if (ncc > maxNCC)
          {
            matching1to2 = j;
            maxNCC = ncc;
          }
          else if (ncc > maxNCCSecond)
          {
            matching1to2Second = j;
            maxNCCSecond = ncc;
          }
        }
      }
      if (maxNCC > distThres && matching1to2 > matching1to2Second + 0.05)
      {
        pairs.push_back(std::pair<int,int>(i,matching1to2));
      }
    }
    return;
  }

  void Harrisby::matchNCC(IplImage *pImage1, IplImage *pImage2, 
    const vector<HarrisCorner>& corners1, const vector<HarrisCorner>& corners2, 
    vector<std::pair<std::pair<int,int>,double> >& pairs, double distThres /* = 1 */)
  {
    for (unsigned int i=0; i < corners1.size(); i++)
    {
      double maxNCC = -100000000;
      double maxNCCSecond = -100000000;
      int matching1to2 = -1;
      int matching1to2Second = -1;
      for (unsigned int j=0; j < corners2.size(); j++)
      {
        int x1 = corners1[i].x;
        int y1 = corners1[i].y;
        int x2 = corners2[j].x;
        int y2 = corners2[j].y;
        if (abs(x1-x2) > (MatchingRange * pImage1->width) || abs(y1-y2) > (MatchingRange * pImage1->height))
        {
          continue;
        }
        if ( x1 >=MatchingLen && x1 < pImage1->width-MatchingLen && y1 >=MatchingLen && y1 < pImage1->height-MatchingLen &&
             x2 >=MatchingLen && x2 < pImage2->width-MatchingLen && y2 >=MatchingLen && y2 < pImage2->height-MatchingLen )
        {
          double ncc = 0;
          double nccX1 = 0;
          double nccX2 = 0;
          double nccX11 = 0;
          double nccX22 = 0;
          double nccX12 = 0;
          //cv::Mat gaussianMat = cv::getGaussianKernel(MatchingLen,-1);
          for (int hi = -MatchingRadius; hi <= MatchingRadius; hi++)
            for (int hj = -MatchingRadius; hj <= MatchingRadius; hj++)
            {
              double gfilter = 1.0/MatchingWindow;//;gaussianMat.at<double>(hi+MatchingRadius,0)*gaussianMat.at<double>(hj+MatchingRadius,0);
              CvScalar p1 = cvGet2D(pImage1,y1+hj,x1+hi);
              CvScalar p2 = cvGet2D(pImage2,y2+hj,x2+hi);
              double v1 = p1.val[0];
              double v2 = p2.val[0];
              nccX1 += gfilter * v1;
              nccX2 += gfilter * v2;
              /*nccX11 += gfilter * v1*v1;
              nccX22 += gfilter * v2*v2;
              nccX12 += gfilter * v1*v2;*/
            }
          for (int hi = -MatchingRadius; hi <= MatchingRadius; hi++)
            for (int hj = -MatchingRadius; hj <= MatchingRadius; hj++)
            {
              double gfilter = 1.0/MatchingWindow;//;gaussianMat.at<double>(hi+MatchingRadius,0)*gaussianMat.at<double>(hj+MatchingRadius,0);
              CvScalar p1 = cvGet2D(pImage1,y1+hj,x1+hi);
              CvScalar p2 = cvGet2D(pImage2,y2+hj,x2+hi);
              double v1 = p1.val[0] - nccX1;
              double v2 = p2.val[0] - nccX2;
              nccX11 += v1*v1;
              nccX22 += v2*v2;
              nccX12 += v1*v2;
            }
          /*nccX1=nccX1/MatchingWindow;
          nccX2=nccX2/MatchingWindow;
          nccX11=nccX11/MatchingWindow;
          nccX12=nccX12/MatchingWindow;
          nccX22=nccX22/MatchingWindow;*/
          //ncc = (nccX12 - nccX1*nccX2)/sqrt(nccX11-nccX1*nccX1)/sqrt(nccX22-nccX2*nccX2);
          ncc = (nccX12)/sqrt(nccX11*nccX22);
          if (ncc > maxNCC)
          {
            matching1to2 = j;
            maxNCC = ncc;
          }
          else if (ncc > maxNCCSecond)
          {
            matching1to2Second = j;
            maxNCCSecond = ncc;
          }
        }
      }
      if (maxNCC > distThres /*&& matching1to2 > matching1to2Second + 0.05*/)
      {
        pairs.push_back(std::pair<std::pair<int,int>,double>(
          std::pair<int,int>(i,matching1to2),maxNCC));
      }
    }
    return;
  }

  void Harrisby::matchSSD(IplImage *pImage1, IplImage *pImage2, 
    const vector<HarrisCorner>& corners1, const vector<HarrisCorner>& corners2, 
    vector<std::pair<int,int> >& pairs, double distThres /* = 1 */)
  {
    for (unsigned int i=0; i < corners1.size(); i++)
    {
      double minSSD = 100000000;
      double minSSDSecond = 100000000;
      int matching1to2 = -1;
      int matching1to2Second = -1;
      for (unsigned int j=0; j < corners2.size(); j++)
      {
        int x1 = corners1[i].x;
        int y1 = corners1[i].y;
        int x2 = corners2[j].x;
        int y2 = corners2[j].y;
        if (abs(x1-x2) > (MatchingRange * pImage1->width) || abs(y1-y2) > (MatchingRange * pImage1->height))
        {
          continue;
        }
        if ( x1 >=MatchingLen && x1 < pImage1->width-MatchingLen && y1 >=MatchingLen && y1 < pImage1->height-MatchingLen &&
          x2 >=MatchingLen && x2 < pImage2->width-MatchingLen && y2 >=MatchingLen && y2 < pImage2->height-MatchingLen )
        {
          double ssd = 0;
          cv::Mat gaussianMat = cv::getGaussianKernel(MatchingLen,-1);
          for (int hi = -MatchingRadius; hi <= MatchingRadius; hi++)
          {
            for (int hj = -MatchingRadius; hj <= MatchingRadius; hj++)
            {
              double gfilter = 1.0/MatchingWindow;//gaussianMat.at<double>(hi+MatchingRadius,0)*gaussianMat.at<double>(hj+MatchingRadius,0);
              CvScalar p1 = cvGet2D(pImage1,y1+hj,x1+hi);
              CvScalar p2 = cvGet2D(pImage2,y2+hj,x2+hi);
              ssd+=gfilter*(p1.val[0]-p2.val[0])*(p1.val[0]-p2.val[0]);
            }
          }
          ssd=sqrt(ssd/*/MatchingWindow*/)/255;
          if (ssd < minSSD)
          {
            matching1to2 = j;
            minSSD = ssd;
          }
          else if (ssd < minSSDSecond)
          {
            matching1to2Second = j;
            minSSDSecond = ssd;
          }
        }
      }
      if (minSSD < distThres /*&& minSSD < minSSDSecond - 0.02*/)
      {
        pairs.push_back(std::pair<int,int>(i,matching1to2));
      }
    }
    return;
  }

  void Harrisby::drawCombinedImage(IplImage *pImage1, IplImage *pImage2, 
    IplImage *pImageOut, const vector<HarrisCorner>& corners1, 
    const vector<HarrisCorner>& corners2, const vector<std::pair<int,int> >& pairs)
  {
    cvZero(pImageOut);
    for (int i=0; i < pImage1->width; i++)
      for (int j=0; j < pImage1->height; j++)
      {
        cvSet2D(pImageOut,j,i,cvGet2D(pImage1,j,i));
      }
    for (int i=0; i < pImage2->width; i++)
      for (int j=0; j < pImage2->height; j++)
      {
        cvSet2D(pImageOut,j,i+pImage1->width,cvGet2D(pImage2,j,i));
      }
    for (unsigned int i=0; i < pairs.size(); i++)
    {
      unsigned int c1 = pairs[i].first;
      unsigned int c2 = pairs[i].second;
      int p1x = corners1[c1].x;
      int p1y = corners1[c1].y;
      int p2x = corners2[c2].x;
      int p2y = corners2[c2].y;
      //cvCircle(pImageOut, cvPoint(p1x, p1y), HarrisRadius, cvScalar(255, 0, 0));
      //cvCircle(pImageOut, cvPoint(p2x+pImage1->width, p2y), HarrisRadius, cvScalar(255, 0, 0));
      cvLine(pImageOut,cvPoint(p1x, p1y), cvPoint(p2x+pImage1->width, p2y), cvScalar(0,255,0));
    }
    for (unsigned int i = 0; i < corners1.size(); i++)
    {
      cvCircle(pImageOut, cvPoint(corners1[i].x, corners1[i].y), HarrisRadius, cvScalar(255, 0, 0));
    }
    for (unsigned int i = 0; i < corners2.size(); i++)
    {
      cvCircle(pImageOut, cvPoint(corners2[i].x+pImage1->width, corners2[i].y), HarrisRadius, cvScalar(0, 0, 255));
    }
  }

  void Harrisby::drawCombinedImage(IplImage *pImage1, IplImage *pImage2, 
    IplImage *pImageOut, const vector<HarrisCorner>& corners1, 
    const vector<HarrisCorner>& corners2, const vector<std::pair<std::pair<int,int>,double> >& pairs
    , bool useRandomColor)
  {
    //if (useRandomColor) srand( (unsigned int)time(NULL));
    //CvScalar fixedPointColor= cvScalar(255,0,0);
    CvScalar fixedLineColor = cvScalar(0,255,0);
    cvZero(pImageOut);
    for (int i=0; i < pImage1->width; i++)
      for (int j=0; j < pImage1->height; j++)
      {
        cvSet2D(pImageOut,j,i,cvGet2D(pImage1,j,i));
      }
      for (int i=0; i < pImage2->width; i++)
        for (int j=0; j < pImage2->height; j++)
        {
          cvSet2D(pImageOut,j,i+pImage1->width,cvGet2D(pImage2,j,i));
        }
        for (unsigned int i=0; i < pairs.size(); i++)
        {
          unsigned int c1 = pairs[i].first.first;
          unsigned int c2 = pairs[i].first.second;
          int p1x = corners1[c1].x;
          int p1y = corners1[c1].y;
          int p2x = corners2[c2].x;
          int p2y = corners2[c2].y;
          //cvCircle(pImageOut, cvPoint(p1x, p1y), HarrisRadius, cvScalar(255, 0, 0));
          //cvCircle(pImageOut, cvPoint(p2x+pImage1->width, p2y), HarrisRadius, cvScalar(255, 0, 0));
          CvScalar colorToDraw;
          if (useRandomColor) 
          {
            CvRNG rng;
            int color = cvRandInt(&rng);
            colorToDraw = CV_RGB(color&255, (color>>8)&255, (color>>16)&255);
          }
          else colorToDraw = fixedLineColor;
          cvLine(pImageOut,cvPoint(p1x, p1y), cvPoint(p2x+pImage1->width, p2y), colorToDraw);
        }
        for (unsigned int i = 0; i < corners1.size(); i++)
        {
          cvCircle(pImageOut, cvPoint(corners1[i].x, corners1[i].y), HarrisRadius, cvScalar(255, 0, 0));
        }
        for (unsigned int i = 0; i < corners2.size(); i++)
        {
          cvCircle(pImageOut, cvPoint(corners2[i].x+pImage1->width, corners2[i].y), HarrisRadius, cvScalar(0, 0, 255));
        }
  }

}
