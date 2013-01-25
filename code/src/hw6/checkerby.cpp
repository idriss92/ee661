#include "checkerby.h"
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

#include <cv.h>
#include <highgui.h>
#include <math.h>

namespace CVisby{

  void Checkerby::LabelCorners(vector<HarrisCorner>& cornersRaster, IplImage* image/* =NULL */)
  {
    IplImage* imageToLabel;
    if (image == NULL) imageToLabel = pImage;
    else pImage = imageToLabel = image;
    assert(imageToLabel!=NULL);

    IplImage* imageCanny = cvCreateImage(cvGetSize(imageToLabel),8,1);
    IplImage* imageHough = cvCreateImage(cvGetSize(imageToLabel),8,3);

    cvCvtColor(imageToLabel,imageCanny,CV_BGR2GRAY);
    cvCanny(imageCanny,imageCanny,50,400,3);
    cvCvtColor(imageCanny, imageHough, CV_GRAY2BGR);
    //cvSaveImage("canny.jpg",imageCanny);

    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* lines = 0;
    lines = cvHoughLines2( imageCanny,
      storage,
      CV_HOUGH_STANDARD,
      1,
      CV_PI/180,
      50,
      0,
      0 );
    //for( int i = 0; i < MIN(lines->total,100); i++ )
    //{
    //  float* line = (float*)cvGetSeqElem(lines,i);
    //  if (line[0] < 0 ) 
    //  {
    //    //line[0] = -line[0];
    //    line[1] += (float) (line[1]>0? -CV_PI:CV_PI);
    //  }
    //}
    //clusterLines(lines, 2*(numCol+numRow));
    ////Drawing all lines for test
    //for( int i = 0; i < MIN((lines)->total,100); i++ )
    //{
    //  float* line = (float*)cvGetSeqElem(lines,i);
    //  float rho = line[0];
    //  cout << rho << endl;
    //  float theta = line[1];
    //  CvPoint pt1, pt2;
    //  double a = cos(theta), b = sin(theta);
    //  double x0 = a*rho, y0 = b*rho;
    //  pt1.x = cvRound(x0 + 1000*(-b));
    //  pt1.y = cvRound(y0 + 1000*(a));
    //  pt2.x = cvRound(x0 - 1000*(-b));
    //  pt2.y = cvRound(y0 - 1000*(a));
    //  cvLine( pImage, pt1, pt2, CV_RGB(255,0,0), 1, 8 );
    //}
    //cvSaveImage("imageBad.jpg",pImage);
    CvSeq* linesHor = cvCloneSeq(lines,storage);
    cvClearSeq( linesHor );
    CvSeq* linesVer = cvCloneSeq(lines,storage);
    cvClearSeq( linesVer );
    sortLines( lines, &linesHor, &linesVer );
    clusterLines(linesHor,2*numRow);
    clusterLines(linesVer,2*numCol);
    ////Drawing horizontal lines for test
    //for( int i = 0; i < MIN((linesHor)->total,100); i++ )
    //{
    //  float* line = (float*)cvGetSeqElem(linesHor,i);
    //  float rho = line[0];
    //  cout << rho << endl;
    //  float theta = line[1];
    //  CvPoint pt1, pt2;
    //  double a = cos(theta), b = sin(theta);
    //  double x0 = a*rho, y0 = b*rho;
    //  pt1.x = cvRound(x0 + 1000*(-b));
    //  pt1.y = cvRound(y0 + 1000*(a));
    //  pt2.x = cvRound(x0 - 1000*(-b));
    //  pt2.y = cvRound(y0 - 1000*(a));
    //  cvLine( pImage, pt1, pt2, CV_RGB(255,0,0), 1, 8 );
    //}
    ////Drawing vertical lines for test
    //for( int i = 0; i < MIN((linesVer)->total,100); i++ )
    //{
    //  float* line = (float*)cvGetSeqElem(linesVer,i);
    //  float rho = line[0];
    //  cout << rho << endl;
    //  float theta = line[1];
    //  CvPoint pt1, pt2;
    //  double a = cos(theta), b = sin(theta);
    //  double x0 = a*rho, y0 = b*rho;
    //  pt1.x = cvRound(x0 + 1000*(-b));
    //  pt1.y = cvRound(y0 + 1000*(a));
    //  pt2.x = cvRound(x0 - 1000*(-b));
    //  pt2.y = cvRound(y0 - 1000*(a));
    //  cvLine( pImage, pt1, pt2, CV_RGB(255,0,0), 1, 8 );
    //}
    //cvSaveImage("imageH.jpg",pImage);
    //cvNamedWindow("abc");
    //cvShowImage("abc",pImage);
    fillCorners(linesHor,linesVer,cornersRaster,true);
    cvReleaseImage(&imageCanny);
    cvReleaseImage(&imageHough);
    cvReleaseMemStorage(&storage);
    
  }

  void Checkerby::clusterLines(CvSeq* lines, int k)
  {
    int loops = lines->total - k;
    for (int n=0; n < loops; n++)
    {
      double minDist = 4000;
      int minI = -1;
      int minJ = -1;
      for( int i = 0; i < lines->total; i++ )
      {
        for (int j = i+1; j < lines->total; j++)
        {
          float* line = (float*)cvGetSeqElem(lines,i);
          float rhoi = line[0];
          float thetai = line[1];
          line = (float*)cvGetSeqElem(lines,j);
          float rhoj = line[0];
          float thetaj = line[1];
          double cost = calcLineMergeCost(rhoi,rhoj,thetai,thetaj);//(fabs(rhoi-rhoj)>27?10000:1)*(thetai-thetaj)*(thetai-thetaj)+fabs((rhoi-rhoj)*(rhoi-rhoj));
          if (cost < minDist /*&& fabs(rhoi-rhoj) < 10*/)
          {
            minDist = cost;
            minI=i;
            minJ=j;
          }
        }
      }
      float* lineI = (float*)cvGetSeqElem(lines,minI);
      float* lineJ = (float*)cvGetSeqElem(lines,minJ);
      assert(minI!=-1);
      assert(minJ!=-1);
      if (lineI[0] * lineJ[0] < 0)
      {
        lineI[1]=(float)(lineI[1]+lineJ[1]-CV_PI)/2;
        lineI[0]=(fabs(lineI[0])+fabs(lineJ[0]))/2;
      }
      else
      {
        lineI[0]=(lineI[0]+lineJ[0])/2;
        lineI[1]=(lineI[1]+lineJ[1])/2;
      }
      
      cvSeqRemove(lines,minJ);
    }
  }

  double Checkerby::calcLineMergeCost(double rho_i,double rho_j, double theta_i,double theta_j)
  {

    double cx = pImage->width / 2.0;
    double cy = pImage->height / 2.0;
    double rho = sqrt(cx*cx+cy*cy);
    int signi = 1;
    int signj = 1;
    double theta_ip = theta_i;
    double theta_jp = theta_j;
    if (rho_i * rho_j < 0)
    {
      if (rho_i < 0)
      {
        theta_i += (float) (theta_i>0? -CV_PI:CV_PI);
        signi = -1;
      }
      if (rho_j < 0)
      {
        theta_j += (float) (theta_j>0? -CV_PI:CV_PI);
        signj = -1;
      }
    }
    double theta=(theta_i+theta_j)/2.0+CV_PI/2;

    if (fabs(theta_i-theta_j)>CV_PI/2)
      return 4000;

    double pC1[3], pC2[3];
    CvMat mpC1 = cvMat(3,1,CV_64FC1, pC1);
    CvMat mpC2 = cvMat(3,1,CV_64FC1, pC2);
    double a = cos(theta), b = sin(theta);
    double x0 = cx, y0 = cy;
    cvmSet(&mpC1,0,0,x0 + 1000*(-b));
    cvmSet(&mpC1,1,0,y0 + 1000*(a));
    cvmSet(&mpC1,2,0,1);
    cvmSet(&mpC2,0,0,x0 - 1000*(-b));
    cvmSet(&mpC2,1,0,y0 - 1000*(a));
    cvmSet(&mpC2,2,0,1);
    CvMat* lC = cvCreateMat(3,1,CV_64FC1);
    cvCrossProduct(&mpC1,&mpC2,lC);
    double pA1[3], pA2[3];
    CvMat mpA1 = cvMat(3,1,CV_64FC1, pA1);
    CvMat mpA2 = cvMat(3,1,CV_64FC1, pA2);
    a = cos(theta_ip);
    b = sin(theta_ip);
    x0 = a*rho_i; 
    y0 = b*rho_i;
    cvmSet(&mpA1,0,0,x0 + 1000*(-b));
    cvmSet(&mpA1,1,0,y0 + 1000*(a));
    cvmSet(&mpA1,2,0,1);
    cvmSet(&mpA2,0,0,x0 - 1000*(-b));
    cvmSet(&mpA2,1,0,y0 - 1000*(a));
    cvmSet(&mpA2,2,0,1);
    CvMat* lA = cvCreateMat(3,1,CV_64FC1);
    cvCrossProduct(&mpA1,&mpA2,lA);
    double pB1[3], pB2[3];
    CvMat mpB1 = cvMat(3,1,CV_64FC1, pB1);
    CvMat mpB2 = cvMat(3,1,CV_64FC1, pB2);
    a = cos(theta_jp);
    b = sin(theta_jp);
    x0 = a*rho_j; 
    y0 = b*rho_j;
    cvmSet(&mpB1,0,0,x0 + 1000*(-b));
    cvmSet(&mpB1,1,0,y0 + 1000*(a));
    cvmSet(&mpB1,2,0,1);
    cvmSet(&mpB2,0,0,x0 - 1000*(-b));
    cvmSet(&mpB2,1,0,y0 - 1000*(a));
    cvmSet(&mpB2,2,0,1);
    CvMat* lB = cvCreateMat(3,1,CV_64FC1);
    cvCrossProduct(&mpB1,&mpB2,lB);

    CvMat* pA = cvCreateMat(3,1,CV_64FC1);
    CvMat* pB = cvCreateMat(3,1,CV_64FC1);
    cvCrossProduct(lC,lA,pA);
    cvCrossProduct(lC,lB,pB);

    /*CvMat* pAB = cvCreateMat(3,1,CV_64FC1);
    cvCrossProduct(lC,lB,pAB);
    double pABx = cvmGet(pAB,0,0)/cvmGet(pAB,2,0);
    double pABy = cvmGet(pAB,1,0)/cvmGet(pAB,2,0);*/

    double a1 = cos(theta_ip), b1 = sin(theta_ip);
    double x1 = cvmGet(pA,0,0)/cvmGet(pA,2,0), y1 = cvmGet(pA,1,0)/cvmGet(pA,2,0);
    double a2 = cos(theta_jp), b2 = sin(theta_jp);
    double x2 = cvmGet(pB,0,0)/cvmGet(pB,2,0), y2 = cvmGet(pB,1,0)/cvmGet(pB,2,0);

    //double distABToC = sqrt((pABx-x1)*(pABx-x1)+(pABy-y1)*(pABy-y1))
    //  +sqrt((pABx-x2)*(pABx-x2)+(pABy-y2)*(pABy-y2));

    double returnValue = 0;
    double unit = 30;
    int window=4;
    for (int k=-window; k<window+1; k++)
    {
      double tx1 = x1 + unit*k*(-b1);
      double ty1 = y1 + unit*k*(a1);
      double tx2 = x2 + unit*k*(-b2);
      double ty2 = y2 + unit*k*(a2);
      double unitdist = sqrt(fabs(tx1-tx2)*fabs(tx1-tx2)+fabs(ty1-ty2)*fabs(ty1-ty2));
      returnValue+=unitdist;
    }
    cvReleaseMat(&lC);
    cvReleaseMat(&lA);
    cvReleaseMat(&lB);
    cvReleaseMat(&pB);
    cvReleaseMat(&pA);
    return returnValue/(2*window+1)/*+100*(theta_i-theta_j)*(theta_i-theta_j)*/+(fabs(rho_i)-fabs(rho_j))*(fabs(rho_i)-fabs(rho_j));// + distABToC;
  }

  void Checkerby::sortLines(CvSeq* lines, CvSeq** linesH, CvSeq** linesV)
  {
    double thetaH = *((float*)cvGetSeqElem(lines,0)+1);
    if ( fabs(CV_PI - thetaH) < CV_PI / 4.0) thetaH -= CV_PI; 
    double thetaV = 0;
    double avgrhoH = 0;
    double avgrhoV = 0;
    for( int i = 0; i < MIN(lines->total,100); i++ )
    {
      float* line = (float*)cvGetSeqElem(lines,i);
      float rho = line[0];
      float theta = line[1];
      if (fabs(theta - thetaH) < CV_PI / 4.0 || fabs(theta - thetaH - CV_PI) < CV_PI / 4.0 )
      {
        CvPoint2D32f pt = cvPoint2D32f(rho,theta);
        cvSeqPush(*linesH,&pt);
        avgrhoH+=rho;
      }
      else
      {
        thetaV+= (CV_PI - theta) < CV_PI/4 ? theta-CV_PI : theta ;
        CvPoint2D32f pt = cvPoint2D32f(rho,theta);
        cvSeqPush(*linesV,&pt);
        avgrhoV+=rho;
      }
    }
    thetaV/=(*linesV)->total;
    avgrhoV/=(*linesV)->total;
    avgrhoH/=(*linesH)->total;

    //assert(((*linesV)->total+(*linesH)->total)==18);
    if (fabs(thetaH-CV_PI/2) > CV_PI/4)//(*linesV)->total > (*linesH)->total )
    {
      CvSeq* temp = *linesV;
      *linesV = *linesH;
      *linesH = temp;
      double tempTheta = thetaH;
      thetaH = thetaV;
      thetaV = tempTheta;
    }
    for( int i = 1; i < MIN((*linesH)->total,100); i++ )
    {
      for (int j = 0; j < i; j++)
      {
        float* linei = (float*)cvGetSeqElem(*linesH,i);
        float* linej = (float*)cvGetSeqElem(*linesH,j);
        if (( fabs(thetaH - CV_PI/2) < CV_PI/4 && linei[0] < linej[0]) 
          ||(( fabs(thetaH - 0) < CV_PI/4 ) && linei[0] > linej[0] )
          ||(( fabs(thetaH - CV_PI) < CV_PI/4 ) && linei[0] < linej[0] )
        )
        {
          float temprho = linei[0];
          float temptheta = linei[1];
          for (int k = i; k >= j+1; k--)
          {
            float* linek = (float*)cvGetSeqElem(*linesH,k);
            float* linekm1 = (float*)cvGetSeqElem(*linesH,k-1);
            linek[0]=linekm1[0];
            linek[1]=linekm1[1];
          }
          linej[0] = temprho;
          linej[1] = temptheta;
        }
      }
    }
    for( int i = 1; i < MIN((*linesV)->total,100); i++ )
    {
      for (int j = 0; j < i; j++)
      {
        float* linei = (float*)cvGetSeqElem(*linesV,i);
        float* linej = (float*)cvGetSeqElem(*linesV,j);
        if (/*( fabs(thetaV - 0) < CV_PI/4 && avgrhoV > 0 && linei[0] < linej[0]) 
          || ( fabs(thetaV - 0) < CV_PI/4 && avgrhoV < 0 && linei[0] > linej[0]) */
          ( fabs(thetaV - 0) < CV_PI/4 && fabs(linei[0]) < fabs(linej[0]))
          || (fabs(thetaV - CV_PI/2) < CV_PI/4 && linei[0] > linej[0]) 
          || ( fabs(thetaV - CV_PI) < CV_PI/4 && linei[0] > linej[0]))
        {
          float temprho = linei[0];
          float temptheta = linei[1];
          for (int k = i; k >= j+1; k--)
          {
            float* linek = (float*)cvGetSeqElem(*linesV,k);
            float* linekm1 = (float*)cvGetSeqElem(*linesV,k-1);
            linek[0]=linekm1[0];
            linek[1]=linekm1[1];
          }
          linej[0] = temprho;
          linej[1] = temptheta;
        }
      }
    }
  }

  void Checkerby::fillCorners(CvSeq* linesHor,CvSeq* linesVer, vector<HarrisCorner>& cornersRaster, bool useHarris /* = true */)
  {
    for (int i=0; i<linesHor->total; i++)
    {
      float* line = (float*) cvGetSeqElem(linesHor,i);
      float rhoi = line[0];
      float thetai = line[1];
      double pH1[3], pH2[3];
      CvMat mpH1 = cvMat(3,1,CV_64FC1, pH1);
      CvMat mpH2 = cvMat(3,1,CV_64FC1, pH2);
      double a = cos(thetai), b = sin(thetai);
      double x0 = a*rhoi, y0 = b*rhoi;
      cvmSet(&mpH1,0,0,x0 + 1000*(-b));
      cvmSet(&mpH1,1,0,y0 + 1000*(a));
      cvmSet(&mpH1,2,0,1);
      cvmSet(&mpH2,0,0,x0 - 1000*(-b));
      cvmSet(&mpH2,1,0,y0 - 1000*(a));
      cvmSet(&mpH2,2,0,1);
      CvMat* lH = cvCreateMat(3,1,CV_64FC1);
      cvCrossProduct(&mpH1,&mpH2,lH);
      for (int j=0; j < linesVer->total; j++)
      {
        float* line = (float*) cvGetSeqElem(linesVer,j);
        float rhoj = line[0];
        float thetaj = line[1];
        double pV1[3], pV2[3];
        CvMat mpV1 = cvMat(3,1,CV_64FC1, pV1);
        CvMat mpV2 = cvMat(3,1,CV_64FC1, pV2);
        double a = cos(thetaj), b = sin(thetaj);
        double x0 = a*rhoj, y0 = b*rhoj;
        cvmSet(&mpV1,0,0,x0 + 1000*(-b));
        cvmSet(&mpV1,1,0,y0 + 1000*(a));
        cvmSet(&mpV1,2,0,1);
        cvmSet(&mpV2,0,0,x0 - 1000*(-b));
        cvmSet(&mpV2,1,0,y0 - 1000*(a));
        cvmSet(&mpV2,2,0,1);
        CvMat* lV = cvCreateMat(3,1,CV_64FC1);
        cvCrossProduct(&mpV1,&mpV2,lV);

        CvMat* pC = cvCreateMat(3,1,CV_64FC1);
        cvCrossProduct(lH, lV, pC);
        cornersRaster.push_back(HarrisCorner(int(cvmGet(pC,0,0)/cvmGet(pC,2,0)),
          int(cvmGet(pC,1,0)/cvmGet(pC,2,0))));
        cvReleaseMat(&lV);
        cvReleaseMat(&pC);
      }
      cvReleaseMat(&lH);
    }

    if (useHarris)
    {
      vector<HarrisCorner> harrisCorners;
      Harrisby harrisby;
      harrisby.setWindowSize(2);
      harrisby.setMatchingSize(5);
      IplImage *pImageGray = cvCreateImage(cvGetSize(pImage),8,1);
      cvCvtColor(pImage, pImageGray,CV_BGR2GRAY);
      harrisby.detectHarrisCorner(pImageGray, harrisCorners,0.08);
      cvReleaseImage(&pImageGray);
      for (unsigned int i=0; i < cornersRaster.size(); i++)
      {
        double minDist = 1000;
        unsigned int minJ = harrisCorners.size()+1;
        for (unsigned int j=0; j < harrisCorners.size(); j++)
        {
          double dist = cornersRaster[i]-harrisCorners[j];
          if ( dist < minDist )
          {
            minDist = dist;
            minJ = j;
          }
        }
        assert(minJ!=harrisCorners.size()+1);
        cornersRaster[i] = harrisCorners[minJ];
      }
    }
    ////numbering points for test
    //for(unsigned int i = 0; i < cornersRaster.size(); i++ )
    //{
    //  CvPoint pt;
    //  pt.x = cornersRaster[i].x;
    //  pt.y = cornersRaster[i].y;
    //  cvCircle(pImage,pt,1, CV_RGB(0,255,0), 1, 8);
    //  CvFont font;
    //  cvInitFont( &font, CV_FONT_HERSHEY_COMPLEX_SMALL, .6, .6, 0, 1, 6);
    //  char number[4];
    //  _itoa(i,number,10);
    //  cvPutText( pImage, number, pt, &font, CV_RGB(255,255,0) );
    //}
    //cvSaveImage("imageCorner.jpg",pImage);
    //cvNamedWindow("abc");
    //cvShowImage("abc",pImage);
  }

  void Checkerby::drawCorners(vector<HarrisCorner>& cornersToDraw, int line_type, 
    CvScalar s, double* H/* =NULL */, IplImage* pImageToDraw)
  {
    if (!pImageToDraw) pImageToDraw = pImage;
    bool insideH = false;
    if (H==NULL)
    {
      H = new double[9];
      memset(H,0,sizeof(double)*9);
      H[0]=H[4]=H[8]=1;
      insideH = true;
    }
    CvMat mH = cvMat(3,3,CV_64FC1,H);
    //drawing corners
    for(unsigned int i = 0; i < cornersToDraw.size(); i++ )
    {
      CvMat* pointWorld = cvCreateMat(3,1,CV_64FC1);
      cvmSet(pointWorld,0,0,cornersToDraw[i].x);
      cvmSet(pointWorld,1,0,cornersToDraw[i].y);
      cvmSet(pointWorld,2,0,1);
      cvMatMul(&mH,pointWorld,pointWorld);
      CvPoint pt;
      pt.x = (int)(cvmGet(pointWorld,0,0)/cvmGet(pointWorld,2,0));
      pt.y = (int)(cvmGet(pointWorld,1,0)/cvmGet(pointWorld,2,0));
      CvPoint pt1;
      pt1.x = pt.x+3;
      pt1.y = pt.y;
      CvPoint pt2;
      pt2.x=pt.x-3;
      pt2.y=pt.y;
      cvLine(pImageToDraw,pt1,pt2,s,1,line_type);
      pt1.x = pt.x;
      pt1.y = pt.y+3;
      pt2.x=pt.x;
      pt2.y=pt.y-3;
      cvLine(pImageToDraw,pt1,pt2,s,1,line_type);
      cvReleaseMat(&pointWorld);
    }
    if (insideH)
      delete[] H;
  }

  void Checkerby::drawCorners(vector<HarrisCorner>& cornersToDraw, int line_type, 
    CvScalar s, double* K, double* Rt, IplImage* pImageToDraw)
  {
    if (!pImageToDraw) pImageToDraw=pImage;
    assert(K!=NULL);
    assert(Rt!=NULL);
    CvMat mK = cvMat(3,3,CV_64FC1,K);
    CvMat* mRt = cvCreateMat(3,3,CV_64FC1);
    cvmSet(mRt,0,0,Rt[0]);
    cvmSet(mRt,1,0,Rt[4]);
    cvmSet(mRt,2,0,Rt[8]);
    cvmSet(mRt,0,1,Rt[1]);
    cvmSet(mRt,1,1,Rt[5]);
    cvmSet(mRt,2,1,Rt[9]);
    cvmSet(mRt,0,2,Rt[3]);
    cvmSet(mRt,1,2,Rt[7]);
    cvmSet(mRt,2,2,Rt[11]);
    CvMat* mH = cvCreateMat(3,3,CV_64FC1);
    cvMatMul(&mK,mRt,mH);
    //drawing corners
    for(unsigned int i = 0; i < cornersToDraw.size(); i++ )
    {
      CvMat* pointWorld = cvCreateMat(3,1,CV_64FC1);
      cvmSet(pointWorld,0,0,cornersToDraw[i].x);
      cvmSet(pointWorld,1,0,cornersToDraw[i].y);
      cvmSet(pointWorld,2,0,1);
      cvMatMul(mH,pointWorld,pointWorld);
      CvPoint pt;
      pt.x = (int)(cvmGet(pointWorld,0,0)/cvmGet(pointWorld,2,0));
      pt.y = (int)(cvmGet(pointWorld,1,0)/cvmGet(pointWorld,2,0));
      //cvCircle(pImage,pt,1, s, 1, line_type);
      CvPoint pt1;
      pt1.x = pt.x+3;
      pt1.y = pt.y+3;
      CvPoint pt2;
      pt2.x=pt.x-3;
      pt2.y=pt.y-3;
      cvLine(pImageToDraw,pt1,pt2,s,1,line_type);
      pt1.x = pt.x+3;
      pt1.y = pt.y-3;
      pt2.x=pt.x-3;
      pt2.y=pt.y+3;
      cvLine(pImageToDraw,pt1,pt2,s,1,line_type);
      cvReleaseMat(&pointWorld);
    }
    cvReleaseMat(&mRt);
    cvReleaseMat(&mH);
  }

}
