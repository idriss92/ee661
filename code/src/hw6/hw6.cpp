#include "utiliby.h"
#include "harrisby.h"
#include "RANSACby.h"
#include "checkerby.h"
#include "zhengyouby.h"
#include "levmarby.h"
#include <iostream>
#include <fstream>
#include <cv.h>
#include <highgui.h>
#include <string>
#include <vector>

#if defined(_WIN32) || defined(WIN32)
#include <windows.h>
#include <stack>
#include <iostream>
#endif

using namespace CVisby;
using namespace std;

#if defined( _WIN32) || defined(WIN32)
bool ListFiles(string path, string mask, vector<string>& files) {
  HANDLE hFind = INVALID_HANDLE_VALUE;
  WIN32_FIND_DATA ffd;
  string spec;
  stack<string> directories;

  directories.push(path);
  files.clear();

  while (!directories.empty()) {
    path = directories.top();
    spec = path + "\\" + mask;
    directories.pop();

    hFind = FindFirstFile(spec.c_str(), &ffd);
    if (hFind == INVALID_HANDLE_VALUE)  {
      return false;
    } 

    do {
      if (strcmp(ffd.cFileName, ".") != 0 && 
        strcmp(ffd.cFileName, "..") != 0) {
          if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            directories.push(path + "\\" + ffd.cFileName);
          }
          else {
            files.push_back(path + "\\" + ffd.cFileName);
          }
      }
    } while (FindNextFile(hFind, &ffd) != 0);

    if (GetLastError() != ERROR_NO_MORE_FILES) {
      FindClose(hFind);
      return false;
    }

    FindClose(hFind);
    hFind = INVALID_HANDLE_VALUE;
  }

  return true;
}
#endif

int usage(char **argv)
{
  std::string filename(argv[0]);
#ifdef _WIN32
  filename = filename.substr( filename.find_last_of( '\\' ) +1 );
#else
  filename = filename.substr( filename.find_last_of( '/' ) +1 );
#endif
  std::cout << "Usage: " << filename << "\nsee test.bash for instructions.";
  return 1;
}

int main(int argc, char **argv)
{
  IplImage *pImage1 = NULL;
  //IplImage *pImage2 = NULL;
  //IplImage *pImCopy1 = NULL;
  //IplImage *pImCopy2 = NULL;
  //IplImage *pImageOut = NULL;
  //IplImage *pImageOut1 = NULL;

  //double harrisRatio = 0.1;
  //int harrisSize = 2;
  //int matchingSize = 15;
  //double SSDThres = 10;
  //double NCCThres = 0.5;

  string fnDirectory;
  bool outputH = false;
  bool inputH = false;
  bool outputX = false;
  bool inputX = false;
  bool outputK = false;
  bool outputR = false;
  bool inputK = false;
  bool inputR = false;
  bool outputMATLAB = false;
  bool outputBPM = false;
  bool outputBPL = false;

  string optMode;
  string fnOUTH;
  string fnINH;
  string fnOUTX;
  string fnINX;
  string fnOUTK;
  string fnOUTR;
  string fnINK;
  string fnINR;
  string fnINBPM;
  string fnINBPL;
  string fnDirectoryBP;

  if ( argc >= 2 )
  {
    
    for (int i=1; i < argc; i++)
    {
      string argi=string(argv[i]);
      if (argi.find("-d") != string::npos)
      {
        if (argi.find("-dbp") != string::npos)
        {
          argi = argi.substr(argi.find("-dbp")+4);
          stringstream ss(argi);
          ss >> fnDirectoryBP;
        }
        else{
          argi = argi.substr(argi.find("-d")+2);
          stringstream ss(argi);
          ss >> fnDirectory;
        }
      }
      else if (argi.find("-oh") != string::npos)
      {
        argi = argi.substr(argi.find("-o")+3);
        stringstream ss(argi);
        ss >> fnOUTH;
        outputH = true;
      }
      else if (argi.find("-rh") != string::npos)
      {
        argi = argi.substr(argi.find("-rh")+3);
        stringstream ss(argi);
        ss >> fnINH;
        inputH = true;
      }
      else if (argi.find("-ox") != string::npos)
      {
        argi = argi.substr(argi.find("-o")+3);
        stringstream ss(argi);
        ss >> fnOUTX;
        outputX = true;
      }
      else if (argi.find("-rx") != string::npos)
      {
        argi = argi.substr(argi.find("-rx")+3);
        stringstream ss(argi);
        ss >> fnINX;
        inputX = true;
      }
      else if (argi.find("-ok") != string::npos)
      {
        argi = argi.substr(argi.find("-ok")+3);
        stringstream ss(argi);
        ss >> fnOUTK;
        outputK = true;
      }
      else if (argi.find("-or") != string::npos)
      {
        argi = argi.substr(argi.find("-or")+3);
        stringstream ss(argi);
        ss >> fnOUTR;
        outputR = true;
      }
      else if (argi.find("-rk") != string::npos)
      {
        argi = argi.substr(argi.find("-rk")+3);
        stringstream ss(argi);
        ss >> fnINK;
        inputK = true;
      }
      else if (argi.find("-rr") != string::npos)
      {
        argi = argi.substr(argi.find("-rr")+3);
        stringstream ss(argi);
        ss >> fnINR;
        inputR = true;
      }
      else if (argi.find("-OPT") != string::npos)
      {
        argi = argi.substr(argi.find("-OPT")+4);
        stringstream ss(argi);
        ss >> optMode;
        outputMATLAB = true;
      }
      else if (argi.find("-BPM") != string::npos)
      {
        argi = argi.substr(argi.find("-BPM")+4);
        stringstream ss(argi);
        ss >> fnINBPM;
        outputBPM = true;
      }
      else if (argi.find("-BPL") != string::npos)
      {
        argi = argi.substr(argi.find("-BPL")+4);
        stringstream ss(argi);
        ss >> fnINBPL;
        outputBPL = true;
      }
    }
  }
  else
  {
    return usage(argv);
  }

  double* allHomos = NULL;
  double* allX = NULL;
  int numObs;

  if (inputH && inputX)
  {
    ifstream fp;
    fp.open(fnINH.c_str(), ios::in);
    vector<string>lines;
    string line;
    while(getline(fp,line))
    {
      lines.push_back(line);
    }
    allHomos = new double[lines.size()*9];
    numObs = lines.size();
    fp.close();
    for (unsigned int i=0; i<lines.size();i++)
    {
      double* H1ToH2 = allHomos+i*9;
      istringstream linestream(lines[i]);
      linestream >> H1ToH2[0];
      linestream >> H1ToH2[1];
      linestream >> H1ToH2[2];
      linestream >> H1ToH2[3];
      linestream >> H1ToH2[4];
      linestream >> H1ToH2[5];
      linestream >> H1ToH2[6];
      linestream >> H1ToH2[7];
      linestream >> H1ToH2[8];
    }
    allX = new double[lines.size()*320];
    ifstream fpX;
    fpX.open(fnINX.c_str(), ios::in);
    vector<string>lines_X;
    while(getline(fpX,line))
    {
      lines_X.push_back(line);
    }
    fpX.close();
    for (unsigned int i=0; i<lines_X.size();i++)
    {
      double* XPos = allX+i*320;
      istringstream linestream(lines_X[i]);
      for (int j=0; j < 320; j++)
        linestream >> XPos[j];
    }
  }
  else if (outputH || outputX)
  {
    ofstream ff;
    if (outputH) 
    {
      ff.open(fnOUTH.c_str());
      ff.flags(std::ios::fixed);
      ff.precision(6);  
    }
    ofstream ffX;
    if (outputX)
    { 
      ffX.open(fnOUTX.c_str());
      ffX.flags(std::ios::fixed);
      ffX.precision(6);
    }
    int countH = 0;
    vector<string> files;
#if defined(_WIN32) || defined(WIN32)
    bool imagesReadable = ListFiles(fnDirectory, "*.jpg", files);
#endif 
    double* H1ToH2 = new double[9];
    if (imagesReadable) {
      allHomos = new double[files.size()*9];
      allX = new double[files.size()*320];
      for (vector<string>::iterator it = files.begin(); 
        it != files.end(); 
        ++it) {
          pImage1 = cvLoadImage(it->c_str());
          Checkerby checkerby(pImage1);
          checkerby.SetRowCol(5,4);
          vector<HarrisCorner> cornersRaster;
          checkerby.LabelCorners(cornersRaster);
          assert(cornersRaster.size()==80);

          vector<HarrisCorner> cornersWorld;
          for (int i = 0; i < 10; i++)
          {
            for (int j = 0; j < 8; j++)
            {
              cornersWorld.push_back(HarrisCorner(j,i));
            }
          }
          vector<std::pair<std::pair<int,int>,double> > pairs;
          for (int i=0; i < 80; i++)
          {
            pairs.push_back(pair<pair<int,int>,double>(pair<int,int>(i,i),1));
          }
          vector<std::pair<std::pair<int,int>,double> > pairs_Inlier;
          RANSACby ransac;
          ransac.runRANSAC(cornersWorld,cornersRaster, pairs, false, H1ToH2, pairs_Inlier);
          for (int i=0; i < 9; i++)
          {
            allHomos[countH*9+i]=H1ToH2[i];
          }
          for (unsigned int i=0; i < cornersRaster.size();i++)
          {
            allX[countH*320+i*4+0]=cornersWorld[i].x;
            allX[countH*320+i*4+1]=cornersWorld[i].y;
            allX[countH*320+i*4+2]=cornersRaster[i].x;
            allX[countH*320+i*4+3]=cornersRaster[i].y;
          }
          countH++;
          //delete[] H1ToH2;
          cvReleaseImage(&pImage1);
          if (outputX)
          {
            assert(cornersRaster.size()==80);
            for (unsigned int i=0; i < cornersRaster.size()-1;i++)
            {
              ffX << cornersWorld[i].x << " "
                 << cornersWorld[i].y << " "
                 << cornersRaster[i].x << " "
                 << cornersRaster[i].y << " ";
            }
            ffX << cornersWorld[cornersRaster.size()-1].x << " "
              << cornersWorld[cornersRaster.size()-1].y << " "
              << cornersRaster[cornersRaster.size()-1].x << " "
              << cornersRaster[cornersRaster.size()-1].y << endl;
          }
      }
      if (outputH)
      {
        for (int i=0; i < countH; i++)
        {
          for (int j=0; j < 8; j++)
          {
            ff << allHomos[i*9+j] <<" ";
          }
          ff << allHomos[i*9+8] <<endl;
        }
        ff.close();
      }
      if (outputX)
      {
        ffX.close();
      }
      
      numObs = countH;
    }
    delete[] H1ToH2;
  }

  double* K = NULL;
  double* allR = NULL;//r1,r2,r3,t
  double* rodrigR = NULL;
  double* distort = NULL;
  if (outputK && outputR)
  {
    //Getting intrinsic parameters
    Zhengyouby zhengyouby;
    zhengyouby.initH(allHomos,numObs);
    K = new double[9];
    zhengyouby.calcK(K);

    //getting extrinsic parameters
    allR = new double[numObs*12];//r1,r2,r3,t
    rodrigR = new double[numObs*6];
    zhengyouby.prepareRt(K);
    for (int i=0; i < numObs; i++)
    {
      zhengyouby.calcRt(allR+i*12,i,rodrigR+i*6);
      cout<<"Rt:"<<i<<endl;
      Utiliby::printMatrix(allR+i*12,3,4,0);
    }
    //zhengyouby.refineLM(allX, K, allR, rodrigR);
    distort = new double[2];
    zhengyouby.calcRadialDistortion(K,allX,distort);

    ofstream ffK;
    ofstream ffR;
    ffK.open(fnOUTK.c_str());
    ffK.flags(std::ios::fixed);
    ffK.precision(6);
    ffR.open(fnOUTR.c_str());
    ffR.flags(std::ios::fixed);
    ffR.precision(6);
    for (int i=0;i<3;i++)
      for (int j=0;j<3;j++)
      {
        ffK << K[i*3+j] << " ";
      }
    ffK << endl;
    ffK << distort[0] <<" "<< distort[1] << endl;
    ffK.close();
    for (int i=0;i<numObs;i++)
    {
      for (int j=0;j<12;j++)
        ffR<<allR[i*12+j] <<" ";
      ffR << endl;
    }
    for (int i=0;i<numObs;i++)
    {
      for (int j=0;j<6;j++)
        ffR<<rodrigR[i*6+j] <<" ";
      ffR << endl;
    }
    ffR.close();

    //delete[] K;
    //delete[] allR;
  }

  else if (inputK && inputR)
  {
    ifstream fp;
    fp.open(fnINR.c_str(), ios::in);
    vector<string>lines;
    string line;
    while(getline(fp,line))
    {
      lines.push_back(line);
    }
    allR = new double[lines.size()/2*12];
    fp.close();
    for (unsigned int i=0; i<lines.size()/2;i++)
    {
      double* RPos = allR+i*12;
      istringstream linestream(lines[i]);
      linestream >> RPos[0];
      linestream >> RPos[1];
      linestream >> RPos[2];
      linestream >> RPos[3];
      linestream >> RPos[4];
      linestream >> RPos[5];
      linestream >> RPos[6];
      linestream >> RPos[7];
      linestream >> RPos[8];
      linestream >> RPos[9];
      linestream >> RPos[10];
      linestream >> RPos[11];
    }
    rodrigR = new double[lines.size()/2*6];
    for (unsigned int i=0; i<lines.size()/2;i++)
    {
      double* RPos = rodrigR+i*6;
      istringstream linestream(lines[i+lines.size()/2]);
      linestream >> RPos[0];
      linestream >> RPos[1];
      linestream >> RPos[2];
      linestream >> RPos[3];
      linestream >> RPos[4];
      linestream >> RPos[5];
    }

    ifstream fpK;
    fpK.open(fnINK.c_str(), ios::in);
    getline(fpK,line);
    istringstream linestream(line);
    K=new double[9];
    linestream >> K[0];
    linestream >> K[1];
    linestream >> K[2];
    linestream >> K[3];
    linestream >> K[4];
    linestream >> K[5];
    linestream >> K[6];
    linestream >> K[7];
    linestream >> K[8];
    getline(fpK,line);
    istringstream linestreamd(line);
    distort=new double[2];
    linestreamd>>distort[0];
    linestreamd>>distort[1];
    fpK.close();
  }

  if (outputMATLAB)
  {
    ofstream fMatlab;
    optMode=optMode+string(".m");
    fMatlab.open(optMode.c_str());;
    fMatlab.flags(std::ios::fixed);
    fMatlab.precision(6);
    fMatlab << "xp=[";
    for (int i=0;i<numObs;i++)
    {
      for (int j=0; j<80; j++)
      {
        if (i==numObs-1 && j==79)
          fMatlab << allX[i*320+j*4+2] << ";" << allX[i*320+j*4+3] << "];" <<endl;
        else
          fMatlab << allX[i*320+j*4+2] << ";" << allX[i*320+j*4+3] << ";";
      }
    }
    fMatlab << "x=[";
    for (int i=0;i<numObs;i++)
    {
      for (int j=0; j<80; j++)
      {
        if (i==numObs-1 && j==79)
          fMatlab << allX[i*320+j*4+0] << ";" << allX[i*320+j*4+1] << "];" <<endl;
        else
          fMatlab << allX[i*320+j*4+0] << ";" << allX[i*320+j*4+1] << ";";
      }
    }
    fMatlab << "syms u0 v0 au av sk real;" << endl;
    fMatlab << "syms k;" << endl;
    fMatlab << "k=[au sk u0;0 av v0;0 0 1;];" << endl;
    for (int i=0;i<numObs;i++)
    {
      fMatlab << "syms wx" << i <<" wy" << i <<" wz" << i <<" tx" << i << " ty" << i <<" tz" << i << ";" << endl;
      fMatlab << "syms theta" << i << ";" << endl;
      fMatlab << "theta" << i << "=sqrt(wx"<< i << "^2+wy" << i << "^2+wz" << i << "^2);" << endl;
      fMatlab << "syms omega" << i << ";" << endl;
      fMatlab << "omega" << i << "=[0 -wz" << i << " wy" << i << "; wz" << i << " 0 -wx" << i << "; -wy" << i << " wx" << i << " 0;];" << endl;
      fMatlab << "syms r" << i << ";" << endl;
      fMatlab << "r" << i <<" = eye(3) + (sin(theta" << i << ")/theta"<< i << ")*omega" << i <<
        " + ((1-cos(theta" << i << "))/theta" << i << "^2)*(omega" << i << "*omega" << i << ");" << endl;
      fMatlab << "syms t" << i << ";" << endl;
      fMatlab << "t" << i << " = [tx" << i << ";ty" << i << ";tz" << i << "];" << endl;
    }
    fMatlab << "syms Fp;" << endl;
    for (int i=0; i < numObs; i++)
    {
      fMatlab << "start=160*" << i << ";" << endl;
      fMatlab << "syms tempH;" << endl;
      fMatlab << "tempH = k *[r" << i << "(:,1) r" << i << "(:,2) t" << i << "];" << endl;
      fMatlab << "for i=1:80," << endl;
      fMatlab << "    tempX=[x(start+i*2-1);x(start+i*2);1];"<<endl;
      fMatlab << "    syms uvs;" << endl;
      fMatlab << "    uvs=tempH*tempX;"<<endl;
      fMatlab << "    Fp(start+i*2-1)=uvs(1)/uvs(3);" << endl;
      fMatlab << "    Fp(start+i*2)=uvs(2)/uvs(3);" << endl;
      fMatlab << "end" << endl;
    }
    fMatlab << "h=[u0 v0 au av sk";
    for (int i=0; i < numObs; i++)
    {
      fMatlab << " wx" << i << " wy" << i << " wz" << i 
        << " tx" << i << " ty" << i << " tz" << i ;
    }
    fMatlab << "];" << endl;
    fMatlab << "H=[" << K[2] << " " << K[5] << " " << K[0] << " " << K[4] << " " << K[1];
    for (int i=0; i < numObs; i++)
    {
      fMatlab << " " << rodrigR[i*6+0] << " " << rodrigR[i*6+1] << " " << rodrigR[i*6+2]
        << " " << rodrigR[i*6+3] << " " << rodrigR[i*6+4] << " " << rodrigR[i*6+5] ;
    }
    fMatlab << "]';" << endl;
    fMatlab << "htext='u0, v0, au, av, sk,";
    for (int i=0; i < numObs; i++)
    {
      fMatlab << " wx" << i << ", wy" << i << ", wz" << i 
        << ", tx" << i << ", ty" << i << ", tz" << i << ",";
    }
    fMatlab << "';" << endl;
    fMatlab << "hdeftext='u0=H(1); v0=H(2); au=H(3); av=H(4); sk=H(5);";
    for (int i=0; i < numObs; i++)
    {
      fMatlab << " wx" << i <<"=H(" << 6+i*6+0 << "); wy" << i << "=H(" << 6+i*6+1 << "); wz" << i 
        <<"=H(" << 6+i*6+2 << "); tx" << i <<"=H(" << 6+i*6+3 << "); ty" << i 
        <<"=H(" << 6+i*6+4 << "); tz" << i << "=H(" << 6+i*6+5 << ");";
    }
    fMatlab << "';" << endl;
    fMatlab << "funchtext='H(1), H(2), H(3), H(4), H(5),";
    for (int i=0; i < numObs-1; i++)
    {
      fMatlab << " H(" << 6+i*6+0 << "), H(" << 6+i*6+1 << "), H(" << 6+i*6+2 << "), H(" << 6+i*6+3 <<
        "), H(" << 6+i*6+4 << "), H(" << 6+i*6+5 <<"),";
    }
    int i=numObs-1;
    fMatlab << " H(" << 6+i*6+0 << "), H(" << 6+i*6+1 << "), H(" << 6+i*6+2 << "), H(" << 6+i*6+3 <<
      "), H(" << 6+i*6+4 << "), H(" << 6+i*6+5 <<")";
    fMatlab << "';" << endl;
    fMatlab << "funcnewhtext='newH(1), newH(2), newH(3), newH(4), newH(5),";
    for (int i=0; i < numObs-1; i++)
    {
      fMatlab << " newH(" << 6+i*6+0 << "), newH(" << 6+i*6+1 << "), newH(" << 6+i*6+2 << "), newH(" << 6+i*6+3 <<
        "), newH(" << 6+i*6+4 << "), newH(" << 6+i*6+5 <<"),";
    }
    i=numObs-1;
    fMatlab << " newH(" << 6+i*6+0 << "), newH(" << 6+i*6+1 << "), newH(" << 6+i*6+2 << "), newH(" << 6+i*6+3 <<
      "), newH(" << 6+i*6+4 << "), newH(" << 6+i*6+5 <<")";
    fMatlab << "';" << endl;
    fMatlab << "paramtext='paramSize=" << 5+numObs*6 <<"';"<<endl;
    fMatlab << "paramSize=" << 5+numObs*6 <<";"<<endl;
    fMatlab.close();
  }

  if (outputBPM)//read in allHomo again (this time it is the output from MATLAB)
  {
    ifstream fp;
    fp.open(fnINBPM.c_str(), ios::in);
    vector<string>lines;
    string line;
    while(getline(fp,line))
    {
      lines.push_back(line);
    }
    allHomos = new double[lines.size()*9];
    numObs = lines.size();
    fp.close();
    for (unsigned int i=0; i<lines.size();i++)
    {
      double* H1ToH2 = allHomos+i*9;
      istringstream linestream(lines[i]);
      linestream >> H1ToH2[0];
      linestream >> H1ToH2[1];
      linestream >> H1ToH2[2];
      linestream >> H1ToH2[3];
      linestream >> H1ToH2[4];
      linestream >> H1ToH2[5];
      linestream >> H1ToH2[6];
      linestream >> H1ToH2[7];
      linestream >> H1ToH2[8];
    }
    assert(!fnDirectory.empty());//for the files in the directory, generate back projection, 
    //the generated backprojection is in another directory -dbp fnDirecotryBP
    assert(!fnDirectoryBP.empty());
    int countH = 0;
    vector<string> files;
#ifdef _WIN32
    bool imagesReadable = ListFiles(fnDirectory, "*.jpg", files);
#endif 
    double* tempH1ToH2 = new double[9];
    if (imagesReadable) {
      for (vector<string>::iterator it = files.begin(); 
        it != files.end(); 
        ++it) {
          pImage1 = cvLoadImage(it->c_str());
          Checkerby checkerby(pImage1);
          checkerby.SetRowCol(5,4);
          vector<HarrisCorner> cornersRaster;
          checkerby.LabelCorners(cornersRaster);
          assert(cornersRaster.size()==80);

          vector<HarrisCorner> cornersWorld;
          for (int i = 0; i < 10; i++)
          {
            for (int j = 0; j < 8; j++)
            {
              cornersWorld.push_back(HarrisCorner(j,i));
            }
          }
          vector<std::pair<std::pair<int,int>,double> > pairs;
          for (int i=0; i < 80; i++)
          {
            pairs.push_back(pair<pair<int,int>,double>(pair<int,int>(i,i),1));
          }
          vector<std::pair<std::pair<int,int>,double> > pairs_Inlier;
          RANSACby ransac;
          ransac.runRANSAC(cornersWorld,cornersRaster, pairs, false, tempH1ToH2, pairs_Inlier);
          double* goodH1ToH2 = allHomos+countH*9;
          
          IplImage* pImage1Copy = cvCloneImage(pImage1);
          checkerby.drawCorners(cornersRaster,8,CV_RGB(0,255,0),NULL,pImage1Copy);
          //checkerby.drawCorners(cornersWorld,5,CV_RGB(255,255,0),tempH1ToH2);
          checkerby.drawCorners(cornersWorld,8,CV_RGB(255,0,0),K,allR+countH*12,pImage1Copy);
          checkerby.drawCorners(cornersWorld,8,CV_RGB(255,255,0),goodH1ToH2);
          
          string fnToSave=it->substr(it->find_last_of("\\")+1,it->find(".jpg")-it->find_last_of("\\")-1);
          fnToSave=fnToSave+"_BP.jpg";
          fnToSave=fnDirectoryBP+"\\"+fnToSave;
          cvSaveImage(fnToSave.c_str(),pImage1);
          fnToSave=it->substr(it->find_last_of("\\")+1,it->find(".jpg")-it->find_last_of("\\")-1);
          fnToSave=fnToSave+"_BP_Bad.jpg";
          fnToSave=fnDirectoryBP+"\\"+fnToSave;
          cvSaveImage(fnToSave.c_str(),pImage1Copy);


          countH++;
          //delete[] H1ToH2;
          cvReleaseImage(&pImage1);
          cvReleaseImage(&pImage1Copy);
      }
    }
    delete[] tempH1ToH2;
  }

  else if (outputBPL)//use levmar
  {
    Levmarby levmarby = Levmarby(allHomos,allR,K,allX,rodrigR,distort,numObs);
    levmarby.lmOpt();
    levmarby.lmResetH();
    ofstream ff;
    ff.open(fnINBPL.c_str());
    ff.flags(std::ios::fixed);
    ff.precision(6);  
    for (int i=0; i < numObs; i++)
    {
      for (int j=0; j < 8; j++)
      {
        ff << allHomos[i*9+j] <<" ";
      }
      ff << allHomos[i*9+8] <<endl;
    }
    ff.close();
    ofstream ffK;
    ofstream ffR;
    ffK.open("K_levmar.txt");
    ffK.flags(std::ios::fixed);
    ffK.precision(6);
    ffR.open("R_levmar.txt");
    ffR.flags(std::ios::fixed);
    ffR.precision(6);
    for (int i=0;i<3;i++)
      for (int j=0;j<3;j++)
      {
        ffK << K[i*3+j] << " ";
      }
    ffK << endl;
    ffK << distort[0] <<" "<< distort[1] << endl;
    ffK.close();
    for (int i=0;i<numObs;i++)
    {
      for (int j=0;j<12;j++)
        ffR<<allR[i*12+j] <<" ";
      ffR << endl;
    }
    ffR.close();

    
    assert(!fnDirectory.empty());//for the files in the directory, generate back projection, 
    //the generated backprojection is in another directory -dbp fnDirecotryBP
    assert(!fnDirectoryBP.empty());
    int countH = 0;
    vector<string> files;
#ifdef _WIN32
    bool imagesReadable = ListFiles(fnDirectory, "*.jpg", files);
#endif 
    double* tempH1ToH2 = new double[9];
    if (imagesReadable) {
      for (vector<string>::iterator it = files.begin(); 
        it != files.end(); 
        ++it) {
          pImage1 = cvLoadImage(it->c_str());
          Checkerby checkerby(pImage1);
          checkerby.SetRowCol(5,4);
          vector<HarrisCorner> cornersRaster;
          checkerby.LabelCorners(cornersRaster);
          assert(cornersRaster.size()==80);

          vector<HarrisCorner> cornersWorld;
          for (int i = 0; i < 10; i++)
          {
            for (int j = 0; j < 8; j++)
            {
              cornersWorld.push_back(HarrisCorner(j,i));
            }
          }
          vector<std::pair<std::pair<int,int>,double> > pairs;
          for (int i=0; i < 80; i++)
          {
            pairs.push_back(pair<pair<int,int>,double>(pair<int,int>(i,i),1));
          }
          vector<std::pair<std::pair<int,int>,double> > pairs_Inlier;
          RANSACby ransac;
          ransac.runRANSAC(cornersWorld,cornersRaster, pairs, false, tempH1ToH2, pairs_Inlier);
          double* goodH1ToH2 = allHomos+countH*9;

          IplImage* pImage1Copy = cvCloneImage(pImage1);
          checkerby.drawCorners(cornersRaster,8,CV_RGB(0,255,0),NULL,pImage1);
          checkerby.drawCorners(cornersRaster,8,CV_RGB(0,255,0),NULL,pImage1Copy);
          //checkerby.drawCorners(cornersWorld,5,CV_RGB(255,255,0),tempH1ToH2);
          checkerby.drawCorners(cornersWorld,8,CV_RGB(255,0,0),K,allR+countH*12,pImage1Copy);
          checkerby.drawCorners(cornersWorld,8,CV_RGB(255,255,0),goodH1ToH2);

          string fnToSave=it->substr(it->find_last_of("\\")+1,it->find(".jpg")-it->find_last_of("\\")-1);
          fnToSave=fnToSave+"_BP.jpg";
          fnToSave=fnDirectoryBP+"\\"+fnToSave;
          cvSaveImage(fnToSave.c_str(),pImage1);
          fnToSave=it->substr(it->find_last_of("\\")+1,it->find(".jpg")-it->find_last_of("\\")-1);
          fnToSave=fnToSave+"_BP_Bad.jpg";
          fnToSave=fnDirectoryBP+"\\"+fnToSave;
          cvSaveImage(fnToSave.c_str(),pImage1Copy);

          countH++;
          //delete[] H1ToH2;
          cvReleaseImage(&pImage1);
          cvReleaseImage(&pImage1Copy);
      }
    }
    delete[] tempH1ToH2;
  }

  if (allHomos != NULL)
    delete[] allHomos;
  if (allX != NULL)
    delete[] allX;
  if (K != NULL)
    delete[] K;
  if (allR != NULL)
    delete[] allR;
  if (rodrigR != NULL)
    delete[] rodrigR;
  if (distort != NULL)
    delete[] distort;

}
