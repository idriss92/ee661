#ifndef __HARRISBY_h__
#define __HARRISBY_h__

#include <cv.h>
#include <vector>

using namespace std;

namespace CVisby{

  class HarrisCorner{
  public:
    int x;
    int y;
    HarrisCorner(int x, int y):x(x),y(y){};
    inline double operator - (const HarrisCorner& other){return sqrt(double((x-other.x)*(x-other.x)+(y-other.y)*(y-other.y)));};
    inline HarrisCorner & operator = (const HarrisCorner& other){x = other.x; y = other.y;return *this;};
  };

  class Harrisby{
  public:
    int HarrisRadius;
    int HarrisLen;
    int HarrisWindow;
    int HarrisDist;
    int MatchingRadius;
    int MatchingLen;
    int MatchingWindow;
    double MatchingRange;

    Harrisby(){
      setWindowSize(2);
      setMatchingSize(5);
    };
    void inline setWindowSize(int m){
      HarrisRadius = m;
      HarrisLen = 2*m + 1;
      HarrisWindow =HarrisLen * HarrisLen;
      HarrisDist = 2*HarrisLen;
    };

    void inline setMatchingSize(int m){
      MatchingRadius = m;
      MatchingLen = 2*m + 1;
      MatchingWindow =MatchingLen * MatchingLen;
      //MatchingRange = 60;
      MatchingRange = 0.15;
    };
    
    void detectHarrisCorner(IplImage *pImage, vector<HarrisCorner>& corners, 
      double threshold = 0.1, IplImage *pImageOut = NULL); // pImage is gray.
    void matchSSD(IplImage *pImage1, IplImage *pImage2, const vector<HarrisCorner>& corners1, 
      const vector<HarrisCorner>& corners2, vector<std::pair<int,int> >& pairs, 
      double distThres = 1);
    void matchNCC(IplImage *pImage1, IplImage *pImage2, const vector<HarrisCorner>& corners1, 
      const vector<HarrisCorner>& corners2, vector<std::pair<int,int> >& pairs, 
      double distThres = 1);
    void matchNCC(IplImage *pImage1, IplImage *pImage2, const vector<HarrisCorner>& corners1, 
      const vector<HarrisCorner>& corners2, vector<std::pair<std::pair<int,int>,double> >& pairs, 
      double distThres = 1);
    void drawCombinedImage(IplImage *pImage1, IplImage *pImage2, IplImage *pImageOut,
      const vector<HarrisCorner>& corners1, 
      const vector<HarrisCorner>& corners2, 
      const vector<std::pair<int,int> >& pairs);
    void drawCombinedImage(IplImage *pImage1, IplImage *pImage2, IplImage *pImageOut,
      const vector<HarrisCorner>& corners1, 
      const vector<HarrisCorner>& corners2, 
      const vector<std::pair<std::pair<int,int>,double> >& pairs
      ,bool useRandomColor=false);
  
  private:

  };

}

#endif
