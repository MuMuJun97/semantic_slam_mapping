/*
IRTES-SET laboratory
Authors: You Li (liyou026@gmail.com)
Descirption: This a sample code of my PhD works
*/

#include "vo.hpp"
#include <cmath>

using namespace std;

VisualOdometry::VisualOdometry (parameters param) : param(param) {
  J         = 0;
  p_observe = 0;
  p_predict = 0;
  Tr_delta = cv::Mat::eye(4,4,CV_64F);
  srand(0);
}

VisualOdometry::~VisualOdometry () {
}

bool VisualOdometry::updateMotion ()
{
  double time = (double)cv::getTickCount();

  // estimate motion
  vector<double> tr_delta = estimateMotion(quadmatches);
  
  // on failure
  if (tr_delta.size()!=6)
    return false;
  
  // set transformation matrix (previous to current frame)
  Tr_delta = transformationVectorToMatrix(tr_delta);

  time = ((double)cv::getTickCount() - time)/cv::getTickFrequency()*1000;
  //cout<<"The odometry estimation costs "<<time<<" ms"<<endl;

  // success
  return true;
}


cv::Mat VisualOdometry::transformationVectorToMatrix (std::vector<double> tr) {

  // extract parameters
  double rx = tr[0];
  double ry = tr[1];
  double rz = tr[2];
  double tx = tr[3];
  double ty = tr[4];
  double tz = tr[5];

  // precompute sine/cosine
  double sx = sin(rx);
  double cx = cos(rx);
  double sy = sin(ry);
  double cy = cos(ry);
  double sz = sin(rz);
  double cz = cos(rz);

  // compute transformation
  cv::Mat Tr(4,4,CV_64FC1,cv::Scalar(0));

  Tr.at<double>(0,0) = +cy*cz;Tr.at<double>(0,1) = -cy*sz; Tr.at<double>(0,2) = +sy;Tr.at<double>(0,3) = tx;
  Tr.at<double>(1,0) = +sx*sy*cz+cx*sz; Tr.at<double>(1,1) = -sx*sy*sz+cx*cz; Tr.at<double>(1,2) = -sx*cy; Tr.at<double>(1,3)= ty;
  Tr.at<double>(2,0) = -cx*sy*cz+sx*sz; Tr.at<double>(2,1) = +cx*sy*sz+sx*cz; Tr.at<double>(2,2) = +cx*cy; Tr.at<double>(2,3) = tz;
  Tr.at<double>(3,0) = 0;              Tr.at<double>(3,1) = 0;               Tr.at<double>(3,2) = 0;      Tr.at<double>(3,3) = 1;

  return Tr;
}

vector<int> VisualOdometry::getRandomSample(int N,int num) {

  // init sample and totalset
  vector<int> sample;
  vector<int> totalset;
  
  // create vector containing all indices
  for (int i=0; i<N; i++)
    totalset.push_back(i);

  // add num indices to current sample
  sample.clear();
  for (int i=0; i<num; i++) {
    int j = rand()%totalset.size();
    sample.push_back(totalset[j]);
    totalset.erase(totalset.begin()+j);
  }
  
  return sample;
}
