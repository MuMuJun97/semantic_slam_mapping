/*
   IRTES-SET laboratory
   Authors: You Li (liyou026@gmail.com)
   Descirption: This a sample code of my PhD works
*/

#ifndef VO_STEREO_H
#define VO_STEREO_H

#include "vo.hpp"
#include <algorithm>
#include "opencv2/core/core.hpp"
#include <quadmatcher.hpp>
using namespace std;

class VisualOdometryStereo : public VisualOdometry
{

public:

 
// stereo-specific parameters (mandatory: base)
  struct parameters : public VisualOdometry::parameters
  {
    double  base;             // baseline (meters)
    int ransac_iters;         // number of RANSAC iterations
    double  inlier_threshold; // fundamental matrix inlier threshold
    bool    reweighting;      // lower border weights (more robust to calibration errors)
    parameters ()
    {
      base             = 1.0;
      ransac_iters     = 200;
      inlier_threshold = 1.1f;
      reweighting      = true;
    }
  };

  // constructor, takes as inpute a parameter structure
  VisualOdometryStereo (parameters param);
  
  // deconstructor
  ~VisualOdometryStereo ();
  
  // process a new images, push the images back to an internal ring buffer.
  // valid motion estimates are available after calling process for two times.
  // inputs: I1 ........ pointer to rectified left image (uint8, row-aligned)
  //         I2 ........ pointer to rectified right image (uint8, row-aligned)
  //         dims[0] ... width of I1 and I2 (both must be of same size)
  //         dims[1] ... height of I1 and I2 (both must be of same size)
  //         dims[2] ... bytes per line (often equal to width)
  //         replace ... replace current images with I1 and I2, without copying last current
  //                     images to previous images internally. this option can be used
  //                     when small/no motions are observed to obtain Tr_delta wrt
  //                     an older coordinate system / time step than the previous one.
  // output: returns false if an error occured
  bool Process (QuadFeatureMatch& quadmatches);

private:

  std::vector<double>  estimateMotion (std::vector<pmatch>& quadmatches);
  //allocate the inliers and outliers of the matched feature points
   void getInOutMatches(std::vector<pmatch>& quadmatches, vector<int>& inliers);

 // given inlier and outlier the disparity for the upcoming segmentation
   void givenMatchesDisparity(cv::Mat& disparity);

  enum                 result { UPDATED, FAILED, CONVERGED };  
  result               updateParameters(std::vector<pmatch>& quadmatches,vector<int> &active, vector<double> &tr,double step_size,double eps);
  void                 computeObservations(std::vector<pmatch> &quadmatches, vector<int> &active);
  void                 computeResidualsAndJacobian(std::vector<double> &tr,std::vector<int> &active);
  std::vector<int> getInlier(std::vector<pmatch>& quadmatches,std::vector<double> &tr);
  double *X,*Y,*Z;    // 3d points
  double *p_residual; // residuals (p_residual=p_observe-p_predict)
  
  // parameters
  parameters param;
};

#endif // VO_STEREO_H

