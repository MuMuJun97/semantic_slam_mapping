/*
   IRTES-SET laboratory
   Authors: You Li (liyou026@gmail.com)
   Descirption: This a sample code of my PhD works
*/

#ifndef UVDisparity__HPP
#define UVDisparity__HPP

#include <iostream>
#include "stereo.h"
#include "vo_stereo.hpp"

using namespace std;

//parameters in segmentation of U-disparity image
struct USegmentPars
{
  USegmentPars():min_intense(32), min_disparity_raw(64), min_area(40){};

  USegmentPars(int min_intense_, int min_disparity_raw_, int min_area_);

  inline USegmentPars& operator=(const USegmentPars& t)
    {
      min_intense = t.min_intense;
      min_disparity_raw = t.min_disparity_raw;
      min_area = t.min_area;
      return *this;
    }

  int min_intense;       // minum gray intensity for starting segmentation
  int min_disparity_raw; //minmm raw disparity for starting segmentation
  int min_area;          //minum area required for candidate mask
};


// intergrate a bundle of methods in U-V disparity image understanding
class UVDisparity
{
  public:
    //constructor and deconstructor
    UVDisparity();
    ~UVDisparity();

    // initialization functions
    inline void SetCalibPars(CalibPars& calib_par)
    {
        calib_ = calib_par;
    }
    inline void SetROI3D(ROI3D& roi_3d)
    {
        roi_ = roi_3d;
    }
    inline void SetUSegmentPars(int min_intense, int min_disparity_raw, int min_area)
    {
        this->u_segment_par_.min_intense = min_intense;
        this->u_segment_par_.min_disparity_raw = min_disparity_raw;
        this->u_segment_par_.min_area = min_area;
    }
    inline void SetOutThreshold(double out_th)
    {
        out_th_ = out_th;
    }  // set outlier threshold

    inline void SetInlierTolerance(int inlier_tolerance)
    {
        inlier_tolerance_ = inlier_tolerance;
    } //set inlier tolerance

    inline void SetMinAdjustIntense(int min_adjust_intense)
    {
        min_adjust_intense_= min_adjust_intense;
    } //set minimum adjust intensity for segmentation



    /**UV disparity segmentation:
    return value: the segmentation result
    **/
   cv:: Mat Process(cv::Mat& img_L, cv::Mat& disp_sgbm,           //input image and disparity map
                    VisualOdometryStereo& vo,     //input visual odoemtry(inlier&outlier),motion
                    cv::Mat& xyz, cv::Mat& roi_mask, Mat &ground_mask,
                    double& pitch1, double& pitch2);
private:


    /*** Calculate U-disparity image***/
    void calUDisparity(const cv::Mat& img_dis, Mat& xyz, Mat &roi_mask, Mat &ground_mask);

    /**Calculate V-disparity image**/
    void calVDisparity(const cv::Mat& img_dis, Mat & xyz);

    /**
     *Estimate the pitch angle from V disparity map
     *Remove ground plane the points outside the ROI3D
    **/
    vector<cv::Mat> Pitch_Classify(cv::Mat& xyz, Mat &ground_mask);

    //filtering the matched feature points by ego-motion
    void filterInOut(const cv::Mat& img_L, const cv::Mat& roi_mask, const Mat &sgbm_roi,
                     VisualOdometryStereo& vo, const double pitch_);

    //find all the candidates
    void findAllMasks(const VisualOdometryStereo& vo,const cv::Mat& img_L,cv::Mat& xyz, cv::Mat& roi_mask);

    //Segmentation in disparity map
    void segmentation(const cv::Mat& disparity, const cv::Mat& img_L, cv::Mat& roi_mask, cv::Mat& mask_moving);
    void confirmed();

    //sigmoid function to adjust the intensity of U-disparity
    double sigmoid(double t,double scale,double range, int mode);
    void adjustUdisIntense(double scale, double range);

    //judge functions for segments merging
    bool isMasksSeparate();
    bool isOverlapped(const cv::Mat &mask1, const cv::Mat &mask2);
    bool isAllZero(const cv::Mat& mat);
    bool isInMask(int u, int v, const cv::Mat& roi_mask);
    int numInlierInMask(const cv::Mat & mask, const VisualOdometryStereo &vo, const cv::Mat &img_L);
    void mergeMasks(); //merge overlapped segments

    //improve the results by inliers
    void verifyByInliers(const VisualOdometryStereo &vo, const cv::Mat &img_L);




private:

    int inlier_tolerance_;             //inlier tolerance threshold
    int min_adjust_intense_;           //minum intensity for U adjustment
    cv::Mat pitch1_measure,pitch2_measure;
    double out_th_;                    //outlier rejection threshold
    double pitch_;                     //pitch_ value of the ground plane
    cv::KalmanFilter* pitch1_KF,*pitch2_KF;
    ROI3D roi_;                        //3D space ROI parameters
    CalibPars calib_;                   //camera calibration parameters
    vector< vector<cv::Mat> > candidates_;
    USegmentPars u_segment_par_;        //Disparity segmentation parameters

    vector<cv::Mat> masks_;            // preliminary masks_ for the moving object

    cv::Mat u_dis_int,u_dis_,u_dis_show;            // u_dis_int is the original u-disparity map
    cv::Mat v_dis_int,v_dis_,v_dis_show;            // v_dis_int is the original v-disparity map
    vector<cv::Mat> masks_confirmed_;
};

#endif
