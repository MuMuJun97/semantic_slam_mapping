#ifndef STEREO__HPP
#define STEREO__HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "basicStructure.hpp"

/*---------  Calculate the disparity may by semi-global block matching (SGBM) algorithm
* img_L, img_R ----------- Rectified left and right image 
* disp         ----------- The disparity result
*/
void calDisparity_SGBM(const cv::Mat& img_L, const cv::Mat& img_R, cv::Mat& disp);


/*---------  3D reconstruction by triangulation and create a 10-dimensional matrix xyz
* img          ----------- left image
* disp         ----------- The disparity result
* xyz          ----------- Output, 10 dimensional matrix contains 1, X, 2, Y, 3,Z, 4, u, 5,v, 6, disparity, 7, intensity, 8, intesity in Udisparity, 9 intensity in Vdisparity, 10, motion mark
* f, cx, cy, b ----------- Calibration parameters
* roi          ----------- Region of Interest
*/
void triangulate10D(const cv::Mat &img, const cv::Mat& disp, cv::Mat& xyz,
                    const double f, const double cx, const double cy, const double b, ROI3D roi);


/* ------------- Rectify the 3D reconstruction results by the estimated pitch angle
*
* Input: xyz       ------- the point cloud
* Input: pitch     ------- the estimated pitch value
* Output: xyz_new  ------- rectified points cloud
* pitch1, pitch2   ------- the two estimated pitch angle from V-disparity image
*/
void correct3DPoints(cv::Mat& xyz, ROI3D &roi_, const double& pitch1, const double &pitch2);

/* ------------- set image ROI according to the 3D reconstruction results and ROI3D
*
* Input: xyz         ------- the point cloud
* Output: roi_mask   -------
*/
void setImageROI ( cv::Mat& xyz, cv::Mat &roi_mask);

#endif
