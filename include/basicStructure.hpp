#ifndef BASICSTRUCTURE__HPP
#define BASICSTRUCTURE__HPP
#include <iostream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"


using namespace std;

//Region of interest in 3D. (usually, we chose a 30X30 meteres wide area with 1m higher than the vision system )
struct ROI3D
{
  ROI3D():x_max(30000),y_max(-1000),z_max(30000){};

  ROI3D(double x, double y, double z)
    {
      x_max = x;
      y_max = y;
      z_max = z;
      
    };

  inline ROI3D& operator=(const ROI3D &t)
    {
      x_max = t.x_max;
      y_max = t.y_max;
      z_max = t.z_max;
      return *this;
    };
  
  double x_max;
  double y_max;//is negtive always
  double z_max;
};


//stereo calibration parameters
struct CalibPars
{ 

    CalibPars():f(0.0),c_x(0.0),c_y(0.0),b(0.0){};

    CalibPars(const double _f, const double _cx, const double _cy, const double _base)
    {
        f = _f;
        c_x = _cx;
        c_y = _cy;
        b = _base;
     };

    CalibPars(const cv::Mat& Q)
    {
      int type = Q.elemSize1();
      if(type == 8)
      {
        f = Q.at<double>(2,3);
        c_x = (-1.0)*Q.at<double>(0,3);
        c_y = (-1.0)*Q.at<double>(1,3);
        b = (-1.0)/Q.at<double>(3,2);
      }
      else
      {
        cout<<"Only double type matrix is allowed!"<<endl;
      }
             
    };



    inline friend ostream &operator <<(ostream &s, CalibPars calib)
    {
      s<<"Focal length f: "<<calib.f<<" c_x, c_y: "<<calib.c_x<<" "<<calib.c_y<<" base length: "<<calib.b<<endl;
      return s;
    };

    inline CalibPars& operator=(const CalibPars &t)
    {
      f = t.f;
      c_x = t.c_x;
      c_y = t.c_y;
      b = t.b;
      return *this;
    };


    double f;      //focal length
    double c_x;    //principle position in x,y
    double c_y;
    double b;      //base line in mm
};


#endif
