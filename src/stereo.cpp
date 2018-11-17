#include "stereo.h"
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

/*---------  Calculate the disparity may by SGBM algorithm
* img_L, img_R ----------- Rectified left and right image 
* disp         ----------- The disparity result
*/
void calDisparity_SGBM(const cv::Mat& img_L, const cv::Mat& img_R, cv::Mat& disp)
{
	cv::StereoSGBM sgbm;

	// set the parameters of sgbm
	int cn = 1; //number of channels
	int SADWindowSize = 11; //Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
	int numberOfDisparities = 80; //Maximum disparity minus minimum disparity, must be n*16
	sgbm.minDisparity = 0; //Minimum possible disparity value
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = 100;
	sgbm.preFilterCap = 63; //Truncation value for the prefiltered image pixels
	sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;
	sgbm.P1 = 4*cn*sgbm.SADWindowSize*sgbm.SADWindowSize; //controlling the disparity smoothness. P2 > P1
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize; //controlling the disparity smoothness.The larger the values are, the smoother the disparity is
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.speckleRange = 32; // devided by 16, 1 or 2 is good enough
	sgbm.disp12MaxDiff = 1;

	sgbm(img_L, img_R, disp);    
/*
	cv::Mat disp8; //scaling the value into 0-255
	disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
	cv::Mat img_equalize,img_color;
	cv::equalizeHist(disp8,img_equalize);
	imshow("disp8", disp8);
*/
}


void triangulate10D(const cv::Mat& img, const cv::Mat& disp, cv::Mat& xyz,
                   const double f, const double cx, const double cy, const double b, ROI3D roi)
{
    //test
    int stype = disp.type();
    int dtype = CV_32FC3;
    //CV_Assert(stype == CV_16SC1);
    xyz.create(disp.size(),CV_MAKETYPE(dtype,10));

    //assign the effective elements of Q matrix
    int rows = disp.rows;
    int cols = disp.cols;

    double px,py,pz;

    //handling the missing values
    double minDisparity = FLT_MAX;
    cv::minMaxIdx(disp, &minDisparity, 0, 0, 0 );

    std::vector<float> _dbuf(cols*3+1);

    int x_max = roi.x_max;
    int y_max = roi.y_max;
    int z_max = roi.z_max;

    for(int i = 0;i<rows;i++)
    {
        const uchar* gray_ptr = img.ptr<uchar>(i);
        const short* disp_ptr = disp.ptr<short>(i);
        //float *dptr = dbuf;

        float *dptr = xyz.ptr<float>(i);

        for (int j = 0; j < cols; j++)
        {
            uchar intensity = gray_ptr[j];
            short d = disp_ptr[j];
            double pw = b/(1.0*static_cast<double>(d));
            px = ((static_cast<double>(j) -cx)*pw)*16.0f;
            py = ((static_cast<double>(i) -cy)*pw)*16.0f;
            pz = (f*pw)*16.0f;

            if (fabs(d-minDisparity) <= FLT_EPSILON )
            {
                px = 1.0/0.0;
                py = 1.0/0.0;
                pz = 1.0/0.0;
            }

            if (fabs(px)>x_max || fabs(pz)>z_max || fabs(py)>y_max) //outside the ROI
            {
                dptr[j*10]     = (float)px; //X
                dptr[j*10 + 1] = (float)py; //Y
                dptr[j*10 + 2] = (float)pz; //Z
                dptr[j*10 + 3] = (float)j;  //u
                dptr[j*10 + 4] = (float)i;  //v
                dptr[j*10 + 5] = (float)d/16.0f; //disparity
                dptr[j*10 + 6] = (int)intensity; //intensity
                dptr[j*10 + 7] = 0;        //I_u
                dptr[j*10 + 8] = 0;        //I_v
                dptr[j*10 + 9] = 0;        //motion mark
            }
            else //in the ROI
            {
                dptr[j*10]     = (float)px;      //X
                dptr[j*10 + 1] = (float)py;      //Y
                dptr[j*10 + 2] = (float)pz;      //Z
                dptr[j*10 + 3] = (float)j;       //u
                dptr[j*10 + 4] = (float)i;       //v
                dptr[j*10 + 5] = (float)d/16.0f; //disparity
                dptr[j*10 + 6] = (int)intensity; //intensity
                dptr[j*10 + 7] = 0;           //I_u
                dptr[j*10 + 8] = 0;           //I_v
                dptr[j*10 + 9] = 0;           //motion mark
            }
        }
    }
}



/*
******this function rectify the coordinates of the 3D point cloud by the estimated pitch angle to the ground
******xyz --- the point cloud
******pitch1 --- the first estimated pitch value
 */
void correct3DPoints(cv::Mat& xyz, ROI3D& roi_, const double& pitch1, const double& pitch2)
{

  double cos_p1 = cos(pitch1);
  double sin_p1 = sin(pitch1);

  int cols = xyz.cols;
  int rows = xyz.rows;
    
  for(int j = 0; j < rows; j++)
  {

      float* xyz_ptr = xyz.ptr<float>(j);
      
      for(int i = 0;i < cols; i++)
      {
        float xp = xyz_ptr[10*i];
        float yp = xyz_ptr[10*i+1];
        float zp = xyz_ptr[10*i+2];

        int d = cvRound(xyz_ptr[10*i+5]);

        if(d < 25 && d>0)
        {
            xyz_ptr[10*i] = xp;
            xyz_ptr[10*i+1] = cos_p1 * yp + sin_p1 * zp;
            xyz_ptr[10*i+2] = cos_p1 * zp - sin_p1 * yp;

            if(xyz_ptr[10*i] > roi_.x_max || xyz_ptr[10*i+1] > roi_.y_max || xyz_ptr[10*i+2]>roi_.z_max) //outside the ROI
            {
                xyz_ptr[10*i+6] = 0;
            }


        }
        else if(d >= 25 && d<100)
        {
            xyz_ptr[10*i] = xp;
            xyz_ptr[10*i+1] = cos_p1 * yp + sin_p1 * zp;
            xyz_ptr[10*i+2] = cos_p1 * zp - sin_p1 * yp;

            if(xyz_ptr[10*i] > roi_.x_max || xyz_ptr[10*i+1] > roi_.y_max || xyz_ptr[10*i+2] > roi_.z_max) //outside the ROI
            {
                xyz_ptr[10*i+6] = 0;
            }
        }
        else
        {
             xyz_ptr[10*i+6] = 0;
        }

      }
    }

}

void setImageROI (cv::Mat& xyz, cv::Mat& roi_mask)
{
      vector<Mat> channels(8);
      split(xyz, channels);
      cv::Mat ch6;
      ch6 = channels[6];

      roi_mask.create(ch6.size(),CV_8UC1);
      cv::convertScaleAbs(ch6,roi_mask);
}
