/*
   IRTES-SET laboratory
   Authors: You Li (liyou026@gmail.com)
   Descirption: This a sample code of my PhD works
*/

#include "uvdisparity.hpp"
#include "stereo.h"
using namespace cv;
using namespace std;








USegmentPars::USegmentPars(int min_intense_, int min_disparity_raw_, int min_area_)
{
    min_intense = min_intense_;
    min_disparity_raw = min_disparity_raw_;
    min_area = min_area_;
}

//constructer
 UVDisparity::UVDisparity()
 {
     out_th_ = 6.0f;                    //outlier rejection threshold
     inlier_tolerance_ = 3;             //inlier tolerance threshold
     min_adjust_intense_ = 19;          //minum adjust intensity


     //init Kalman filter parameters
     pitch1_KF = new KalmanFilter(2,1,0);
     pitch1_KF->transitionMatrix = *(Mat_<float>(2, 2) << 1,0,0,1);
     cv::setIdentity(pitch1_KF->measurementMatrix);
     cv::setIdentity(pitch1_KF->processNoiseCov, Scalar::all(0.000005));
     cv::setIdentity(pitch1_KF->measurementNoiseCov, Scalar::all(0.001));
     cv::setIdentity(pitch1_KF->errorCovPost, Scalar::all(1));

     pitch2_KF = new KalmanFilter(2,1,0);
     pitch2_KF->transitionMatrix = *(Mat_<float>(2, 2) << 1,0,0,1);
     cv::setIdentity(pitch2_KF->measurementMatrix);
     cv::setIdentity(pitch2_KF->processNoiseCov, Scalar::all(0.000005));
     cv::setIdentity(pitch2_KF->measurementNoiseCov, Scalar::all(0.001));
     cv::setIdentity(pitch2_KF->errorCovPost, Scalar::all(1));

     pitch1_measure.create(1,1,CV_32F);
     pitch2_measure.create(1,1,CV_32F);


 }

 //deconstructor
 UVDisparity::~UVDisparity()
 {
     delete pitch1_KF,pitch2_KF;
 }


 /*Since the reconstructed 3D points sometimes not correspond the
  *triangulated coordinates (the reason is disparity map is smoothed
  *while the disparity between two matched points is not smoothed),
  *we re-calculate the 3D position, and filter out the inliers outsid
  *e the ROI
*/
void UVDisparity::filterInOut(const Mat &image, const Mat &roi_mask,const Mat &sgbm_roi,
                              VisualOdometryStereo& vo, const double pitch)
{
  /*calibration parameters*/
  double f = calib_.f;
  double cu = calib_.c_x;
  double cv = calib_.c_y;
  double base = calib_.b;

  double cos_p = cos(pitch);
  double sin_p = sin(pitch);
  int threshold = -3000;

  cv::Mat motion1 = vo.getMotion();
  cv::Mat show_in,show_out,xyz0,xyz1;
  
  cvtColor(image, show_in, CV_GRAY2BGR);
  cvtColor(image, show_out, CV_GRAY2BGR);

  vector<pmatch>::iterator it_in = vo.quadmatches_inlier.begin();
  //cout<<"the inliers: "<<vo.quadmatches_inlier.size()<<endl;


  for(; it_in!=vo.quadmatches_inlier.end(); )
  {
    /*current feature point position*/
    int uc = (*it_in).u1c;
    int vc = (*it_in).v1c;
  
    /*previous feature point position*/
    int up = (*it_in).u1p;
    int vp = (*it_in).v1p;
  
    if(roi_mask.at<uchar>(vc,uc) > 0)
    {
      double d = max(uc - (*it_in).u2c, 1.0f);
      double yc = (vc - cv)*base/d;
      double zc = f*base/d;

      yc = cos_p * yc + sin_p * zc;
  
       if(true)
      {
        cv::circle(show_in, cv::Point(uc,vc),2,cv::Scalar(0,0,255),2,8,0);
        cv::circle(show_in, cv::Point(up,vp),2,cv::Scalar(0,0,255),2,8,0);
        cv::line(show_in,  cv::Point(up,vp),  cv::Point(uc,vc), cv::Scalar(0,255,0),1,8,0);

        (*it_in).dis_c = sgbm_roi.at<short>(vc,uc);

        it_in++;
      }
       else
       {
         it_in = vo.quadmatches_inlier.erase(it_in);
       }
       
    }
    else
    {
      it_in = vo.quadmatches_inlier.erase(it_in);
    }
  }
  //cout<<"After the filter process, the number of inlier is: "<<vo.p_matched_inlier.size()<<endl;
  vector<pmatch>::iterator it_out = vo.quadmatches_outlier.begin();
 // cout<<"before the filter process, the number of outlier is: "<<vo.p_matched_outlier.size()<<endl;
  for(; it_out!=vo.quadmatches_outlier.end(); )
  {
    
    int uc = (*it_out).u1c;
    int vc = (*it_out).v1c;
    //int vc2 = (*it).v2c;

    int up = (*it_out).u1p;
    int vp = (*it_out).v1p;
      
    if(roi_mask.at<uchar>(vc,uc) > 0)
    {
      double d = max(uc - (*it_out).u2c, 1.0f);            
      double xc = (uc - cu)*base/d;
      double yc = (vc - cv)*base/d;
      double zc = f*base/d;
      
      yc = cos_p * yc + sin_p * zc;
      zc = cos_p * zc - sin_p * yc;
            
      if(xc > threshold)
      {
        cv::circle(show_in, cv::Point(uc,vc),2,cv::Scalar(255,0,0),2,8,0);
        cv::circle(show_in, cv::Point(up,vp),2,cv::Scalar(255,0,0),2,8,0);
        cv::line(show_in,  cv::Point(up,vp),  cv::Point(uc,vc), cv::Scalar(0,255,0),1,8,0);
        (*it_out).dis_c = sgbm_roi.at<short>(vc,uc);

        it_out++;
      }
      else
      {
        it_out = vo.quadmatches_outlier.erase(it_out);
      }
      
    }

    else
    {
      it_out = vo.quadmatches_outlier.erase(it_out);
    }
    
  }
  
  /* show the inliers and outliers in one picture */
  if(true)
  {
     //cv::imshow("show_in",show_in);
     //cv::waitKey(0);

  }

  if(false)
  {
      cv::imshow("show_in",show_in);
      cv::waitKey(0);
  }

}




void UVDisparity::calUDisparity(const cv::Mat& img_dis, cv::Mat& xyz,cv::Mat& roi_mask,cv::Mat& ground_mask)
{
  double max_dis = 0.0;
  cv::minMaxIdx (img_dis,NULL,&max_dis,NULL,NULL);
  max_dis = max_dis/16;//the stereosgbm algorithm amplifies the real disparity value by 16
      
  int d_cols = img_dis.cols;int d_rows = img_dis.rows;
  int u_rows = cvCeil(max_dis)+1;int u_cols = img_dis.cols;
    
  //allocate the memory for v-disparity map and initialize it as all zeros
  u_dis_int = Mat::zeros(u_rows, u_cols, CV_32SC1);

    for(int i = 0;i<d_rows;i++)
    {
      //const uchar* gray_ptr = img_rgb.ptr<uchar>(i);
      const short* disp_ptr = img_dis.ptr<short>(i);
 
      for(int j = 0; j<d_cols; j++)
      {
        short d = disp_ptr[j];

        if(!cvIsInf(d) && !cvIsNaN(d) && d > 0)
        {
          int dis = cvRound(d/16);
          //set value for the udisparity map
          int* udis_ptr = u_dis_int.ptr<int>(dis);

          if(roi_mask.at<uchar>(i,j) > 0 && ground_mask.at<uchar>(i,j) > 0 && dis > 0)
          {
             udis_ptr[j]++;
          }

        }
        
      }
        
    }

   //assign to the uchar u-disparity map
    u_dis_.create(u_dis_int.size(),CV_8UC1);

    float scale = 255*1.0f/xyz.rows;

    for(int i = 0;i < u_rows;i++)
    {
        //const uchar* gray_ptr = img_rgb.ptr<uchar>(i);
        const int* u_ptr = u_dis_int.ptr<int>(i);
        uchar* u_char_ptr = u_dis_.ptr<uchar>(i);

        for(int j = 0; j < u_cols; j++)
        {
            int u = u_ptr[j];
            u_char_ptr[j] = u*scale;
        }
      }

    //assign for visualize
    cvtColor(u_dis_, u_dis_show, CV_GRAY2BGR);

//    cv::imshow("color",u_dis_);
//    cv::waitKey(0);

    int xyz_cols = xyz.cols;
    int xyz_rows = xyz.rows;

    for(int j = 0; j < xyz_rows; j++)
    {
        float* xyz_ptr = xyz.ptr<float>(j);
        for(int i = 0;i < xyz_cols; i++)
        {

             int u = cvRound(xyz_ptr[10*i+3]);
             int d = cvRound(xyz_ptr[10*i+5]);
             int u_i = u_dis_.at<uchar>(d,u);
             xyz_ptr[10*i+7] = u_i;
        }
    }
  // GaussianBlur(u_dis_,u_dis_,Size(3,3),0,0);
  //cv::imwrite("U_disparity.png",u_dis_);
}


void UVDisparity::calVDisparity(const cv::Mat& img_dis,cv::Mat& xyz)
{
  double max_dis = 0.0;
  cv::minMaxIdx (img_dis,NULL,&max_dis,NULL,NULL);
  //cout<<"the max disparity is: "<<max_dis/16<<endl;
  max_dis = max_dis/16;//the stereosgbm algorithm amplifies the real disparity value by 16
  //cout<<"the maxim disparity is: "<<max_dis<<endl;
    
  int d_cols = img_dis.cols;
  int d_rows = img_dis.rows;

  int v_cols = cvCeil(max_dis);
  int v_rows = img_dis.rows;

   
  //allocate the memory for v-disparity map and initialize it as all zeros
   v_dis_int = Mat::zeros(v_rows, v_cols, CV_32SC1);

  for(int i = 0;i<d_rows;i++)
    {
      //const uchar* gray_ptr = img_rgb.ptr<uchar>(i);
      const short* disp_ptr = img_dis.ptr<short>(i);
      int* vdis_ptr = v_dis_int.ptr<int>(i);
 
      for(int j = 0; j<d_cols; j++)
      {
        short d = disp_ptr[j];

        if(!cvIsInf(d) && !cvIsNaN(d) && d > 0)
        {
          int dis = cvRound(d/16.0f);
          int id = max(0,min(v_cols,dis));

          vdis_ptr[id]++;
        }
      }
    }


  int xyz_cols = xyz.cols;
  int xyz_rows = xyz.rows;

  float scale = 255*1.0f/xyz.cols;

  //assign the int matrix to uchar matrix for visualize
  v_dis_.create(v_dis_int.size(),CV_8UC1);
  for(int i = 0;i < v_rows;i++)
  {
      //const uchar* gray_ptr = img_rgb.ptr<uchar>(i);
      const int* v_ptr = v_dis_int.ptr<int>(i);
      uchar* v_char_ptr = v_dis_.ptr<uchar>(i);

      for(int j = 0; j < v_cols; j++)
      {
          int v = v_ptr[j];
          v_char_ptr[j] = v*scale;
      }
    }

  //assign for visualize
  cvtColor(v_dis_, v_dis_show, CV_GRAY2BGR);

//  cv::imshow("v_dis",v_dis_uchar);
//  cv::waitKey(0);

  //assign to xyz 10D matrix
  for(int j = 0; j < xyz_rows; j++)
  {
      float* xyz_ptr = xyz.ptr<float>(j);
      for(int i = 0;i < xyz_cols; i++)
      {

           int v = cvRound(xyz_ptr[10*i+4]);
           int d = cvRound(xyz_ptr[10*i+5]);

           if(d>0)
           {
             int v_i = v_dis_.at<int>(v,d);
             xyz_ptr[10*i+8] = (float)v_i;
           }
           else
           {
             xyz_ptr[10*i+8] = 0;
           }

      }
  }


}

vector<cv::Mat> UVDisparity::Pitch_Classify(cv::Mat &xyz,cv::Mat& ground_mask)
{
   vector<cv::Mat> pitch_measure;

   cv::Mat pitch1;
   pitch1.create(1,1,CV_32F);
   cv::Mat pitch2;
   pitch2.create(1,1,CV_32F);

   cv::Mat bin,dst,dst1;

  GaussianBlur(v_dis_,dst,Size(3,3),0,0);

  Mat element = getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1));
  erode(dst,dst1,element);

  cv::threshold(dst1,bin,0,255,THRESH_OTSU);

  //cv::imshow("color",bin);
  //cv::waitKey(0);
    
  std::vector<Point> pt_list;
  //select the points to estimate line function
  for(int i = 26;i < bin.cols;i++)
  {
    for(int j = bin.rows-1;j>=0;j--)
    {
      int v = bin.at<uchar>(j,i);
      if(v == 255)
      {
        //cout<<"the lowest pixel is: "<<i<<","<<j<<endl;
        pt_list.push_back(cv::Point(i,j));

        for(int k = j; k > max(j-30,0); k--)
        {
            int v_a = bin.at<uchar>(k,i);
            if(v_a == 255)
            {
                pt_list.push_back(cv::Point(i,k));

            }
        }
        
        break;
      }
    }
  }

  std::vector<Point> pt_list2;
  //select the points to estimate line function
  for(int i = 12;i < 26;i++)
  {
    for(int j = bin.rows-1;j>=0;j--)
    {
      int v = bin.at<uchar>(j,i);
      if(v == 255)
      {
        pt_list2.push_back(cv::Point(i,j));
        break;
      }
    }
  }

  //cout<<"length of list is: "<<pt_list.size()<<endl;
  Vec4f line1;
  Vec4f line2;


  //fitting line function
  cv::fitLine(pt_list,line1,CV_DIST_L2,0,0.01,0.01);
  cv::fitLine(pt_list,line2,CV_DIST_L2,0,0.01,0.01);
  //cv::fitLine(pt_list2,line2,CV_DIST_L2,0,0.01,0.01);

  float a = line1[0];float b = line1[1];
  int x0 = cvRound(line1[2]);int y0 = cvRound(line1[3]);

  float a2 = line2[0];float b2 = line2[1];
  int x2 = cvRound(line2[2]);int y2 = cvRound(line2[3]);

  double V_C = y0 - (b/a)*x0;
  double V_C2 = y2 - (b2/a2)*x2;


  vector<Vec4i> lines;
  double V_0 = calib_.c_y;
  double F = calib_.f;

  double theta = atan((V_0-V_C)/F);
  double theta2 = atan((V_0-V_C2)/F);

  cv::line(v_dis_show,cv::Point(0,(b2/a2)*0+V_C2),cv::Point(26,(b2/a2)*26+V_C2),cv::Scalar(0,255,0),2,8);
  cv::line(v_dis_show,cv::Point(0,(b2/a2)*0+V_C2-20),cv::Point(26,(b2/a2)*26+V_C2-20),cv::Scalar(0,0,255),2,8);

  cv::line(v_dis_show,cv::Point(26,(b/a)*26+V_C),cv::Point(100,(b/a)*100+V_C),cv::Scalar(255,0,0),2,8);
  cv::line(v_dis_show,cv::Point(26,(b/a)*26+V_C-20),cv::Point(100,(b/a)*100+V_C-20),cv::Scalar(0,0,255),2,8);

  pitch1.at<float>(0)=theta;
  pitch2.at<float>(0)=theta2;

  //classify the points on ground plane and obstacles with respect to its distance to the line in V-disparity
 int xyz_cols = xyz.cols;
 int xyz_rows = xyz.rows;

  for(int j = 0; j < xyz_rows; j++)
  {

      float* xyz_ptr = xyz.ptr<float>(j);
      for(int i = 0;i < xyz_cols; i++)
      {

           float v = xyz_ptr[10*i + 4];
           float d = xyz_ptr[10*i + 5];
           int intensity = cvRound(xyz_ptr[10*i + 6]);
           float distance = (v-(b/a)*d-V_C);

           if(d > 26.0f)
           {
               if(distance > -14.0f)
               {
                   xyz_ptr[10*i+9] = 0.0f;//make the intensity as zero
               }
               else
               {
                   xyz_ptr[10*i+9] = abs(intensity);
               }
           }
           else if(d < 26.1f && d > 8.0f)
           {
               if(distance > -14.0f)
               {
                   xyz_ptr[10*i+9] = 0.0f;//make the intensity as zero
               }
               else
               {
                   xyz_ptr[10*i+9] = abs(intensity);
               }
           }
           else
           {
               xyz_ptr[10*i+9] = 0;//make the intensity as zero
           }

      }


  }


  vector<Mat> channels(8);
//  split img:
  split(xyz, channels);
//  get the channels (dont forget they follow BGR order in OpenCV)
  cv::Mat ch9 = channels[9];
  ground_mask.create(ch9.size(),CV_8UC1);
  cv::convertScaleAbs(ch9,ground_mask);

  pitch_measure.push_back(pitch1);
  pitch_measure.push_back(pitch2);

  return pitch_measure;
 }


/* Find all the possible moving segmentation by projecting
 *outliers into u-disparity map
*/
void UVDisparity::findAllMasks(const VisualOdometryStereo &vo, const Mat &img_L, cv::Mat& xyz, cv::Mat& roi_mask)
{
    
  cv::Mat img_show;
  cvtColor(img_L, img_show, CV_GRAY2BGR);
  cv::Mat ushow;
  cvtColor(u_dis_, ushow, CV_GRAY2BGR);
  
  cv::Scalar newVal(255);
  
  int numOutlier =  vo.quadmatches_outlier.size();

  //parameters for segmentation
  int min_intense = this->u_segment_par_.min_intense;// the lowest threshold of intensity in u-disparity
  int min_disparity_raw = this->u_segment_par_.min_disparity_raw;
  int min_area = this->u_segment_par_.min_area;

  //find all possible masks_
  for(int i = 0; i< numOutlier; i++)
  {
    int u = vo.quadmatches_outlier[i].u1c;
    short d = vo.quadmatches_outlier[i].dis_c;

    if(d > min_disparity_raw)
    {
       int dis = cvRound(d/16.0f);
       int utense = u_dis_.at<uchar>(dis,u);

       cv::circle(ushow, cv::Point(u,dis),2,cv::Scalar(255,0,0),2,8,0);

       if(utense > min_intense)
        {
          cv::Point seed(u,dis);
          cv::Rect ccomp;
          cv::Scalar low;
          cv::Scalar up(255-utense);
         //make the low and up difference
          if(0.5*utense > min_intense)
          {
            low = cv::Scalar(0.5*utense);
          }
          else
          {
            low = cv::Scalar(abs(utense - min_intense));
          }
          
          cv::Mat mask1,mask2;
          mask1.create(u_dis_.rows+2, u_dis_.cols+2,CV_8UC1);
          mask1 = Scalar::all(0);
          
          int newMaskVal = 255;
          Scalar newVal = Scalar( 120, 120, 120 );
          int connectivity = 8;

          int flags = connectivity + (newMaskVal << 8 ) +FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
          
          int area = floodFill(u_dis_,mask1,seed,newVal,&ccomp,low,up,flags);
          
          mask2 = mask1( Range(1, mask1.rows-1 ), Range(1, mask1.cols-1) );
          
          if(area > min_area)
          {
            masks_.push_back(mask2);
          }
        }
        
    }
  }

  int numInlier =  vo.quadmatches_inlier.size();

  //find all possible masks_
  for(int i = 0; i< numInlier; i++)
  {
    int u = vo.quadmatches_inlier[i].u1c;
    short d = vo.quadmatches_inlier[i].dis_c;

    if(d > min_disparity_raw)
    {
       int dis = cvRound(d/16.0f);
       cv::circle(ushow, cv::Point(u,dis),2,cv::Scalar(0,0,255),2,8,0);
    }
  }

  //cv::imshow("udisparity",ushow);
}


//select the finally Confirmed detection results from 2 frame's candidates_
void UVDisparity::confirmed()
{
  if(candidates_.size() == 0)
  {
    return;
  }
  else if(candidates_.size() == 1)
  {
    masks_confirmed_ = candidates_[0];
  }
  else if(candidates_.size() == 2)
  {
    vector<cv::Mat> cdt0 = candidates_[0];
    vector<cv::Mat> cdt1 = candidates_[1];
    cout<<"size of cdt0,cdt1: "<<cdt0.size()<<" "<<cdt1.size()<<endl;
    
    for(int i = 0; i < cdt1.size(); i++)
    {
      cv::Mat maski = cdt1[i];

      for(int j = 0; j < cdt0.size(); j++)
      {
        cv::Mat maskj = cdt0[j];

        int minRows = std::min(maski.rows, maskj.rows);
        
        if(minRows == maski.rows)
        {
          cv::Mat maskjj =  maskj( Range(0, minRows), Range(0, maskj.cols) );
          
          if(isOverlapped(maski,maskjj))
         {
           masks_confirmed_.push_back(maski);
         }

        }

         if(minRows == maskj.rows)
        {
          cv::Mat maskii =  maski( Range(0, minRows), Range(0, maski.cols) );

          if(isOverlapped(maskj,maskii))
         {
           masks_confirmed_.push_back(maski);
         }

        }   
      }
    }
  }
}



/*
  project the outliers in vo into the coordinates of udis
 */
void UVDisparity::verifyByInliers(const VisualOdometryStereo& vo, const cv::Mat& img_L)
{
  vector<cv::Mat>::iterator it = masks_.begin();
  int i = 0;

  for(; it!= masks_.end();)
  {
    cv::Mat mask = *it;     
    int num = numInlierInMask(mask,vo,img_L);
    
    if(num<inlier_tolerance_)
    {
      it++;
    }
    else
    {
      it = masks_.erase(it);
    }
    i++;
  }
}

//count the number of inliers in a mask
int UVDisparity::numInlierInMask(const cv::Mat& mask, const VisualOdometryStereo& vo,const cv::Mat& img_L)
{
   cv::Mat img_show,mask_show;
   cvtColor(img_L, img_show, CV_GRAY2BGR);
   cvtColor(mask, mask_show, CV_GRAY2BGR);
  
  int numInlier =  vo.quadmatches_inlier.size();

  int N = 0;
  
  for(int i = 0; i< numInlier; i++)
  {
    int u = vo.quadmatches_inlier[i].u1c;
    //int v = vo.p_matched_inlier[i].v1c;
    short d = vo.quadmatches_inlier[i].dis_c;
    int dis = cvRound(d/16.0f);
    
    int utense = mask.at<uchar>(dis,u);
    if(dis > 0 && utense!=0)
    {
      N++;
    }
  }

   // cv::imshow("inliershow",mask_show);
   // cv::waitKey(0);
  return N;
  
}


//judge if two masks_ are overlapped
bool UVDisparity::isOverlapped(const cv::Mat& mask1, const cv::Mat& mask2)
{
  cv::Mat result_and;//, result_or;

  bitwise_and(mask1,mask2,result_and);
  
  if(isAllZero(result_and))
  {
    return false;
  }
  else
  {
    return true;
  }
 
}

//judge if all the masks_ are separated
bool UVDisparity::isMasksSeparate()
{
  int numMasks = masks_.size();
  if(numMasks == 0) {return false;}
  vector<cv::Mat>::iterator it1, it2;
  
  for(it1 = masks_.begin(); it1 != masks_.end(); it1++)
  {
    for(it2 = it1+1; it2 != masks_.end(); it2++)
    {
      cv::Mat mask1 = *it1;
      cv::Mat mask2 = *it2;
            
      if(isOverlapped(mask1,mask2))
      {
        return false;
      }
    }
  }

  return true;
  
}



//merge two overlapped masks_
void UVDisparity::mergeMasks()
{
   vector<cv::Mat>::iterator it1, it2;

    for(it1 = masks_.begin(); it1 != masks_.end(); it1++)
    {
      for(it2 = it1+1; it2 != masks_.end(); )
      {
        cv::Mat mask1 = *it1;
        cv::Mat mask2 = *it2;
        cv::Mat mask_merge;
        
        if(isOverlapped(mask1,mask2))
        {
          bitwise_or(mask1,mask2,mask_merge);
          *it1 = mask_merge;
          it2 = masks_.erase(it2);
        }
        else
        {
          it2++;
        }
      }
    }
}

//adjust udisparity intense by sigmoid function
void UVDisparity::adjustUdisIntense(double scale, double range)
{

  for(int j = 0; j < u_dis_.rows; j++)
  {   
      uchar* u_ptr = u_dis_.ptr<uchar>(j);
      int disparity = j;

      double rate = sigmoid(disparity,scale,range,1);//0.02 -- 0.03之间比较好
      //cout<<"the rate is: "<<rate<<endl;
        
      for(int i = 0; i < u_dis_.cols; i++)
      {
        int intense = u_ptr[i];
        double intense_new = intense*1.0f * rate;
        int intense_int = cvRound(intense_new);
        
          if(intense_int > 255)
          {
            u_ptr[i] = 255;
          }
          else
          {
            u_ptr[i] = intense_int;
          }
        }

   }


}



//general processing function of UVDisparity based function
cv::Mat UVDisparity::Process(cv::Mat& img_L, cv::Mat& disp_sgbm,
                             VisualOdometryStereo& vo, cv::Mat& xyz,
                             cv::Mat& roi_mask, cv::Mat& ground_mask,
                             double& pitch1, double& pitch2)
{
    cv::Mat mask_moving;
    calVDisparity(disp_sgbm,xyz);

    //sequentially estimate pitch angles by Kalman Filter
    vector<cv::Mat> pitch_measures;

    pitch_measures = Pitch_Classify(xyz,ground_mask);
    pitch1_KF->predict();
    pitch1_KF->correct(pitch_measures[0]);

    pitch2_KF->predict();
    pitch2_KF->correct(pitch_measures[1]);

    pitch1 = pitch_measures[0].at<float>(0);
    pitch2 = pitch_measures[1].at<float>(0);


    //Improve 3D reconstruction results by pitch angles
    correct3DPoints(xyz,roi_,pitch1_KF->statePost.at<float>(0),pitch2_KF->statePost.at<float>(0));

    //set image ROI according to ROI3D (ROI within a 3D space)
    setImageROI(xyz, roi_mask);

    //filter inliers and outliers
    filterInOut(img_L,roi_mask,disp_sgbm,vo,pitch1);

    //calculate Udisparity image
    calUDisparity(disp_sgbm,xyz,roi_mask,ground_mask);

    //using sigmoid function to adjust Udisparity image for segmentation
    double scale = 0.02, range = 32;
    adjustUdisIntense(scale,range);

     //Find all possible segmentation
     findAllMasks(vo,img_L,xyz,roi_mask);

    if(masks_.size()>0)
    {
       //merge overlapped masks
       mergeMasks();

       //improve the segments by inliers
       verifyByInliers(vo,img_L);
     }

   //perform segmentation in disparity image
   segmentation(disp_sgbm,img_L,roi_mask,mask_moving);

   //demonstration
   cv::Mat img_show;
   img_L.copyTo(img_show,mask_moving);
   //cv::imshow("moving",img_show);
   //cv::waitKey(1);

   masks_.clear();
   return mask_moving;
}


void UVDisparity::segmentation(const cv::Mat& disparity, const cv::Mat& img_L,
                            cv::Mat& roi_mask, cv::Mat& mask_moving)
{
  cv::Mat img_show,img_show_last;
  mask_moving = Mat::zeros(img_L.rows, img_L.cols, CV_8UC1);
  cvtColor(img_L, img_show, CV_GRAY2BGR);

  double eps = 1.5f;

  int numMask = masks_.size();
  if(numMask == 0)
  {
    //cout<<"DON'T FIND MOVING OBJECT"<<endl;
    return;
  }

  cout<<"FIND MOVING OBJECT !!!"<<endl;
  for(int m = 0; m < numMask; m++)
  {

    cv::Mat mask = masks_[m];

    for(int i = 1; i < mask.rows; i++)
    {
      uchar* mask_ptr = mask.ptr<uchar>(i);

      for(int j = 1; j < mask.cols; j++)
      {
        int intense = (int)mask_ptr[j];
        //cout<<" "<<num_accu<<" ";

        if( intense != 0 )
        {

          for(int k = 0; k < disparity.rows; k++)
          {
            short dis_raw = disparity.at<short>(k,j);
            double dis_real = (dis_raw/16.0f);

            if(abs(dis_real - i) < eps)
            {
              cv::Point pt(j,k);
              if(isInMask(j,k,roi_mask))
              {
                mask_moving.at<uchar>(k,j)=255;
              }
            }
          }
        }
      }
    }



  }

  img_show.copyTo(img_show_last,mask_moving);
}


//judge if a mat is all zero or not
bool UVDisparity::isAllZero(const cv::Mat& mat)
{
   for(int i = 0;i<mat.rows;i++)
    {
      const uchar* mat_ptr = mat.ptr<uchar>(i);

      for(int j = 0; j<mat.cols; j++)
      {
        int d = mat_ptr[j];
        if(d!=0) return false;
      }

    }

   return true;
}

bool UVDisparity::isInMask(int u, int v, const cv::Mat& roi_mask)
{
  if(roi_mask.at<uchar>(v,u) >0) return true;
  else return false;
}


double UVDisparity::sigmoid(double t,double scale,double range, int mode = 1)
{
  double result;
  if(mode == 1)
  {
    result = range*1.0f/(1+exp(t*scale));//flipped sigmoid function
  }
  else if(mode == 0)
  {
    result = range*1.0f/(1+exp(-1.0*t*scale));//standard sigmoid function
  }
  return result;
}

