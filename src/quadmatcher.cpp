/*
   IRTES-SET laboratory
   Authors: You Li (liyou026@gmail.com)
   Descirption: This a sample code of my PhD works
*/

#include "quadmatcher.hpp"
#include <array>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"


using namespace cv;
using namespace std;







QuadFeatureMatch::QuadFeatureMatch(cv::Mat& img_lc_,cv::Mat& img_rc_,
                                   cv::Mat& img_lp_, cv::Mat& img_rp_,
				   cv::Mat& img_s_rc_, cv::Mat& img_s_rp_,
                                   bool mode_track_)
{
   img_lc = img_lc_;
   img_rc = img_rc_;
   img_lp = img_lp_;
   img_rp = img_rp_;
   img_s_rc = img_s_rc_;
   img_s_rp = img_s_rp_;

   mode_track = mode_track_;
}


void QuadFeatureMatch::matching(vector<KeyPoint>& keypoints1, cv::Mat& descriptors1,
                                 vector<KeyPoint>& keypoints2, cv::Mat& descriptors2,
                                 int search_width, int search_height, vector<DMatch>& matches)
 {
     //double time_test1 = (double)cv::getTickCount();
     //int min_distance_int = 9999999;

     cv::Point2f pt1,pt2;
     int num1 = keypoints1.size();
     int num2 = keypoints2.size();
     const KeyPoint* ptr1 = (num1 != 0) ? &keypoints1.front() : 0;
     const KeyPoint* ptr2 = (num2 != 0) ? &keypoints2.front() : 0;

     for(int i=0;i<num1;i++)
     {
         pt1 = ptr1[i].pt;
         int id = 0;
         float min_distance = 999999999.9f;

         for(int j = 0; j<num2;j++)
         {
             pt2 = ptr2[j].pt;
             if(std::abs(pt2.x - pt1.x)<search_width && std::abs(pt2.y - pt1.y)<search_height)
             {
                 float distance = caldistance(descriptors1.row(i),descriptors2.row(j),descriptor_binary);
                 cout<<distance<<endl;
		 
                 if(distance<min_distance)
                 {
		     {
                         min_distance = distance;
                         id = j;
		     }
                 }
             }
         }

         if(min_distance > distance_threshold) id = -1;

         matches.push_back(DMatch(i,id,min_distance));
     }

 }




void QuadFeatureMatch::drawMatchesQuad(int time)
{
    cv::Mat img_quad(img_lc.rows*2,img_lc.cols*2,img_lc.type());

    img_lc.copyTo(img_quad(cv::Rect(0,0,img_lc.cols,img_lc.rows)));
    img_rc.copyTo(img_quad(cv::Rect(img_lc.cols,0,img_lc.cols,img_lc.rows)));
    img_lp.copyTo(img_quad(cv::Rect(0,img_lc.rows,img_lc.cols,img_lc.rows)));
    img_rp.copyTo(img_quad(cv::Rect(img_lc.cols,img_lc.rows,img_lc.cols,img_lc.rows)));

    cv::Mat img_quad_show;
    cvtColor(img_quad, img_quad_show, CV_GRAY2BGR);

    cv::Scalar red(0,0,255);
    cv::Scalar green(0,255,0);
    cv::Scalar blue(255,0,0);
    int num_matches = quadmatches.size();
    cv::Point points_lc, points_rc,points_lp,points_rp;

    for(int i = 0;i<num_matches;i++)
    {
        cv::Mat test;
        img_quad_show.copyTo(test);

        if(1)
        {
            points_lc.x = quadmatches[i].u1c;
            points_lc.y = quadmatches[i].v1c;

            points_rc.x = quadmatches[i].u2c;
            points_rc.y = quadmatches[i].v2c;

            points_lp.x = quadmatches[i].u1p;
            points_lp.y = quadmatches[i].v1p;

            points_rp.x = quadmatches[i].u1p;
            points_rp.y = quadmatches[i].v1p;

        }

        if(0)
        {
            points_lc.x = point_lc[i].x;
            points_lc.y = point_lc[i].y;

            points_rc.x = point_rc[i].x;
            points_rc.y = point_rc[i].y;

            points_lp.x = point_lp[i].x;
            points_lp.y = point_lp[i].y;

            points_rp.x = point_rp[i].x;
            points_rp.y = point_rp[i].y;
        }



        cv::circle(img_quad_show,points_lc,1,red,2,8,0);
        cv::circle(img_quad_show,points_rc+cv::Point(img_lc.cols,0),1,blue,2,8,0);

        cv::circle(img_quad_show,points_lp+cv::Point(0,img_lc.rows),1,red,2,8,0);
        cv::circle(img_quad_show,points_rp+cv::Point(img_lc.cols,img_lc.rows),1,blue,2,8,0);

        cv::line(img_quad_show,points_lc,points_rc+cv::Point(img_lc.cols,0),green,1,8,0);
        cv::line(img_quad_show,points_lc,points_lp+cv::Point(0,img_lc.rows),green,1,8,0);
        cv::line(img_quad_show,points_rc + cv::Point(img_lc.cols,0),
                     points_rp + cv::Point(img_lc.cols,img_lc.rows),green,1,8,0);
        cv::line(img_quad_show,points_lp+cv::Point(0,img_lc.rows),
                     points_rp+cv::Point(img_lc.cols,img_lc.rows),green,1,8,0);

    }

    cv::imshow("image quad",img_quad_show);
    cv::imwrite("quadmatch.jpg",img_quad_show);
    cv::waitKey(time);


}

 void QuadFeatureMatch::drawMatchesFlow(int time)
 {
     cv::Mat img_show;
     cvtColor(img_lc, img_show, CV_GRAY2BGR);

     cv::Scalar red(0,0,255);
     cv::Scalar green(0,255,0);
     cv::Scalar blue(255,0,0);
     int num = quadmatches.size();

     for(int i = 0; i < num; i++)
     {
          cv::Point pt;
          pt.x = quadmatches[i].u1c;
          pt.y = quadmatches[i].v1c;

          cv::Point pt2;
          pt2.x = quadmatches[i].u1p;
          pt2.y = quadmatches[i].v1p;

          cv::circle(img_show,pt,1,red,2,8,0);
          cv::line(img_show,pt,pt2,green,1,8,0);
     }

     cv::imshow("image sparse flow",img_show);
     cv::waitKey(time);
 }

 void QuadFeatureMatch::drawMatchesSimple(int time)
 {
     cv::Mat img_show;
     cvtColor(img_lc, img_show, CV_GRAY2BGR);

     cv::Scalar red(0,0,255);
     cv::Scalar green(0,255,0);
     cv::Scalar blue(255,0,0);

     int num = point_lc.size();

     for(int i = 0; i < num; i++)
     {
         cv::Point2f pt = point_lc[i];
         cv::circle(img_show,pt,1,red,2,8,0);
     }

     cv::imshow("Detected Features",img_show);
     cv::waitKey(time);
 }





 void QuadFeatureMatch::init(int detector_type, int descriptor_type)
 {
    switch(detector_type)
     {
        case DET_FAST:
        {
            detector = FeatureDetector::create("FAST");
            detector->set("nonmaxSuppression", true);
            detector->set("threshold",20);
            break;
        }

        case DET_FAST_ADAPT:
        {
          detector = new DynamicAdaptedFeatureDetector (new FastAdjuster(30,true),800,1000,10);
          break;
        }
        case DET_FAST_GRID:
        {
         //detector = new GridAdaptedFeatureDetector(new FastFeatureDetector(10,true),800,10,10);

         detector = new GridAdaptedFeatureDetector(new DynamicAdaptedFeatureDetector (new FastAdjuster(20,true),12,15,10),
                                                   800,10,10);
          break;
        }
        case DET_STAR:
        {
            detector = FeatureDetector::create("STAR");
            detector->set("lineThresholdProjected",10);
            detector->set("lineThresholdBinarized",8);
            detector->set("suppressNonmaxSize",5);
            detector->set("responseThreshold",30);
            detector->set("maxSize", 16);
            break;
        }
       case DET_STAR_ADAPT:
       {
            detector = new DynamicAdaptedFeatureDetector (new StarAdjuster(40),800,1000,10);
            break;
       }

        case DET_STAR_GRID:
        {
         detector = new GridAdaptedFeatureDetector (new StarFeatureDetector(16,10,10,8,5),800,4,4);
            break;
        }

        case DET_ORB:
        {
            detector = FeatureDetector::create("ORB");
            //printParams(detector);
            detector->set("WTA_K",2);
            detector->set("scoreType",ORB::HARRIS_SCORE);
            detector->set("patchSize",31);
            detector->set("edgeThreshold",31);
            detector->set("scaleFactor", 1.2f);
            detector->set("nLevels", 5);
            detector->set("nFeatures",800);
            break;
        }
        case DET_SURF:
        {
            detector = FeatureDetector::create("SURF");
            //printParams(detector);
            detector->set("hessianThreshold", 400);
            detector->set("nOctaves",4);
            detector->set("nOctaveLayers",2);
            detector->set("upright",1);
            break;
        }
        case DET_SIFT:
        {
            detector = FeatureDetector::create("SIFT");
            //printParams(detector);
            detector->set("nFeatures", 1000);
            detector->set("nOctaveLayers",3);
            detector->set("contrastThreshold",0.04);
            detector->set("edgeThreshold",10);
            detector->set("sigma",1.6);
            break;
        }

        case DET_GFTT:
        {
             detector = FeatureDetector::create("GFTT");
             //printParams(detector);
             detector->set("qualityLevel",0.04);
             detector->set("minDistance",8);
            break;
        }

        case DET_GFTT_GRID:
        {
         detector = new GridAdaptedFeatureDetector (new GoodFeaturesToTrackDetector(20,0.01,5),800,8,8);
            //printParams(detector);
           // detector->set("qualityLevel",0.05);
           // detector->set("minDistance",5);
            break;
        }
     }

    if(mode_track == false)
     {
         switch(descriptor_type)
         {
            case DES_SIFT:
            {
                descriptor = DescriptorExtractor::create("SIFT");
                distance_threshold = 8000.0f;
                descriptor_binary = false;
                break;
            }
            case DES_SURF:
            {
                descriptor = DescriptorExtractor::create("SURF");
                distance_threshold = 0.3f;
                descriptor_binary = false;
                break;
            }
            case DES_BRISK:
            {
                descriptor = DescriptorExtractor::create("BRISK");
                distance_threshold = 120.0f;
                descriptor_binary = true;
                break;
            }
            case DES_FREAK:
            {
                descriptor = DescriptorExtractor::create("FREAK");
                distance_threshold = 100.0f;
                descriptor_binary = true;
                break;
            }
            case DES_ORB:
            {
                descriptor = DescriptorExtractor::create("ORB");
                distance_threshold = 80.0f;
                descriptor_binary = true;
                break;
            }
         }
     }

 }

void QuadFeatureMatch::extractDescriptor()
{

    if(!keypoint_lc.size() || !keypoint_lp.size()
            ||!keypoint_rc.size() || !keypoint_rp.size())
    {
        cout<<"Please Detect Feature Points At First!"<<endl;
        return;
    }

    {
            double time_descriptor = (double)cv::getTickCount();
            descriptor->compute( img_lc, keypoint_lc, descriptor_lc );
            descriptor->compute( img_rc, keypoint_rc, descriptor_rc );
            descriptor->compute( img_lp, keypoint_lp, descriptor_lp );
            descriptor->compute( img_rp, keypoint_rp, descriptor_rp );

            time_descriptor = ((double)cv::getTickCount() - time_descriptor)/cv::getTickFrequency()*1000;
            cout<<"The feature descriptor extraction in 4 images costs "<<time_descriptor<<" ms"<<endl;
     }
}



void QuadFeatureMatch::detectFeature()
{
    if(mode_track == false)
    {
        double time_detector = (double)cv::getTickCount();
        detector->detect(img_lc,keypoint_lc);
        detector->detect(img_rc,keypoint_rc);
        detector->detect(img_lp,keypoint_lp);
        detector->detect(img_rp,keypoint_rp);

        time_detector = ((double)cv::getTickCount() - time_detector)/cv::getTickFrequency()*1000;

        KeyPoint2Point(keypoint_lc, point_lc);
        KeyPoint2Point(keypoint_rc, point_rc);
        KeyPoint2Point(keypoint_lp, point_lp);
        KeyPoint2Point(keypoint_rp, point_rp);


    }
    else
    {
        double time_detector = (double)cv::getTickCount();

        detector->detect(img_lc,keypoint_lc);
        KeyPoint2Point(keypoint_lc, point_lc);

        time_detector = ((double)cv::getTickCount() - time_detector)/cv::getTickFrequency()*1000;
    }

}


void QuadFeatureMatch::filteringTracks(vector<Point2f>& point_lc, vector<Point2f>& point_rc,
                    vector<Point2f>& point_lp, vector<Point2f>& point_rp,
                     vector<Point2f>& point_lp_direct)
{

    if(point_lc.size()!=point_rc.size()
       || point_lc.size()!=point_lp.size()
       || point_rc.size()!=point_rp.size()
       || point_rp.size()!=point_lp.size()
       || point_lp.size()!=point_lp_direct.size())
    {
        cout<<"ERROR! The Size of Input are not equal!"<<endl;
        return;
    }

    int minHeightDif =20;
    int minHeightDif2 =30;
    int minWidthDif =200;
    int minDisparity = 3;
    cv::Size region(1280,960);

    int num_lc = point_lc.size();

    for(int i = 0; i < num_lc; i++)
    {
        cv::Point2f pt_lc = point_lc[i];
        cv::Point2f pt_rc = point_rc[i];
        cv::Point2f pt_lp = point_lp[i];
        cv::Point2f pt_rp = point_rp[i];
        cv::Point2f pt_lp_predict = point_lp_direct[i];



        int dif_height1 = cvRound(abs(pt_lc.y - pt_rc.y));
        int dif_height2 = cvRound(abs(pt_lp.y - pt_rp.y));

        int dif_height11 = cvRound(abs(pt_lc.y - pt_lp.y));
        int dif_height22 = cvRound(abs(pt_rc.y - pt_rp.y));

        int dif_width1 = cvRound(abs(pt_lc.x - pt_lp.x));
        int dif_width2 = cvRound(abs(pt_rc.x - pt_rp.x));

        int disparity1 = cvRound(abs(pt_lc.x - pt_rc.x));
        int disparity2 = cvRound(abs(pt_lp.x - pt_rp.x));

        int dif_x = cvRound(abs(pt_lp.x - pt_lp_predict.x ));
        int dif_y = cvRound(abs(pt_lp.y - pt_lp_predict.y ));

        if(     withinRegion(pt_lc,region)           //outliers
               && withinRegion(pt_lp,region)
               && withinRegion(pt_rc,region)
               && withinRegion(pt_rp,region)
               && dif_height1 < minHeightDif
               && dif_height2 < minHeightDif
               && dif_height11 < minHeightDif2
                && dif_height22 < minHeightDif2
                && dif_width1 < minWidthDif
                && dif_width2 < minWidthDif
                && disparity1 > minDisparity
                && disparity2 > minDisparity
                && dif_x < 1 && dif_y < 1
           )
        {
            pmatch res;
            res.u1c = pt_lc.x;
            res.v1c = pt_lc.y;
            res.u1p = pt_lp.x;
            res.v1p = pt_lp.y;
            res.u2c = pt_rc.x;
            res.v2c = pt_rc.y;
            res.u2p = pt_rp.x;
            res.v2p = pt_rp.y;
            res.i1c = res.i1p = res.i2c = res.i2p = i;

		// semantic validation
		int cy = res.v2c;
		int cx = res.u2c;
		int py = res.v2p;
		int px = res.u2p;
		//if ( img_s_rc.ptr<uchar>(cy)[3*cx+0] == img_s_rp.ptr<uchar>(py)[3*px+0] && img_s_rc.ptr<uchar>(cy)[3*cx+1] == img_s_rp.ptr<uchar>(py)[3*px+1] && img_s_rc.ptr<uchar>(cy)[3*cx+2] == img_s_rp.ptr<uchar>(py)[3*px+2] )
		    quadmatches.push_back(res);
        }
    }
}


void QuadFeatureMatch::KeyPoint2Point(vector<KeyPoint>& keypoint, vector<Point2f>& pt)
{
    pt.clear();      cv::Mat ground_mask;

    int num = keypoint.size();
    for(int i = 0;i<num;i++)
    {
        pt.push_back(keypoint[i].pt);
    }
}


bool QuadFeatureMatch::withinRegion(cv::Point2f& pt, cv::Size& region)
{
    if(pt.x<region.width && pt.x>0.0f && pt.y<region.height && pt.y>0.0f) return true;
    else return false;
}


float QuadFeatureMatch::caldistance(const cv::Mat& vec1, const cv::Mat& vec2, bool descriptor_binary)
{
    CV_Assert(vec1.rows == 1 && vec2.rows == 1 && vec1.cols == vec2.cols && vec1.type()==vec2.type());

    float d = 0.0f;

    if(!descriptor_binary)// for sift and surf, L2_NORM
    {
        CV_Assert(vec1.type()==CV_32F);
        d = (float) cv::norm(vec1,vec2,NORM_L2);

    }
    else
    {
        CV_Assert(vec1.type()==CV_8U);
        d = (float) cv::norm(vec1,vec2,NORM_HAMMING);
    }

    return d;
}



void QuadFeatureMatch::circularMatching()
 {
    if(mode_track) // circular tracking feature points
    {
        vector<uchar> status_lrc,status_ll,status_rr,status_lrp;
        vector<float> error_lrc, error_ll, error_rr, error_lrp;
        int winsize = 11;
        int maxlvl = 3;

        vector<cv::Point2f> point_lp_direct;
        vector<uchar> status_lcp_direct;
        vector<float> error_lcp_direct;

        cv::TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS,200,0.01);

        {
          // double time_KLT = (double)cv::getTickCount();

           cv::calcOpticalFlowPyrLK(img_lc, img_rc, point_lc, point_rc, status_lrc, error_lrc,
                                    cv::Size(winsize, winsize), maxlvl,criteria,OPTFLOW_LK_GET_MIN_EIGENVALS,0.000001);

           cv::calcOpticalFlowPyrLK(img_rc, img_rp, point_rc, point_rp, status_rr, error_rr,
                                    cv::Size(winsize, winsize), maxlvl,criteria,OPTFLOW_LK_GET_MIN_EIGENVALS,0.000001);

           cv::calcOpticalFlowPyrLK(img_rp, img_lp, point_rp, point_lp, status_lrp, error_lrp,
                                    cv::Size(winsize, winsize), maxlvl,criteria,OPTFLOW_LK_GET_MIN_EIGENVALS,0.000001);

           cv::calcOpticalFlowPyrLK(img_lc, img_lp, point_lc, point_lp_direct, status_lcp_direct, error_lcp_direct,
                                    cv::Size(winsize, winsize), maxlvl,criteria,OPTFLOW_LK_GET_MIN_EIGENVALS,0.000001);

          // time_KLT = ((double)cv::getTickCount() - time_KLT)/cv::getTickFrequency()*1000;
        }

       /*Filtering out inaccuracy tracked points*/
       double time_filtering = (double)cv::getTickCount();

       filteringTracks(point_lc,point_rc,point_lp,point_rp,point_lp_direct);

       time_filtering = ((double)cv::getTickCount() - time_filtering)/cv::getTickFrequency()*1000;

    }


    else  //circular matching feature points
    {
        this->extractDescriptor();//计算左右四张图的关键点和描述子

        vector<DMatch> matches_lcp, matches_lrc,matches_rcp,matches_rlp;
        double time_quadmatch = (double)cv::getTickCount();

        matching(keypoint_lc,descriptor_lc, keypoint_rc,descriptor_rc, 20, 2, matches_lrc);

        matching(keypoint_rc,descriptor_rc, keypoint_rp,descriptor_rp, 20, 20,matches_rcp);

        matching(keypoint_rp,descriptor_rp, keypoint_lp,descriptor_lp, 20, 2,matches_rlp);

        time_quadmatch = ((double)cv::getTickCount() - time_quadmatch)/cv::getTickFrequency()*1000;
//        cout<<"The Circular matching through 4 images costs "<<time_quadmatch<<"ms"<<endl;
//        cout<<"the size of keypoints_lc is: "<<keypoint_lc.size()<<" The matches_lrc's size is: ' "<<matches_lrc.size()<<endl;
//        cout<<"the size of keypoints_rc is: "<<keypoint_rc.size()<<" The matches_rcp's size is: ' "<<matches_rcp.size()<<endl;
//        cout<<"the size of keypoints_rp is: "<<keypoint_rp.size()<<" The matches_lrp's size is: ' "<<matches_rlp.size()<<endl;

        int min_disparity=3;
        int max_delta_x = 2;

        int num_lc = keypoint_lc.size();
        quadmatches.clear();

        for(int i = 0; i<num_lc; i++)
        {
            int id_lc,id_rc,id_rp,id_lp = 0;
            //std::array<int,4> tmp;
            id_lc = i;
            id_rc = matches_lrc[i].trainIdx;
            if(id_rc > 0)
            {
                id_rp = matches_rcp[id_rc].trainIdx;

               if(id_rp > 0)
               {
                   id_lp = matches_rlp[id_rp].trainIdx;

                   if(id_lp > 0)
                   {
                       pmatch test;
                       test.u1c = keypoint_lc[id_lc].pt.x;
                       test.v1c = keypoint_lc[id_lc].pt.y;
                       test.i1c = id_lc;

                       test.u2c = keypoint_rc[id_rc].pt.x;
                       test.v2c = keypoint_rc[id_rc].pt.y;
                       test.i2c = id_rc;

                       test.u2p = keypoint_rp[id_rp].pt.x;
                       test.v2p = keypoint_rp[id_rp].pt.y;
                       test.i2p = id_rp;

                       test.u1p = keypoint_lp[id_lp].pt.x;
                       test.v1p = keypoint_lp[id_lp].pt.y;
                       test.i1p = id_lp;

                       int delta_x;
                       delta_x = std::abs(std::abs(test.u1c - test.u1p) - std::abs(test.u2c - test.u2p));
                       int disparity = std::abs(test.u1c-test.u2c);
                        if(delta_x < max_delta_x && disparity > min_disparity)
                       {
                           quadmatches.push_back(test);
                       }
                   }
               }
            }
        }

    }


}




//print usage parameters of feature detector and descriptor
void QuadFeatureMatch::printParams( cv::Algorithm* algorithm )
{
    std::vector<std::string> parameters;
    algorithm->getParams(parameters);

    for (int i = 0; i < (int) parameters.size(); i++) {
        std::string param = parameters[i];
        int type = algorithm->paramType(param);
        std::string helpText = algorithm->paramHelp(param);
        std::string typeText;

        switch (type) {
        case cv::Param::BOOLEAN:
            typeText = "bool";
            break;
        case cv::Param::INT:
            typeText = "int";
            break;
        case cv::Param::REAL:
            typeText = "real (double)";
            break;
        case cv::Param::STRING:
            typeText = "string";
            break;
        case cv::Param::MAT:
            typeText = "Mat";
            break;
        case cv::Param::ALGORITHM:
            typeText = "Algorithm";
            break;
        case cv::Param::MAT_VECTOR:
            typeText = "Mat vector";
            break;
        }
        std::cout << "Parameter '" << param << "' type=" << typeText << " help=" << helpText << std::endl;
    }
}

