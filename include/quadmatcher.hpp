/*
   IRTES-SET laboratory
   Authors: You Li (liyou026@gmail.com)
   Descirption: This a sample code of my PhD works
*/

#ifndef QUADMATCHER__HPP
#define QUADMATCHER__HPP

#include <iostream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;


enum { DET_FAST, DET_STAR, DET_ORB, DET_SIFT, DET_SURF, DET_GFTT,
       DET_STAR_ADAPT, DET_FAST_ADAPT, DET_FAST_GRID, DET_STAR_GRID,DET_GFTT_GRID};

enum { DES_SIFT, DES_SURF, DES_BRISK, DES_FREAK,DES_ORB};


//core struct storing quadmatching result
struct pmatch {
    float   u1p,v1p; // u,v-coordinates in previous left  image
    int32_t i1p;     // feature index (for tracking)
    float   u2p,v2p; // u,v-coordinates in previous right image
    int32_t i2p;     // feature index (for tracking)
    float   u1c,v1c; // u,v-coordinates in current  left  image
    int32_t i1c;     // feature index (for tracking)
    float   u2c,v2c; // u,v-coordinates in current  right image
    int32_t i2c;     // feature index (for tracking)
    short dis_c,dis_p;//disparity for the current and previous

    pmatch(){}
    pmatch(float u1p,float v1p,int32_t i1p,float u2p,float v2p,int32_t i2p,
            float u1c,float v1c,int32_t i1c,float u2c,float v2c,int32_t i2c):
            u1p(u1p),v1p(v1p),i1p(i1p),u2p(u2p),v2p(v2p),i2p(i2p),
            u1c(u1c),v1c(v1c),i1c(i1c),u2c(u2c),v2c(v2c),i2c(i2c) {}
  };

class QuadFeatureMatch
{
public: QuadFeatureMatch(){};

        QuadFeatureMatch(cv::Mat& img_lc_, cv::Mat& img_rc_,
                         cv::Mat& img_lp_, cv::Mat& img_rp_,
			 cv::Mat& img_s_rc_, cv::Mat& img_s_rp,
			 bool mode_track_);

        //init the detector type or descriptor type
        void init(int detector_type, int descriptor_type);

        //detect image feature points
        void detectFeature();

        //extract feature descriptor for feature matching
        void extractDescriptor();

        //quad matching/tracking feature points
        void circularMatching();

        //print usage information of features
        void  printParams( cv::Algorithm* algorithm );


private:


        //filter out bad tracks
        void filteringTracks(vector<Point2f>& point_lc, vector<Point2f>& point_rc,
                            vector<Point2f>& point_lp, vector<Point2f>& point_rp,
                             vector<Point2f>& point_lp_direct);

        //nearest neighboring feature matching
        void matching(vector<KeyPoint> &keypoints1, Mat &descriptors1,
                      vector<KeyPoint> &keypoints2, Mat &descriptors2,
                      int search_width, int search_height, vector<DMatch>& matches);

        std::vector<pmatch> getMatches() { return quadmatches; }

        //demonstration functions
        void drawMatchesQuad(int time);
        void drawMatchesFlow(int time);
        void drawMatchesSimple(int time);

        bool withinRegion(cv::Point2f& pt, cv::Size& region); //judge if a point is within a certain region
        void KeyPoint2Point(vector<KeyPoint>& keypoint, vector<Point2f>& pt); //transform from keypoint to point2f
        float caldistance(const cv::Mat& vec1, const cv::Mat& vec2, bool descriptor_binary); //calculate descriptor difference

public:

        //quadmatches  --- the final output p_match vevtors, will used in visual odometry
        vector<pmatch> quadmatches;

private:

        /* ======================================================================================
         * img_lc,img_lp,img_rc,img_rp -- input images in left current, left previous, right current, right previous
         * keypoint_lc, keypoint_rc, keypoint_lp, keypoint_rp --  cv::KeyPoint vectors
         * point_lc ... ---- cv::Point2f vectors using for KLT tracking
         * descriptor_lc ... --- extracted feature descriptor by SIFT, SURF, ORB...
         * detector     --- cv::Ptr structure using for detect image feature points
         * descriptor   --- cv::Ptr structure using for extract feature descriptor
         * matcher      --- cv::Ptr structure using for matching features
         * ====================================================================================*/
         cv::Mat img_lc, img_lp, img_rc, img_rp, img_s_rc, img_s_rp;
         vector<KeyPoint> keypoint_lc,keypoint_rc,keypoint_lp,keypoint_rp;
         vector<cv::Point2f> point_lc,point_rc,point_lp,point_rp;
         cv::Mat descriptor_lc,descriptor_rc,descriptor_lp,descriptor_rp;

         Ptr<cv::FeatureDetector> detector;
         Ptr<cv::DescriptorExtractor> descriptor;
         Ptr<cv::DescriptorMatcher> matcher;


          /*  ==================================================================================
           *  mode_track -- indicator of tracking feature points or matching feature points
           *  descriptor_binary -- indicator of using binary feature descriptor or not
           *  distance_threshold -- threshold using in feature matching, if bigger than it, not matched
           *  ===================================================================================*/
         bool mode_track,descriptor_binary;
         float distance_threshold;


};

#endif

