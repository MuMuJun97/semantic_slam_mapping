#ifndef MAPPER_H
#define MAPPER_H

#include "common_headers.h"
#include "rgbdframe.h"
#include "pose_graph.h"

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>

namespace rgbd_tutor
{
using namespace rgbd_tutor;

class Mapper
{
public:
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    Mapper( const ParameterReader& para, PoseGraph& graph )
        : parameterReader( para ), poseGraph( graph )
    {
	    resolution = para.getData<double>("mapper_resolution");
        max_distance = para.getData<double>("mapper_max_distance");
        viewerThread = std::make_shared<std::thread> ( std::bind( &Mapper::viewer, this ));

	    area_thres = para.getData<int>("motion_area_thres");
        overlay_portion_thres = para.getData<double>("motion_overlay_portion_thres");
    }

    void shutdown()
    {
        shutdownFlag = true;
        if (viewerThread != nullptr)
	    {
        	viewerThread->join();
	    }
    }

    // viewer线程
    void viewer();

    void SaveMap();


protected:
    PointCloud::Ptr generatePointCloud( const RGBDFrame::Ptr &frame );

    // viewer thread
    std::shared_ptr<std::thread>	viewerThread = nullptr;
    const ParameterReader& parameterReader;
    PoseGraph&  poseGraph;

    int    keyframe_size    = 0;
    double resolution       = 0.8;
    double max_distance     = 8.0;
    bool   shutdownFlag	    = false;


    // motion
    int area_thres = 1000;
    double overlay_portion_thres = 0.143;
    cv::Mat moving_mask;

    void semantic_motion_fuse( const RGBDFrame::Ptr &frame);

    //PointCloud::Ptr globalMap;

};


}


#endif // MAPPER_H
