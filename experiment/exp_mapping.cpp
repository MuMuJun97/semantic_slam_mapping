#include "rgbdframe.h"
#include "track.h"
#include "pose_graph.h"
#include "common_headers.h"
#include "mapper.h"
#include "readGTPose.h"
#include "vo_stereo.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;
using namespace rgbd_tutor;

int main()
{

    ParameterReader	parameterReader;
    VisualOdometryStereo::parameters voparam; 
    double f = parameterReader.getData<double>("camera.fx");
    double c_u = parameterReader.getData<double>("camera.cx");
    double c_v = parameterReader.getData<double>("camera.cy");
    double base = parameterReader.getData<double>("camera.baseline");
    double inlier_threshold = parameterReader.getData<double>("inlier_threshold");
    voparam.calib.f  = f;      voparam.calib.cu = c_u;
    voparam.calib.cv = c_v;    voparam.base     = base;	
    voparam.inlier_threshold = inlier_threshold;
    Tracker::Ptr	tracker( new Tracker(parameterReader, voparam) );
    FrameReader		frameReader( parameterReader );
    PoseGraph		poseGraph( parameterReader, tracker );
    Mapper              mapper( parameterReader, poseGraph );

    while ( RGBDFrame::Ptr frame = frameReader.next() )
    {
        cout<<frame->id<<endl;
        boost::timer timer;
        cv::imshow("image", frame->rgb);
        //cv::imshow("depth", frame->depth);
        cv::waitKey(1);
        tracker->updateFrame( frame );
        poseGraph.tryInsertKeyFrame( frame );
        
        if (tracker->getState() == Tracker::LOST)
        {
            cout << "tracker is lost" << endl;
            //break;
        }
    }
 
    // shutdown
    mapper.SaveMap();
    poseGraph.shutdown();
    mapper.shutdown();

    return 0;
}
