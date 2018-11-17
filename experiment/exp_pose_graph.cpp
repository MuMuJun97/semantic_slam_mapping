#include "track.h"
#include "pose_graph.h"
#include "vo_stereo.hpp"
#include "readGTPose.h"

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

    while( RGBDFrame::Ptr frame = frameReader.next() )
    {
        cout<<"*******************************************"<<endl;
        boost::timer timer;
        cout<<"loading frame "<<frame->id<<endl;
        Eigen::Isometry3d T = tracker->updateFrame( frame );
        cout<<"current frame T = "<<endl<<T.matrix()<<endl;
        cv::imshow( "image", frame->rgb );
        if ( poseGraph.tryInsertKeyFrame( frame ) == true )
        {
            cout<<"Insert key-frame succeed"<<endl;
            cv::waitKey(1);
        }
        else
        {
            cout<<"Insert key-frame failed"<<endl;
            cv::waitKey(1);
        }
        cout<<GREEN<<"time cost="<<timer.elapsed()<<RESET<<endl;
    }

    poseGraph.shutdown();
    return 0;
}
