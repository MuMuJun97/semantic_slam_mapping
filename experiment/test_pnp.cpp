#include "pnp.h"
#include "common_headers.h"
#include "readGTPose.h"

using namespace std;
using namespace rgbd_tutor;

int main()
{
    cout<<"running test pnp"<<endl;
    ParameterReader para;
    FrameReader frameReader( para );
    OrbFeature  orb(para);
    PnPSolver   pnp(para, orb);

    RGBDFrame::Ptr refFrame = frameReader.next();
    orb.detectFeatures( refFrame );
    Eigen::Isometry3d   speed = Eigen::Isometry3d::Identity();

    // plot
    string pose_file = para.getData<string>("gtpose_source");
    std::cout << "Sequence: " << pose_file << std::endl;
    PoseReader poseReader(pose_file);
    cv::Mat poseMap(1500,1500,CV_8UC3,cv::Scalar(0,0,0));
    int n = 0;
    while (1)
    {
        cout<<"*************************************"<<endl;
        boost::timer timer;
        RGBDFrame::Ptr currFrame = frameReader.next();

        if ( currFrame == nullptr )
        {
            break;
        }
        currFrame->T_f_w = speed * refFrame->T_f_w ;
        orb.detectFeatures( currFrame );

        PNP_INFORMATION info;
        bool result = pnp.solvePnPLazy( refFrame, currFrame, info, true );

        if ( result == false )
        {
            cout<<"pnp failed"<<endl;
            refFrame = currFrame;
            cv::waitKey(1);
        }
        else
        {
            currFrame->T_f_w = info.T * refFrame->T_f_w;
            cout<<"result.T="<<endl;
            cout<<info.T.matrix()<<endl;
            cout<<"current = "<< endl << currFrame->T_f_w.matrix() << endl;
            speed = info.T;
            refFrame = currFrame;
            cv::waitKey(2);
        }

        cout<<GREEN<<"time used = "<<timer.elapsed()<<RESET<<endl;

	int x = -currFrame->T_f_w(0,3);
	int y = -currFrame->T_f_w(2,3);
	cout << "x: " << x << " y: " << y << endl;

	// gt_pose
	n += 1;
	cv::Mat gtpose;
	poseReader.getData(n+1, gtpose);

	cv::circle(poseMap, cv::Point(poseMap.cols/2+x, poseMap.rows/2-y), 2, cv::Scalar(255,0,0));  
	cv::circle(poseMap, cv::Point(poseMap.cols/2+gtpose.ptr<double>(0)[3], poseMap.rows/2-gtpose.ptr<double>(2)[3]), 2, cv::Scalar(0,0,255));  
	cv::namedWindow("pose", 0);
	cv::imshow("pose", poseMap);
	cv::waitKey(3);
    }

    return 0;
}
