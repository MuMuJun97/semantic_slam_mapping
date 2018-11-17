#include "rgbdframe.h"

using namespace rgbd_tutor;

int main()
{    
    ParameterReader para;
    FrameReader     fr(para);

    int i = 1 ;
    while( RGBDFrame::Ptr frame = fr.next() )
    {
        char _filename[256];
        sprintf(_filename,"/home/relaybot/Mu_Link/KittiData/01/depth/%06d.png",i);
        i++;

        cv::imshow( "image", frame->rgb );
	    cv::imshow( "depth", frame->depth );
	    cv::imshow( "semantic", frame->semantic );

        cv::imwrite(_filename,frame->depth);

	    cv::waitKey(1);
    }

    return 0;
}
