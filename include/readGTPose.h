#include <iostream>
#include <fstream>
using namespace std;

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//the following are UBUNTU/LINUX ONLY terminal color
#define RESET "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */
#define BOLDBLACK "\033[1m\033[30m" /* Bold Black */
#define BOLDRED "\033[1m\033[31m" /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m" /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m" /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m" /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m" /* Bold White */

struct POSEFRAME
{
	double pose[12];
};

// 参数读取类
class PoseReader
{
public:
    PoseReader( string filename="../../05.txt" )
    {
        ifstream infile( filename.c_str() );
        if (!infile)
        {
            cerr<<"parameter file does not exist."<<endl;
            return;
        }
        while (!infile.eof())
        {
            POSEFRAME tmpPose;
            infile >> tmpPose.pose[0] >> tmpPose.pose[1] >> tmpPose.pose[2] >> tmpPose.pose[3]
                     >> tmpPose.pose[4] >> tmpPose.pose[5] >> tmpPose.pose[6] >> tmpPose.pose[7]
                      >> tmpPose.pose[8] >> tmpPose.pose[9] >> tmpPose.pose[10] >> tmpPose.pose[11];
            poseframe.push_back(tmpPose);
        }
        infile.close();
    }

    void getData( int index, cv::Mat& poseMatrix )
    {
        // Index Boundary Control
        if (index > poseframe.size()-2) 
        {
            std::cout << BOLDRED"ERROR! INDEX OUT OF RANGE!" << RESET" " << std::endl;
            return;
        }

        std::vector<POSEFRAME>::iterator iter = poseframe.begin();
        iter += index;
        poseMatrix = cv::Mat::zeros(3,4,CV_64FC1);

    	for (int i = 0; i < 12; i++)
    	{
            poseMatrix.at<double>(i/4, i%4) = (*iter).pose[i];
    	}

        //std::cout << BOLDWHITE"GTPose[ " << index << "/" << poseframe.size()-2 << "]" << std::endl 
        //                << BOLDGREEN"" << poseMatrix << RESET" " << std::endl << std::endl;
    }

private:
    std::vector<POSEFRAME> poseframe;
};
