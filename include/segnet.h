#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory> 
#include <cstring>
#include <utility>
#include <vector>
#include <iostream>

//using namespace caffe;  // NOLINT(build/namespaces)
//using namespace std;
//using namespace cv;

//#define CPU_ONLY

typedef std::pair<std::string, int> Prediction; //<labels_, index_>

class Classifier 
{
public:
	Classifier();

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 1);

private:
	void SetMean();

	std::vector<float> Predict(const cv::Mat& img);

	std::vector<int> Argmax(const std::vector<float>& v, int N);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

private:
	caffe::shared_ptr<caffe::Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<std::string> labels_;
};
