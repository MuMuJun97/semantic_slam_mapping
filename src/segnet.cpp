#include "segnet.h"

using namespace caffe;

Classifier::Classifier() 
{

#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

    //std::string model_file = "/home/relaybot/Desktop/slam3d/models/segnet_model_driving_webdemo.prototxt";
    //std::string trained_file = "/home/relaybot/Desktop/slam3d/models/segnet_weights_driving_webdemo.caffemodel";
    //std::string label_file = "/home/relaybot/Desktop/slam3d/models/semantic12.txt";
    std::string model_file = "../models/segnet_model_driving_webdemo.prototxt";
    std::string trained_file = "../models/segnet_weights_driving_webdemo.caffemodel";
    std::string label_file = "../models/semantic12.txt";

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean();

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	std::string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	//Blob<float>* output_layer = net_->output_blobs()[0];
}

std::vector<int> Classifier::Argmax(const std::vector<float>& v, int N) 
{
	// specify capacity
	Blob<float>* output_layer = net_->output_blobs()[0];
	//int output_channels = output_layer->channels();
	int output_size = output_layer->height() * output_layer->width();

	std::vector<int> result;
	result.reserve(output_size);
	for (int i = 0; i < output_size; ++i)
	{
		int idx = int(v[i]);
		result.push_back(idx); //N=1
	}
	return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) 
{
	std::vector<float> output = Predict(img);
	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	predictions.reserve(input_geometry_.height * input_geometry_.width);
	for (int i = 0; i < input_geometry_.height * input_geometry_.width; ++i)
	{
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], idx));
	}
	
	return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean() 
{
	//mean_ = cv::Mat(input_geometry_, CV_32FC3, cv::Scalar(104.00698793, 116.66876762, 122.67891434)); //BGR
	mean_ = cv::Mat(input_geometry_, CV_32FC3, cv::Scalar(0, 0, 0)); //BGR
}

std::vector<float> Classifier::Predict(const cv::Mat& img) 
{
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->ForwardPrefilled();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->height() * output_layer->width() * output_layer->channels();

	return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) 
{
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) 
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) 
{
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, CV_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, CV_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, CV_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, CV_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}
