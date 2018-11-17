#include "rgbdframe.h"
#include "common_headers.h"
#include "parameter_reader.h"
#include "stereo.h"

using namespace rgbd_tutor;

RGBDFrame::Ptr   FrameReader::next()
{
    switch (dataset_type) {
    case NYUD:
        //TODO 增加nyud的接口
        break;
    case TUM:
    {
        if (currentIndex < start_index || currentIndex >= rgbFiles.size())
            return nullptr;

        RGBDFrame::Ptr   frame (new RGBDFrame);
        frame->id = currentIndex;
        frame->rgb = cv::imread( dataset_dir + rgbFiles[currentIndex]);
        frame->depth = cv::imread( dataset_dir + depthFiles[currentIndex], CV_LOAD_IMAGE_UNCHANGED);

        if (frame->rgb.data == nullptr || frame->depth.data==nullptr)
        {
            // 数据不存在
            return nullptr;
        }

        frame->camera = this->camera;
        currentIndex ++;
        return frame;
    }    
    case KITTI:
    {
        if (currentIndex < start_index || currentIndex >= rgbFiles.size() || currentIndex >= parameterReader.getData<int>("end_index"))
            return nullptr;

        string rgb_dir = parameterReader.getData<string>("rgb_dir"); 
        string depth_dir = parameterReader.getData<string>("depth_dir"); 

	// RGB
        RGBDFrame::Ptr   frame (new RGBDFrame);
        frame->id = currentIndex;
        frame->rgb = cv::imread(dataset_dir + rgb_dir + rgbFiles[currentIndex+1]);

//	frame->rgb_cur_r = cv::imread(dataset_dir + "image_1/" + rgbFiles[currentIndex+1], 1);
//	frame->rgb_pre_r = cv::imread(dataset_dir + "image_1/" + rgbFiles[currentIndex], 1);
//
//	// libviso2
//	frame->img_lc = cv::imread(dataset_dir + "image_0/" + rgbFiles[currentIndex+1], 0);
//	frame->img_lp = cv::imread(dataset_dir + "image_0/" + rgbFiles[currentIndex], 0);
//	frame->img_rc = cv::imread(dataset_dir + "image_1/" + rgbFiles[currentIndex+1], 0);
//	frame->img_rp = cv::imread(dataset_dir + "image_1/" + rgbFiles[currentIndex], 0);
//
//	// motion
//	frame->moving_mask = cv::Mat::zeros(frame->img_lc.rows, frame->img_lc.cols, CV_8UC1);
//
//	// Depth
//	cv::Mat img_l = cv::imread(dataset_dir + "image_0/" + rgbFiles[currentIndex+1], 0);
//	cv::Mat img_r = cv::imread(dataset_dir + "image_1/" + rgbFiles[currentIndex+1], 0);


        frame->rgb_cur_r = cv::imread(dataset_dir + "image_3/" + rgbFiles[currentIndex+1], 1);
        frame->rgb_pre_r = cv::imread(dataset_dir + "image_3/" + rgbFiles[currentIndex], 1);

        // libviso2
        frame->img_lc = cv::imread(dataset_dir + "image_2/" + rgbFiles[currentIndex+1], 0);
        frame->img_lp = cv::imread(dataset_dir + "image_2/" + rgbFiles[currentIndex], 0);
        frame->img_rc = cv::imread(dataset_dir + "image_3/" + rgbFiles[currentIndex+1], 0);
        frame->img_rp = cv::imread(dataset_dir + "image_3/" + rgbFiles[currentIndex], 0);

        // motion
        frame->moving_mask = cv::Mat::zeros(frame->img_lc.rows, frame->img_lc.cols, CV_8UC1);

        // Depth
        cv::Mat img_l = cv::imread(dataset_dir + "image_2/" + rgbFiles[currentIndex+1], 0);
        cv::Mat img_r = cv::imread(dataset_dir + "image_3/" + rgbFiles[currentIndex+1], 0);


	cv::Mat disp_sgbm; 
	calDisparity_SGBM(img_l, img_r, disp_sgbm);
	frame->disparity = disp_sgbm.clone();

	double minDisparity = FLT_MAX; cv::minMaxIdx(disp_sgbm, &minDisparity, 0, 0, 0 );
	frame->depth = cv::Mat(img_l.size(), CV_16UC1, cv::Scalar(0));
	double baseline = parameterReader.getData<double>("camera.baseline");
	double cu = parameterReader.getData<double>("camera.cx");
	double cv = parameterReader.getData<double>("camera.cy");
	double f = parameterReader.getData<double>("camera.fx");
	double roix = parameterReader.getData<double>("camera.roix");
	double roiy = parameterReader.getData<double>("camera.roiy");
	double roiz = parameterReader.getData<double>("camera.roiz");
	double scale = parameterReader.getData<double>("camera.scale");

	for (int v = 0; v < img_l.rows; ++v)
	{
		ushort* depth_img_ptr = frame->depth.ptr<ushort>(v);
		const short* disparity_ptr = disp_sgbm.ptr<short>(v);
		for (int u = 0; u < img_l.cols; ++u)
		{
			short d = disparity_ptr[u];
			if (fabs(d)>FLT_EPSILON) //remove moving objects and outside the ROI
			{
				double pw = baseline/(1.0*static_cast<double>(d));
				double px = ((static_cast<double>(u)-cu)*pw)*16.0f;
				double py = ((static_cast<double>(v)-cv)*pw)*16.0f;
				double pz = (f*pw)*16.0f;

		    		if (fabs(d-minDisparity) <= FLT_EPSILON)
		    			continue;
    				if (fabs(px)<roix && fabs(py)<roiy && fabs(pz)<roiz && pz>0)//outside the ROI
					depth_img_ptr[u] = ushort(pz * scale);
			}
		}
	}

	// Semantic now
//	cv::Mat new_frame;
//	cv::resize(frame->rgb, new_frame, cv::Size(480,360));
//	std::vector<Prediction> predictions = classifier.Classify(new_frame);
//	cv::Mat segnet(new_frame.size(), CV_8UC3, cv::Scalar(0,0,0));
//	for (int i = 0; i < 360; ++i)
//	{	
//		uchar* segnet_ptr = segnet.ptr<uchar>(i);
//		for (int j = 0; j < 480; ++j)
//		{
//			segnet_ptr[j*3+0] = predictions[i*480+j].second;
//			segnet_ptr[j*3+1] = predictions[i*480+j].second;
//			segnet_ptr[j*3+2] = predictions[i*480+j].second;
//		}
//	}
//	resize(segnet, segnet, frame->rgb.size());
//	cv::cvtColor(segnet, frame->raw_semantic, CV_BGR2GRAY);;
//	cv::LUT(segnet, frame->color, segnet);
//	frame->semantic = segnet.clone();
		
		frame->semantic = cv::imread(dataset_dir+"segnet_0/"+segnet_2[currentIndex+1]);
        frame->result   = cv::imread(dataset_dir+"result_0/"+segnet_2[currentIndex+1]);
        //这里符号少掉了导致路径错误报错：Signal: SIGSEGV (Segmentation fault)

	// Semantic pre_r
//	cv::resize(frame->rgb_pre_r, new_frame, cv::Size(480,360));
//	std::vector<Prediction> predictions_pre_r = classifier.Classify(new_frame);
//	cv::Mat segnet_pre_r(new_frame.size(), CV_8UC3, cv::Scalar(0,0,0));
//	for (int i = 0; i < 360; ++i)
//	{	
//		uchar* segnet_ptr = segnet_pre_r.ptr<uchar>(i);
//		for (int j = 0; j < 480; ++j)
//		{
//			segnet_ptr[j*3+0] = predictions_pre_r[i*480+j].second;
//			segnet_ptr[j*3+1] = predictions_pre_r[i*480+j].second;
//			segnet_ptr[j*3+2] = predictions_pre_r[i*480+j].second;
//		}
//	}
//	resize(segnet_pre_r, segnet_pre_r, frame->rgb_pre_r.size());
//	cv::LUT(segnet_pre_r, frame->color, segnet_pre_r);
//	frame->semantic_pre_r = segnet_pre_r.clone();
		
		frame->semantic_pre_r =  cv::imread(dataset_dir+"segnet_1/"+segnet_3[currentIndex]);

	// Semantic cur_r
//	cv::resize(frame->rgb_cur_r, new_frame, cv::Size(480,360));
//	std::vector<Prediction> predictions_cur_r = classifier.Classify(new_frame);
//	cv::Mat segnet_cur_r(new_frame.size(), CV_8UC3, cv::Scalar(0,0,0));
//	for (int i = 0; i < 360; ++i)
//	{	
//		uchar* segnet_ptr = segnet_cur_r.ptr<uchar>(i);
//		for (int j = 0; j < 480; ++j)
//		{
//			segnet_ptr[j*3+0] = predictions_cur_r[i*480+j].second;
//			segnet_ptr[j*3+1] = predictions_cur_r[i*480+j].second;
//			segnet_ptr[j*3+2] = predictions_cur_r[i*480+j].second;
//		}
//	}
//	resize(segnet_cur_r, segnet_cur_r, frame->rgb_cur_r.size());
//	cv::LUT(segnet_cur_r, frame->color, segnet_cur_r);
//	frame->semantic_cur_r = segnet_cur_r.clone();
		
		frame->semantic_cur_r =  cv::imread(dataset_dir+"segnet_1/"+segnet_3[currentIndex+1]);

        if (frame->rgb.data == nullptr || frame->depth.data==nullptr)
        {
            // 数据不存在
  	    cout << "No data found." << endl;
            return nullptr;
        }
        frame->camera = this->camera;
        currentIndex ++;
        return frame;
    }
    default:
        break;
    }

    return nullptr;
}

void FrameReader::init_tum( const ParameterReader& para )
{
    dataset_dir = parameterReader.getData<string>("data_source");
    string  associate_file  =   dataset_dir+"/associate.txt";
    ifstream    fin(associate_file.c_str());
    if (!fin)
    {
        cerr<<"找不着assciate.txt啊！在tum数据集中这尼玛是必须的啊!"<<endl;
        cerr<<"请用python assicate.py rgb.txt depth.txt > associate.txt生成一个associate文件，再来跑这个程序！"<<endl;
        return;
    }

    while( !fin.eof() )
    {
        string rgbTime, rgbFile, depthTime, depthFile;
        fin>>rgbTime>>rgbFile>>depthTime>>depthFile;
        if ( !fin.good() )
        {
            break;
        }
        rgbFiles.push_back( rgbFile );
        depthFiles.push_back( depthFile );
    }

    cout<<"一共找着了"<<rgbFiles.size()<<"个数据记录哦！"<<endl;
    camera = parameterReader.getCamera();
    start_index = parameterReader.getData<int>("start_index");
    currentIndex = start_index;
}

void FrameReader::init_kitti( const ParameterReader& para )
{
    dataset_dir = parameterReader.getData<string>("data_source");
    string rgb_dir = parameterReader.getData<string>("rgb_dir"); 
    string index_dir = dataset_dir + rgb_dir;

    //number of files(count)
    char directory[256]; 
    strcpy(directory, index_dir.c_str()); 
    struct dirent *de;
    DIR *dir = opendir(directory);
    int count = 0;	
    while ((de = readdir(dir))) ++count;
    closedir(dir);

    // linux 
    count = count - 2;

    char rgbFile[256];
    char depthFile[256];
	char segnetFile[256];
    for (int i = 0; i < count; i++)
    {
	    sprintf(rgbFile, "%06d.png", i);
	    sprintf(depthFile, "%06d.png", i);
		sprintf(segnetFile,"%06d.png",i);
        rgbFiles.push_back( rgbFile );
        depthFiles.push_back( depthFile );
		segnet_2.push_back(segnetFile);
		segnet_3.push_back(segnetFile);
    }

    cout<<"一共找着了"<<rgbFiles.size()<<"个数据记录哦！"<<endl;
    camera = parameterReader.getCamera();
    start_index = parameterReader.getData<int>("start_index");
    currentIndex = start_index;
}
