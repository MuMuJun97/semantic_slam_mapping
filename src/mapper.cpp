#include "mapper.h"

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/filters/radius_outlier_removal.h>  
//#include <pcl/filters/statistical_outlier_removal.h>

using namespace rgbd_tutor;

Mapper::PointCloud::Ptr Mapper::generatePointCloud( const RGBDFrame::Ptr &frame )
{
    semantic_motion_fuse(frame);

    PointCloud::Ptr tmp( new PointCloud() );
    if ( frame->pointcloud == nullptr )
    {
        // point cloud is null ptr
        frame->pointcloud = boost::make_shared<PointCloud>();
#pragma omp parallel for
        for ( int m=0; m<frame->depth.rows; m+=1 )
        {
	        uchar* motion_ptr = moving_mask.ptr<uchar>(m);
            for ( int n=0; n<frame->depth.cols; n+=1 )
            {
                ushort d = frame->depth.ptr<ushort>(m)[n];
                if (d == 0)
                    continue;  //深度为0，不考虑
                if (d > max_distance * frame->camera.scale)
                    continue; //距离较远的点，不考虑
	        	if (motion_ptr[n] == 255)
		            continue; //移动物体的点，不考虑

                PointT p;
                cv::Point3f p_cv = frame->project2dTo3d(n, m);
                p.b = frame->semantic.ptr<uchar>(m)[n*3];
                p.g = frame->semantic.ptr<uchar>(m)[n*3+1];
                p.r = frame->semantic.ptr<uchar>(m)[n*3+2];

                if (
                        (p.b==128 && p.g==128 && p.r==128) || //sky
                        (p.b==128 && p.g==192 && p.r==192) || // pole
                       // (p.b==0   && p.g==69 && p.r==255)  || // Road_marking
                       // (p.b==128 && p.g==128 && p.r==192) || // SignSymbol
                       // (p.b==128 && p.g==64 && p.r==64)   || // fence
                       // (p.b==0   && p.g==64 && p.r==64)   || // pedestrian
                        (p.b==192 && p.g==128 && p.r==0)  //||    //cyclist

                        //(p.b==128 && p.g==0 && p.r==64)    // car
                      //(p.b==128 && p.g==64 && p.r==128) || // road
                      //(p.b==222 && p.g==40 && p.r==60)  || // Pavement
                      //(p.b==0   && p.g==128 && p.r==128)|| // Tree
                      //(p.b==0   && p.g==0 && p.r==128)  || //building
                    ) continue;

                p.x = p_cv.x;
                p.y = p_cv.y;
                p.z = p_cv.z;

                /////////////////////////////////////
                PointT point_result;//这里是投影点的空间坐标
                point_result.b = frame->result.ptr<uchar>(m)[n*3];
                point_result.g = frame->result.ptr<uchar>(m)[n*3+1];
                point_result.r = frame->result.ptr<uchar>(m)[n*3+2];
                point_result.x = p_cv.x;
                point_result.y = p_cv.y;
                point_result.z = p_cv.z;

                /////////////////////////////////////
                /////////////////////////////////////
                PointT point_img;//这里是投影点的空间坐标
                point_img.b = frame->rgb.ptr<uchar>(m)[n*3];
                point_img.g = frame->rgb.ptr<uchar>(m)[n*3+1];
                point_img.r = frame->rgb.ptr<uchar>(m)[n*3+2];
                point_img.x = p_cv.x;
                point_img.y = p_cv.y;
                point_img.z = p_cv.z;

                /////////////////////////////////////

                //frame->pointcloud->points.push_back( p );
                //frame->pointcloud->points.push_back(point_result);
                frame->pointcloud->points.push_back(point_img);
            }
        }
    }

    //Eigen::Isometry3d T = frame->getTransform().inverse();
    Eigen::Isometry3d T = frame->getTransform();
    pcl::transformPointCloud( *frame->pointcloud, *tmp, T.matrix());
    tmp->is_dense = false;
    return tmp;
}

void Mapper::viewer()
{
    pcl::visualization::CloudViewer viewer("viewer");

    //pcl::visualization::PCLVisualizer view("map");
    //view.setBackgroundColor(255,255,255);

    PointCloud::Ptr globalMap (new PointCloud);

 
    pcl::VoxelGrid<PointT>	voxel;
    voxel.setLeafSize( resolution, resolution, resolution );

    while (shutdownFlag == false)
    {
        boost::timer timer;

        static int cntGlobalUpdate = 0;
        if ( poseGraph.keyframes.size() <= this->keyframe_size )
        {
            usleep(1000);
            continue;
        }
        // keyframe is updated
        PointCloud::Ptr	tmp(new PointCloud());
        if (cntGlobalUpdate % 15 == 0)
        {
            // update all frames
            cout<<"redrawing frames"<<endl;
            globalMap->clear();
            for ( int i=0; i<poseGraph.keyframes.size(); i+=2 )
            {
                PointCloud::Ptr cloud = this->generatePointCloud(poseGraph.keyframes[i]);
                *globalMap += *cloud;
            }
        }
        else
        {
            for ( int i=poseGraph.keyframes.size()-1; i>=0 && i>poseGraph.keyframes.size()-6; i-- )
            {
                PointCloud::Ptr cloud = this->generatePointCloud(poseGraph.keyframes[i]);
/*
                // filter
        		PointCloud::Ptr cloud_out_filtered (new PointCloud());
        		pcl::StatisticalOutlierRemoval<PointT> sor;
        		sor.setInputCloud(cloud);
        		sor.setMeanK(30);
        		sor.setStddevMulThresh(1.0);
        		sor.filter(*cloud_out_filtered);
                *globalMap += *cloud_out_filtered;
*/
                *globalMap += *cloud;
            }
        }

        cntGlobalUpdate ++ ;

        //voxel
        voxel.setInputCloud( globalMap );
        voxel.filter( *tmp );

        keyframe_size = poseGraph.keyframes.size();
        globalMap->swap( *tmp );
        viewer.showCloud( globalMap );

        cout << "points in global map: " << globalMap->points.size() << endl;
        cout << "Mapping cost time: " << timer.elapsed() * 1000.0 << "ms" << endl;

    }
    if(poseGraph.shutDownFlag == true)
    {
        pcl::PCDWriter writer;
        writer.write("/home/relaybot/Mu_Link/KittiData/18/map18_img.pcd", *(globalMap));
        cout << "Map saved!" << endl;
    }
//    while (!viewer.wasStopped ())
//    {
//        poseGraph.shutDownFlag == true
//    }
//    pcl::PCDWriter writer;
//    writer.write("map.pcd", *globalMap);
//    cout << "Map saved!" << endl;
}
void Mapper::SaveMap()
{
//    if(poseGraph.shutDownFlag == true)
//    {
//        pcl::PCDWriter writer;
//        writer.write("map.pcd", *(globalMap));
//        cout << "Map saved!" << endl;
//    }
}

void Mapper::semantic_motion_fuse(const RGBDFrame::Ptr &frame)
{
	moving_mask = cv::Mat::zeros(frame->semantic.size(), CV_8UC1);

	// get semantic_maybe_motion mask
	cv::Mat imgcolor = frame->semantic.clone();//语义分割图
	cv::Mat img = cv::Mat::zeros(imgcolor.size(), CV_8UC1);
	for (int i = 0; i < imgcolor.rows; i++)
	{
		uchar* imgcolor_ptr = imgcolor.ptr<uchar>(i);//语义图
		uchar* img_ptr = img.ptr<uchar>(i);
		for (int j = 0; j < imgcolor.cols; j++)
		{

			uchar pb = imgcolor_ptr[j*3];
			uchar pg = imgcolor_ptr[j*3+1];
			uchar pr = imgcolor_ptr[j*3+2];//语义图
			if ( //(pb==128 && pg==0 && pr==64) || //Car
				 (pb==0 && pg==64 && pr==64) ||  //Pedestrian
				 (pb==192 && pg==128 && pr==0) ) //Bicyclist
			    {
				   img_ptr[j] = 255;
                }
		}
	}
	cv::dilate(img, img, cv::Mat(3,3,CV_8UC1), cv::Point(-1,-1), 2);//膨胀操作

	moving_mask = img.clone();
/*
	// get motion mask
	cv::Mat motion = frame->moving_mask.clone();

	// get semantic contours mask
	std::vector<std::vector<cv::Point> > contours; 
	cv::findContours(img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	cv::Mat semantic_mask(img.size(), CV_8U, cv::Scalar(0)); 
	cv::drawContours(semantic_mask, contours, -1, cv::Scalar(255), 2); 

	// get motion_semantic result
	std::vector<cv::Mat> result_masks;
	for (int i = 0; i < contours.size(); i++)
	{
		cv::Mat mask = cv::Mat::zeros(img.size(), img.type());

		cv::Mat tmp_mask(img.size(), CV_8U, cv::Scalar(0));
		cv::drawContours(tmp_mask, contours, i, cv::Scalar(255,255,255), CV_FILLED, 8);

		if (cv::contourArea(contours[i]) > area_thres)
		{
			cv::drawContours(mask, contours, i, cv::Scalar(255,255,255), CV_FILLED, 8);
			int overlay_count = 0;
			int mask_count = 1;
			for (int v = 0; v < img.rows; v++)
			{
				uchar* motion_ptr = motion.ptr<uchar>(v);
				uchar* mask_ptr = mask.ptr<uchar>(v);
				for (int u = 0; u < img.cols; u++)
				{
					if (mask_ptr[u] == 255) 
					{
						mask_count ++;
						if (motion_ptr[u] == 255)
						{
							overlay_count ++;
						}
					}
				}
			}
			double overlay_portion = overlay_count * 1.0f / mask_count;
			//std::cout << "area_portion_" << i << ": " << overlay_portion << std::endl;
			if (overlay_portion > overlay_portion_thres)
			{
				result_masks.push_back(mask);
			}
		}
	}

	if (result_masks.size() <= 0) 
		return;
	
	for (int i = 0; i < result_masks.size(); i++)
		moving_mask += result_masks[i];
*/
}
