#include "track.h"
#include <unistd.h>
#include "readFile.h"

// tracker的主线程
using namespace rgbd_tutor;

Eigen::Isometry3d Tracker::updateFrame( RGBDFrame::Ptr& newFrame )
{
    unique_lock<mutex> lck(adjustMutex);
    currentFrame = newFrame;
    if ( state == NOT_READY )
    {
        initFirstFrame( );
        return Eigen::Isometry3d::Identity();
    }
    if ( state == OK )
    {
	    estimateVO();//在slam3d的代码里使用的是这个函数，下面的被注释
        //trackRefFrame();//下面的trackrefframe来自于高博的RGBDFrame跟踪
        return currentFrame->getTransform();
    }
    // state = LOST

    lostRecover();
    return currentFrame->getTransform();

}

void Tracker::initFirstFrame( )
{
    orb->detectFeatures( currentFrame );
    refFrames.push_back(currentFrame);
    speed = Eigen::Isometry3d::Identity();
    state = OK;
}

void Tracker::estimateVO( )
{    
	// 初始值
    	currentFrame->setTransform( speed * refFrames.back()->getTransform() );
    	orb->detectFeatures( currentFrame );
    
	// VISO2
	QuadFeatureMatch* quadmatcher = new QuadFeatureMatch(currentFrame->img_lc,
                                                         currentFrame->img_rc,
                                                         currentFrame->img_lp,
                                                         currentFrame->img_rp,
                                                         currentFrame->semantic_cur_r,
                                                         currentFrame->semantic_pre_r,
                                                         true);
	quadmatcher->init(DET_GFTT,DES_SIFT);
    //GFTT——cvGoodFeaturesToTrack 特征点
	quadmatcher->detectFeature();
	quadmatcher->circularMatching();

	//visual odometry valid
	bool success = false;
	if (viso.Process(*quadmatcher) == true)
	{
		//get ego-motion matrix (6DOF)
		cv::Mat motion;
		motion = viso.getMotion();
		//cout << "motion: " << motion << endl;

		// moving
		triangulate10D(currentFrame->img_lc,
                       currentFrame->disparity,
                       currentFrame->xyz,
                       calib_.f, calib_.c_x, calib_.c_y, calib_.b,
                       roi_3d);

		currentFrame->moving_mask = uv_disparity.Process(
                currentFrame->img_lc, //左图
                currentFrame->disparity, //SGBM得来的深度图
                viso,
                currentFrame->xyz,
                currentFrame->roi_mask,
                currentFrame->ground_mask, pitch1, pitch2);

		//visual odometry
		Matrix_ M = Matrix_::eye(4);
		for (int32_t i=0; i<4; ++i)
			for (int32_t j=0; j<4; ++j)
				M.val[i][j] = motion.at<double>(i,j);
		poseChanged = poseChanged * Matrix_::inv(M);
		pose = pose * Matrix_::inv(M);
/*	
		// gt
		cv::Mat gtpose;
		Matrix_ gt_ = Matrix_::eye(4); 
		pd.getData( n+1, gtpose );
		for (int32_t i=0; i<4; ++i)
			for (int32_t j=0; j<4; ++j)
				gt_.val[i][j] = gtpose.at<double>(i,j);
		//pose = gt_;
		gtpose_ = gt_;

		// key_frame
		float rt_change = poseChanged.l2norm();
		if (rt_change > RT_Threshold)
		{
			key_pose = key_pose * poseChanged;
			poseChanged = Matrix_::eye(4);
			key_detected = true;
			cout << "Key_frame[" << n << "]" << " is detected." << endl;
		}
*/
		success = true;
	}
	delete quadmatcher;
	
	if (!success)
	{
		cntLost ++;
		if (cntLost > max_lost_frame)
		{
		    state = LOST;
		}
        	return;
	}
	
	Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
	T(0,0) = pose.val[0][0]; T(0,1) = pose.val[0][1]; T(0,2) = pose.val[0][2]; T(0,3) = pose.val[0][3];
	T(1,0) = pose.val[1][0]; T(1,1) = pose.val[1][1]; T(1,2) = pose.val[1][2]; T(1,3) = pose.val[1][3];
	T(2,0) = pose.val[2][0]; T(2,1) = pose.val[2][1]; T(2,2) = pose.val[2][2]; T(2,3) = pose.val[2][3];
	T(3,0) = pose.val[3][0]; T(3,1) = pose.val[3][1]; T(3,2) = pose.val[3][2]; T(3,3) = pose.val[3][3];
	currentFrame->setTransform( T );
	cntLost = 0;
	speed = T * lastPose.inverse();
	lastPose = currentFrame->getTransform();
	refFrames.push_back( currentFrame );
	while (refFrames.size() > refFramesSize )
	{
		refFrames.pop_front();
	}
}


void Tracker::trackRefFrame()
{
    //adjustMutex.lock();
    // 初始值
    currentFrame->setTransform( speed * refFrames.back()->getTransform() );
    orb->detectFeatures( currentFrame );
    
    // build local BA
    vector<cv::Point3f> obj;
    vector<cv::Point2f> img;
    for (auto pFrame: refFrames)
    {
        vector<cv::DMatch> matches = orb->match( pFrame, currentFrame );
        vector<cv::DMatch>  validMatches;
        Eigen::Isometry3d invPose = pFrame->getTransform().inverse();
        for (auto m:matches)
        {
            cv::Point3f pObj = pFrame->features[m.queryIdx].position;
            if (pObj == cv::Point3f(0,0,0))
                continue;
            Eigen::Vector4d vec = invPose * Eigen::Vector4d(pObj.x, pObj.y, pObj.z,1 );
            obj.push_back( cv::Point3f(vec(0), vec(1), vec(2) ) );
            img.push_back( currentFrame->features[m.trainIdx].keypoint.pt );
        }
    }
    
    if ( img.size() < 15 )
    {
        cntLost ++;
        if (cntLost > max_lost_frame)
        {
            state = LOST;
        }
        return;
    }
    
    vector<int> inlierIndex;
    Eigen::Isometry3d T = speed * lastPose;
    bool b = pnp->solvePnP( img, obj, currentFrame->camera, inlierIndex, T );
    if ( inlierIndex.size() < 15 )
    {
        cntLost ++;
        if (cntLost > max_lost_frame)
        {
            state = LOST;
        }
        return;
    }
    
    currentFrame->setTransform( T );
    cntLost = 0;
    speed = T * lastPose.inverse();
    lastPose = currentFrame->getTransform();
    refFrames.push_back( currentFrame );
    while (refFrames.size() > refFramesSize )
    {
        refFrames.pop_front();
    }
    
    //cout<<"speed="<<endl<<speed.matrix()<<endl;
}

void    Tracker::lostRecover()
{
    cout<<"trying to recover from lost"<<endl;
    orb->detectFeatures( currentFrame );
    currentFrame->setTransform( refFrames.back()->getTransform() );
    refFrames.clear();
    refFrames.push_back( currentFrame );
    state = OK;
    cntLost = 0;
    cout<<"recover returned"<<endl;
}
