#ifndef TRACK_H
#define TRACK_H

/**
  * the tracker
  * Tracker跟踪当前输入的帧, 在丢失的时候进行重定位
  * 单纯使用tracker时可能会漂移，它需要后端pose graph提供一个重定位
  * 
  * 算法描述
  * Tracker内部有初始化、跟踪成功和丢失三个态
  * 每次用updateFrame放入一个新帧，记为F
  * Tracker把新帧F与它自己维护的一个候选队列里的参考帧相匹配。建立Local bundle adjustment.
  * Local BA是一个简易的pose graph.
  * 匹配成功时，把当前帧作为新的候选帧，放到候选队列末尾。如果队列太长则删除多余的帧。
  * 队列中的候选帧有一定间隔。
  * 如果整个候选队列都没匹配上，认为此帧丢失。开始丢失计数。
  * 丢失计数大于一定值时进入丢失态。这样做是为了防止因抖动等原因产生的模糊。
  * 如果进入丢失态，可能有两种情况：暂时性遮挡或kidnapped。
  * 对于遮挡，最好的策略是原地等待； 对于kidnapped，则应该重置并通知pose graph，使用回环检测来确定全局位置。
  *  
  */

// libviso2
#include "quadmatcher.hpp"
#include "vo.hpp"
#include "vo_stereo.hpp"
#include "matrix_.h"

#include "common_headers.h"
#include "parameter_reader.h"
#include "rgbdframe.h"
#include "orb.h"
#include "pnp.h"

#include <thread>
#include <mutex>
#include <functional>

// motion
#include "basicStructure.hpp"
#include "vo_stereo.hpp"
#include "uvdisparity.hpp"

using namespace rgbd_tutor;

namespace rgbd_tutor {

class PoseGraph;

class Tracker
{
    
public:
    typedef shared_ptr<Tracker> Ptr;
    enum    trackerState
    {
        NOT_READY=0,
        OK,
        LOST
    };

public:
    //  公共接口
    Tracker( const rgbd_tutor::ParameterReader& para, VisualOdometryStereo::parameters param ) :
        parameterReader( para ), viso( param )
    {
        orb = make_shared<rgbd_tutor::OrbFeature> (para);
        pnp = make_shared<rgbd_tutor::PnPSolver> (para, *orb);
        max_lost_frame = para.getData<int>("tracker_max_lost_frame");
        refFramesSize = para.getData<int>("tracker_ref_frames");

        pose = Matrix_::eye(4); //visual odometry
        poseChanged = Matrix_::eye(4); //visual odometry

        // motion
        double baseline = parameterReader.getData<double>("camera.baseline");
        double cu = parameterReader.getData<double>("camera.cx");
        double cv = parameterReader.getData<double>("camera.cy");
        double f = parameterReader.getData<double>("camera.fx");
        double roix = (int)parameterReader.getData<double>("camera.roix");
        double roiy = (int)parameterReader.getData<double>("camera.roiy");
        double roiz = (int)parameterReader.getData<double>("camera.roiz");

        roi_3d.x_max = roix; //ROI 感兴趣区域
        roi_3d.y_max = roiy;
        roi_3d.z_max = roiz;
        /**
         * camera.roix=20
         * camera.roiy=5
         * camera.roiz=30
         * */
        calib_.f = f;
        calib_.c_x = cu;
        calib_.c_y = cv;
        calib_.b = baseline;

        uv_disparity.SetCalibPars(calib_);  //给出相机标定参数
        uv_disparity.SetROI3D(roi_3d);        //划定感兴趣区域 ---ROI
        uv_disparity.SetOutThreshold(6.0f);   //outlier rejection threshold  局外点否定范围
        uv_disparity.SetInlierTolerance(3);   //set inlier tolerance 局内点的容差范围
        uv_disparity.SetMinAdjustIntense(20); //设置分割的最小调整强度
    }

    void setPoseGraph( shared_ptr<PoseGraph> poseGraph_ )
    {
        this->poseGraph = poseGraph_;
    }
    //  放入一个新帧，返回此帧所在的姿态
    Eigen::Isometry3d    updateFrame( RGBDFrame::Ptr& newFrame );

    trackerState getState() const { return state; }

    // adjust the frame according to the given ref frame
    bool    adjust( const RGBDFrame::Ptr& ref )
    {
        unique_lock<mutex> lck(adjustMutex);
        cout<<"adjust frame frame "<<ref->id<<" to "<<currentFrame->id<<endl;
        PNP_INFORMATION info;
        if (pnp->solvePnPLazy( ref, currentFrame, info) == true)
        {
            currentFrame->setTransform( info.T*ref->getTransform() );
            refFrames.clear();
            refFrames.push_back(ref);
            cntLost = 0; 
            state = OK;
            cout<<"adjust ok"<<endl;
            return true;
        }
        cout<<"adjust failed."<<endl;
        return false;
    }


    // motion
    ROI3D roi_3d = ROI3D(30,10,30);  
    CalibPars calib_;
    UVDisparity uv_disparity;
    double pitch1, pitch2;

protected:

    // 私有方法
    // 第一帧时初始化，对新帧提取特征并作为refframe
    void    initFirstFrame( );
    // 正常的track，比较current和refframe
    void    trackRefFrame();
    // 丢失恢复
    void    lostRecover();

protected:
    // 数据

    // 当前帧
    rgbd_tutor::RGBDFrame::Ptr  currentFrame    =nullptr;

    // 当前帧的参考帧
    // 队列结构，长度固定
    deque< RGBDFrame::Ptr >  refFrames;
    int refFramesSize   =5;
    Eigen::Isometry3d   lastPose = Eigen::Isometry3d::Identity();
    
    // 速度
    Eigen::Isometry3d   speed;

    // 参数
    const rgbd_tutor::ParameterReader&   parameterReader;

    // 状态
    trackerState    state   =NOT_READY;

    // 参数
    int     cntLost = 0;
    int     max_lost_frame = 5;

    // pose graph 
    shared_ptr<PoseGraph>   poseGraph =nullptr;
    mutex   adjustMutex;

public:
    // libviso2
    Matrix_ pose = Matrix_::eye(4); //visual odometry
    Matrix_ poseChanged = Matrix_::eye(4); //visual odometry
    VisualOdometryStereo viso;
    void estimateVO();

protected:
    // 其他用途
    shared_ptr<rgbd_tutor::PnPSolver>   pnp;
    shared_ptr<rgbd_tutor::OrbFeature>  orb;

};

}



#endif // TRACK_H
