add_library( rgbd_tutor_lib SHARED    
    rgbdframe.cpp parameter_reader.cpp orb.cpp pnp.cpp track.cpp looper.cpp pose_graph.cpp 
    mapper.cpp stereo.cpp segnet.cpp 
    quadmatcher.cpp vo.cpp vo_stereo.cpp matrix_.cpp
)

add_library( motion_lib SHARED    
    uvdisparity.cpp vo_stereo.cpp vo.cpp quadmatcher.cpp stereo.cpp 
)

target_link_libraries( rgbd_tutor_lib
    ${OpenCV_LIBRARIES}
    ${CHOLMOD_LIBRARIES}
    ${PCL_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/Thirdparty/orbslam_modified/lib/libORB_SLAM2.so
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sim3 g2o_types_sba
)

target_link_libraries( motion_lib
    ${OpenCV_LIBRARIES}
)
