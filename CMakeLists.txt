cmake_minimum_required( VERSION 2.8 )
#cmake 工程要求的最小的cmake版本

project(  learning )

set( CMAKE_BUILD_TYPE "Debug" )
#构建类型为Debug

set( CMAKE_CXX_COMPILER "g++" )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
#list  list(APPEND <list> <element> [<element> ...])
#使用APPEND选项时，该命令将会在该list之后追加若干元素。
#注意：在CMake中，一个list是一个由封号;分割的一组字符串。
#使用set命令可以创建一个list。例如，set(var a b c d e)命令将会创建一个list：a;b;c;d;e；
#而set(var "a b c d e")命令创建的只是一个字符串，或者说是只有一个项的list。

#当编译一个需要使用第三方库的软件时，我们需要知道：
#去哪儿找头文件 .h                         #对比GCC的 -I 参数
#去哪儿找库文件 (.so/.dll/.lib/.dylib/...) #对比GCC的 -L 参数
#需要链接的库文件的名字                     #对比GCC的 -l 参数
#这也是一个 CMAKE finder 需要返回的最基本的信息。

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( PCL 1.7 REQUIRED )
find_package( Caffe REQUIRED )
find_package( G2O REQUIRED )

#添加头文件
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package( Cholmod )
include_directories( ${CHOLMOD_INCLUDE_DIR} )

include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib )

include_directories(
    ${PROJECT_SOURCE_DIR}/
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/Thirdparty/
    ${EIGEN3_INCLUDE_DIR}
)


add_subdirectory( ${PROJECT_SOURCE_DIR}/src/ )
add_subdirectory( ${PROJECT_SOURCE_DIR}/experiment/ )
