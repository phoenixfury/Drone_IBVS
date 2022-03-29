cmake_minimum_required(VERSION 2.8.3)
project(drone_ibvs)
add_definitions(-std=c++11)
## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy sensor_msgs std_msgs genmsg geometry_msgs fiducial_msgs mavros_msgs cv_bridge image_transport ibvs_msgs)

find_package(Armadillo REQUIRED)
include_directories(include ${ARMADILLO_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

## Declare a catkin package
catkin_package()

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})

#add_executable(sub_ibvs src/sub_ibvs.cpp src/pub_ibvs.cpp)
#target_link_libraries(sub_ibvs ${catkin_LIBRARIES})
#add_dependencies(sub_ibvs ${${PROJECT_NAME}_EXPORTED_TARGETS}
 #                ${catkin_EXPORTED_TARGETS})

add_executable(sub_ibvs src/sub_ibvs.cpp)
target_link_libraries(sub_ibvs ${catkin_LIBRARIES} ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(sub_ibvs ${${PROJECT_NAME}_EXPORTED_TARGETS}
                 ${catkin_EXPORTED_TARGETS} )

add_executable(pub_ibvs src/pub_ibvs.cpp)
target_link_libraries(pub_ibvs ${catkin_LIBRARIES} ${ARMADILLO_LIBRARIES})
add_dependencies(pub_ibvs ${${PROJECT_NAME}_EXPORTED_TARGETS}
                 ${catkin_EXPORTED_TARGETS})
#add_dependencies(sub_ibvs beginner_tutorials_generate_messages_cpp)
