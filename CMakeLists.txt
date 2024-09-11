cmake_minimum_required(VERSION 3.5)
project(demo)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action  REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core calib3d)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(apriltag REQUIRED)
find_package(tf2 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED NO_MODULE)
find_package(nav2_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
add_executable(demo src/demo.cpp)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)

ament_target_dependencies(
  ${PROJECT_NAME}
  geometry_msgs
  rclcpp
  tf2_ros
  tf2
  nav_msgs
  image_transport
  cv_bridge
  rclcpp_action 
  tf2_geometry_msgs
  nav2_msgs
  pcl_conversions
)

target_link_libraries(demo
    apriltag::apriltag
    opencv_core
    opencv_calib3d
    Eigen3::Eigen
    yaml-cpp
    ${PCL_LIBRARIES}
)

target_include_directories(demo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS})



install(TARGETS demo
  DESTINATION lib/${PROJECT_NAME})

#install(DIRECTORY
#  ${PROJECT_SOURCE_DIR}/launch/
#  DESTINATION share/${PROJECT_NAME}/launch)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
