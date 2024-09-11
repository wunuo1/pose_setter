#include "nav_loc/nav_loc_node.hpp"
#include "nav_loc/fun.hpp"
#include "nav_loc/con.hpp"
#include <opencv2/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <thread>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

using NavigateToPose = nav2_msgs::action::NavigateToPose;
using GoalHandleNavigateToPose = rclcpp_action::ClientGoalHandle<NavigateToPose>;

geometry_msgs::msg::Transform
pnp(apriltag_detection_t* const detection, const std::array<double, 4>& intr, double tagsize)
{
    const std::vector<cv::Point3d> objectPoints{
        {-tagsize / 2, -tagsize / 2, 0},
        {+tagsize / 2, -tagsize / 2, 0},
        {+tagsize / 2, +tagsize / 2, 0},
        {-tagsize / 2, +tagsize / 2, 0},
    };

    const std::vector<cv::Point2d> imagePoints{
        {detection->p[0][0], detection->p[0][1]},
        {detection->p[1][0], detection->p[1][1]},
        {detection->p[2][0], detection->p[2][1]},
        {detection->p[3][0], detection->p[3][1]},
    };

    cv::Matx33d cameraMatrix;
    cameraMatrix(0, 0) = intr[0];// fx
    cameraMatrix(1, 1) = intr[1];// fy
    cameraMatrix(0, 2) = intr[2];// cx
    cameraMatrix(1, 2) = intr[3];// cy

    cv::Mat rvec, tvec;
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, {}, rvec, tvec);

    return tf2::toMsg<std::pair<cv::Mat_<double>, cv::Mat_<double>>, geometry_msgs::msg::Transform>(std::make_pair(tvec, rvec));
}

tf2::Transform transformMsgToTF2(const geometry_msgs::msg::Transform& transform_msg) {
    tf2::Transform tf2_transform;

    // 设置平移部分
    tf2_transform.setOrigin(tf2::Vector3(
        transform_msg.translation.x,
        transform_msg.translation.y,
        transform_msg.translation.z
    ));

    // 设置旋转部分
    tf2::Quaternion quaternion;
    quaternion.setX(transform_msg.rotation.x);
    quaternion.setY(transform_msg.rotation.y);
    quaternion.setZ(transform_msg.rotation.z);
    quaternion.setW(transform_msg.rotation.w);
    tf2_transform.setRotation(quaternion);

    return tf2_transform;
}

void p_tf(tf2::Transform &transform){
    tf2::Vector3 origin = transform.getOrigin();
    std::cout << "Origin: (" << origin.x() << ", "
              << origin.y() << ", "
              << origin.z() << ")" << std::endl;

    // 获取旋转部分（四元数）
    tf2::Quaternion rotation = transform.getRotation();
    std::cout << "Rotation (quaternion): (" << rotation.x() << ", "
              << rotation.y() << ", "
              << rotation.z() << ", "
              << rotation.w() << ")" << std::endl;  
}

NavLocNode::NavLocNode(const std::string& node_name, const rclcpp::NodeOptions& options)
    : rclcpp::Node(node_name, options)
{
    this->declare_parameter<bool>("build_map", build_map_);

    this->get_parameter<bool>("build_map", build_map_);
    init_param();

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    // callback_group_ = create_callback_group(
    //     rclcpp::CallbackGroupType::MutuallyExclusive);
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(),
        this->get_node_timers_interface());
    tf_buffer_->setCreateTimerInterface(timer_interface);
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    init_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("initialpose", 10);
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("initial_pose_tmp", 10);
    std::string topic;
    if(build_map_ == true){
        topic = "/nebula200/mtof_rgb/image_raw";
    } else {
        topic = "/nebula200/stof_rgb/image_raw";
    }
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                topic, 10, std::bind(&NavLocNode::sub_cam_callback, this, std::placeholders::_1));
    client_ptr_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");
    tf_ = tag36h11_create();
    td_ = apriltag_detector_create();
    goal_msg_ = NavigateToPose::Goal();
    apriltag_detector_add_family(td_, tf_);

    map_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("point_cloud_map", 10);
    timer_ = this->create_wall_timer(
      std::chrono::seconds(5),
      [this]() { this->publish_point_cloud(); }
    );
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("/root/amr_ws/src/demo/output/final-voxel.pcd", pcl_cloud_) == -1)
    {
        RCLCPP_ERROR(this->get_logger(), "Couldn't read PCD file");
        return;
    }

    send_goal_options.goal_response_callback =
        std::bind(&NavLocNode::goal_response_callback, this, std::placeholders::_1);
    send_goal_options.feedback_callback =
        std::bind(&NavLocNode::feedback_callback, this, std::placeholders::_1, std::placeholders::_2);
    send_goal_options.result_callback =
        std::bind(&NavLocNode::result_callback, this, std::placeholders::_1);
    if (!client_) {
        client_ = std::make_shared<std::thread>(
            std::bind(&NavLocNode::client_control, this));
    }
}

NavLocNode::~NavLocNode(){
    process_stop_ = true;
    has_pub_ = true;
    cv_.notify_all();
    task_cv_.notify_all();
    if (client_ && client_->joinable()) {
        client_->join();
        client_ = nullptr;
    }
}


void NavLocNode::init_param(){
    Eigen::Matrix3d tof_imu;
    tof_imu << -0.993063627638373553564 ,0.052518963674774226572 ,-0.01498737732389914978, 
            -0.033814436045461613632 ,-0.016796566829113524293 ,-0.99928717438014507495, 
            -0.00801808196977665167 ,-0.99847864521832871386  ,0.03465425204866725379;
    Eigen::Quaterniond tof_imu_qua(tof_imu.inverse());
    tf2::Quaternion q_imu_tof(tof_imu_qua.x(), tof_imu_qua.y(), tof_imu_qua.z(), tof_imu_qua.w());
    tf2::Vector3 t_imu_tof(-0.0039799203111821468, -0.19001955020902342461, -0.027189542185995823831);
    tf2::Transform imu_tof_tf(q_imu_tof, t_imu_tof);
    imu_tof_tf_ = imu_tof_tf;

    Eigen::Matrix3d rob_imu;
    rob_imu << 0.052268869631951098023, -0.93852017660344945467, 0.033846560965168889575, 
                0.93808656435923075909, -0.051737213472478319612, 0.015014368436287678477, 
                0.033019704102458273174	, 0.01675474093936925155, 0.99931446977489903968;
    Eigen::Quaterniond rob_imu_qua(rob_imu);
    tf2::Quaternion q_rob_imu(rob_imu_qua.x(), rob_imu_qua.y(), rob_imu_qua.z(), rob_imu_qua.w());
    tf2::Vector3 t_rob_imu(0.12588064308116462841, 0.008162559660163853708, 0.20705414746130172184);
    tf2::Transform rob_imu_tf(q_rob_imu, t_rob_imu);
    rob_imu_tf_ = rob_imu_tf;
    
    Eigen::Matrix3d tof_rgb;
    tof_rgb << 0.99977, -0.00463365, -0.0209421, 
                0.00463365, 0.99999, 0.0000363365, 
                0.0209402, -0.00463365, 0.99977;
    Eigen::Quaterniond tof_rgb_qua(tof_rgb);
    tf2::Quaternion q_tof_rgb(tof_rgb_qua.x(), tof_rgb_qua.y(), tof_rgb_qua.z(), tof_rgb_qua.w());
    tf2::Vector3 t_tof_rgb(0.0110838, -0.00124392, -0.00186265);
    tf2::Transform tof_rgb_tf(q_tof_rgb, t_tof_rgb);
    tof_rgb_tf_ = tof_rgb_tf;

    imu_rgb_tf_ = imu_tof_tf_ * tof_rgb_tf_;
    rob_rgb_tf_ = rob_imu_tf_ * imu_rgb_tf_;

    if(build_map_ == false) {
        YAML::Node config = YAML::LoadFile("/root/amr_ws/src/demo/output/config.yaml");
        if (config["build_map"]["map_tag"]) {
            std::vector<double> map_tag_v = config["build_map"]["map_tag"].as<std::vector<double>>();
            tf2::Quaternion q_map_tag(map_tag_v[3], map_tag_v[4], map_tag_v[5], map_tag_v[6]);
            tf2::Vector3 t_map_tag(map_tag_v[0], map_tag_v[1], map_tag_v[2]);
            tf2::Transform map_tag_tf(q_map_tag, t_map_tag);
            map_tag_tf_ = map_tag_tf;
        } else {
            std::cerr << "Failed to load array 'a' from config.yaml." << std::endl;
        }

        if (config["build_map"]["nav_point1"]) {
            for (const auto& value : config["build_map"]["nav_point1"]) {
                point1_vec_.push_back(value.as<float>());
            }
        } else {
            std::cerr << "Failed to load array 'a' from config.yaml." << std::endl;
        }
        if (config["build_map"]["nav_point2"]) {
            for (const auto& value : config["build_map"]["nav_point2"]) {
                point2_vec_.push_back(value.as<float>());
            }
        } else {
            std::cerr << "Failed to load array 'a' from config.yaml." << std::endl;
        }
        if (config["build_map"]["nav_point3"]) {
            for (const auto& value : config["build_map"]["nav_point3"]) {
                point3_vec_.push_back(value.as<float>());
            }
        } else {
            std::cerr << "Failed to load array 'a' from config.yaml." << std::endl;
        }
        // std::cout<<point2_vec_.size()<<std::endl;
        // std::cout<<point2_vec_[0]<<" "<<point2_vec_[1]<<" "<<point2_vec_[2]<<std::endl;
    }
}

void NavLocNode::mean_transform(const std::vector<tf2::Transform> &tf_tmp)
{
    static tf2::Vector3 mean_translation(0.0, 0.0, 0.0);
    // 四元数部分
    static tf2::Quaternion mean_quaternion(0.0, 0.0, 0.0, 0.0);
    for (const auto& transform : tf_tmp) {
        mean_translation += transform.getOrigin();
        mean_quaternion += transform.getRotation();
    }
    mean_translation /= tf_tmp.size();
    mean_quaternion /= tf_tmp.size();
    if(build_map_ == true){
        YAML::Node node;
        std::vector<double> map_tag = {mean_translation.x(), mean_translation.y(), mean_translation.z(),
        mean_quaternion.x(), mean_quaternion.y(), mean_quaternion.z(), mean_quaternion.w()};
        node["build_map"]["map_tag"] = map_tag;
        std::ofstream fout("/root/amr_ws/src/demo/output/config.yaml");
        fout << node;
        fout.close();
        std::cout << "Array has been written!" << std::endl;
    } else {
        map_rob_tf_.setOrigin(mean_translation);
        map_rob_tf_.setRotation(mean_quaternion);
    }

}

void NavLocNode::sub_cam_callback(const sensor_msgs::msg::Image::SharedPtr msg){
    if(has_pub_ == false)
    {       
        cv::Mat image = cv_bridge::toCvCopy(msg, "mono8")->image;
        const std::array<double, 4> intrinsics = {418.06, 417.369, 476.969, 276.035};
        // 创建AprilTag检测的图像结构
        image_u8_t im = {.width = image.cols,
                            .height = image.rows,
                            .stride = image.cols,
                            .buf = image.data};

        // 进行AprilTag检测
        zarray_t *detections = apriltag_detector_detect(td_, &im);
        // 遍历检测到的AprilTags并打印信息
        geometry_msgs::msg::TransformStamped tf_tmp;
        for (int i = 0; i < zarray_size(detections); i++)
        {
            apriltag_detection_t *det;
            zarray_get(detections, 0, &det);
            RCLCPP_INFO(this->get_logger(), "Detected tag ID: %d", det->id);
            // 这里可以添加更多处理检测结果的代码
            tf_tmp.header = msg->header;
            // set child frame name by generic tag name or configured tag name
            tf_tmp.child_frame_id = "tag";
            tf_tmp.transform = pnp(det, intrinsics, 0.144);
            
        }
        if (zarray_size(detections) > 0){
            tf2::Transform rgb_tag_tf = transformMsgToTF2(tf_tmp.transform);
            tf2::Transform imu_tag_tf = imu_rgb_tf_ * rgb_tag_tf;

            if(build_map_ == true){
                mean_tf_v_.push_back(imu_tag_tf);
                if(mean_tf_v_.size() == mean_num_){
                    mean_transform(mean_tf_v_);
                }
            } else {
                tf2::Transform map_rob_tf = map_tag_tf_ * rgb_tag_tf.inverse() * rob_rgb_tf_.inverse();
                mean_tf_v_.push_back(map_rob_tf);
                std::cout<<mean_tf_v_.size()<<std::endl;
                if(mean_tf_v_.size() == mean_num_){
                    mean_transform(mean_tf_v_);
                    geometry_msgs::msg::PoseWithCovarianceStamped pose_with_covariance;
                    tf2::toMsg(map_rob_tf_, pose_with_covariance.pose.pose);
                    // 设置时间戳和坐标系
                    pose_with_covariance.header.stamp = msg->header.stamp;
                    pose_with_covariance.header.frame_id = "map";
                    pose_with_covariance.pose.covariance = std::array<double, 36>{0.0};
                    if(pub_inital_post_ == false){
                        init_pose_pub_->publish(pose_with_covariance);
                        pub_inital_post_ = true;
                    } else {
                        geometry_msgs::msg::TransformStamped transform_stamped;
                        try
                        {
                            transform_stamped = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
                            pose_with_covariance.pose.pose.position.x = transform_stamped.transform.translation.x;
                            pose_with_covariance.pose.pose.position.y = transform_stamped.transform.translation.y;
                            // pose_with_covariance.pose.pose.position.z = transform_stamped.transform.translation.z;
                            pose_with_covariance.pose.pose.orientation = transform_stamped.transform.rotation;
                            init_pose_pub_->publish(pose_with_covariance);
                        }
                        catch (tf2::TransformException &ex)
                        {
                            RCLCPP_WARN(this->get_logger(), "Could not transform: %s", ex.what());
                        }
                        // pose_pub_->publish(pose_with_covariance);
                    }
                    
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    {
                        init_pose_pub_->publish(pose_with_covariance);
                        has_pub_ = true;
                        cv_.notify_all();
                    }
                    std::cout<<"sent success"<<std::endl;
                    //std::this_thread::sleep_for(std::chrono::seconds(5));
                    p_tf(map_rob_tf_);
                }
            }
        }

        //tf_broadcaster_->sendTransform(tf_tmp);
        apriltag_detections_destroy(detections);
    }
}

void NavLocNode::publish_point_cloud()
{
// Load PCD file

    // Convert PCL point cloud to ROS message
    sensor_msgs::msg::PointCloud2 ros_cloud;
    pcl::toROSMsg(pcl_cloud_, ros_cloud);

    // Set header information
    ros_cloud.header.frame_id = "map";
    ros_cloud.header.stamp = this->get_clock()->now();

    // Publish the point cloud message
    map_publisher_->publish(ros_cloud);
}

void NavLocNode::goal_response_callback(std::shared_ptr<GoalHandleNavigateToPose> future)
{
    auto goal_handle = future.get();
    if (!goal_handle) {
        has_pub_ = false;
        pub_inital_post_ = false;
        mean_tf_v_.clear();
        RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
}

void NavLocNode::feedback_callback(
    GoalHandleNavigateToPose::SharedPtr,
    const std::shared_ptr<const NavigateToPose::Feedback> feedback)
{
    static int num = 0;
    if(num == 30){
        RCLCPP_INFO(this->get_logger(), "Current progress: (%.2f, %.2f)",
                    feedback->current_pose.pose.position.x,
                    feedback->current_pose.pose.position.y);
        num = 0;
    }
    num++;
}

void NavLocNode::result_callback(const GoalHandleNavigateToPose::WrappedResult & result)
{
    switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
            RCLCPP_INFO(this->get_logger(), "Goal was reached");
            std::this_thread::sleep_for(std::chrono::seconds(5));
            task_done_ = true;
            task_cv_.notify_all();
            break;
        case rclcpp_action::ResultCode::ABORTED:
            this->client_ptr_->async_send_goal(goal_msg_, send_goal_options);
            RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
            break;
        case rclcpp_action::ResultCode::CANCELED:
            RCLCPP_WARN(this->get_logger(), "Goal was canceled");
            break;
        default:
            RCLCPP_ERROR(this->get_logger(), "Unknown result code");
            break;
    }
}

void NavLocNode::client_control(){
    std::cout<<"--------------------"<<std::endl;

    std::unique_lock<std::mutex> lock(mtx_);
    while (process_stop_ == false){
        // 等待条件变量通知
        cv_.wait(lock, [&]{ return has_pub_; });
        if(process_stop_ == false){
            // 执行任务
            std::cout << "Thread is working..." << std::endl;
            goal_msg_.pose.header.frame_id = "map";
            goal_msg_.pose.header.stamp = this->get_clock()->now();
            goal_msg_.pose.pose.position.x = point2_vec_[0];
            goal_msg_.pose.pose.position.y = point2_vec_[1];
            goal_msg_.pose.pose.orientation.x = point2_vec_[2];
            goal_msg_.pose.pose.orientation.y = point2_vec_[3];
            goal_msg_.pose.pose.orientation.z = point2_vec_[4];
            goal_msg_.pose.pose.orientation.w = point2_vec_[5];
            this->client_ptr_->async_send_goal(goal_msg_, send_goal_options);
            {
                std::unique_lock<std::mutex> task_lock(task_mtx_);
                task_cv_.wait(lock, [&]{ return task_done_; });
                if(process_stop_ == true) break;
                task_done_ = false;
                goal_msg_.pose.header.frame_id = "map";
                goal_msg_.pose.header.stamp = this->get_clock()->now();
                goal_msg_.pose.pose.position.x = point3_vec_[0];
                goal_msg_.pose.pose.position.y = point3_vec_[1];
                goal_msg_.pose.pose.orientation.x = point3_vec_[2];
                goal_msg_.pose.pose.orientation.y = point3_vec_[3];
                goal_msg_.pose.pose.orientation.z = point3_vec_[4];
                goal_msg_.pose.pose.orientation.w = point3_vec_[5];
                this->client_ptr_->async_send_goal(goal_msg_, send_goal_options);
            }
            {
                std::unique_lock<std::mutex> task_lock(task_mtx_);
                task_cv_.wait(lock, [&]{ return task_done_; });
                if(process_stop_ == true) break;
                task_done_ = false;
                goal_msg_.pose.header.frame_id = "map";
                goal_msg_.pose.header.stamp = this->get_clock()->now();
                goal_msg_.pose.pose.position.x = point1_vec_[0];
                goal_msg_.pose.pose.position.y = point1_vec_[1];
                goal_msg_.pose.pose.orientation.x = point1_vec_[2];
                goal_msg_.pose.pose.orientation.y = point1_vec_[3];
                goal_msg_.pose.pose.orientation.z = point1_vec_[4];
                goal_msg_.pose.pose.orientation.w = point1_vec_[5];
                this->client_ptr_->async_send_goal(goal_msg_, send_goal_options);
            }
            {
                std::unique_lock<std::mutex> task_lock(task_mtx_);
                task_cv_.wait(lock, [&]{ return task_done_; });
                if(process_stop_ == true) break;
                task_done_ = false;
                std::this_thread::sleep_for(std::chrono::seconds(2));
                has_pub_ = false;
                mean_tf_v_.clear();
            }
        }

    }
}


int main(int argc, char* argv[]) {

    rclcpp::init(argc, argv);

    rclcpp::spin(std::make_shared<NavLocNode>("NavLocNode"));

    rclcpp::shutdown();

    RCLCPP_WARN(rclcpp::get_logger("NavLocNode"), "Pkg exit.");
    
    return 0;
}