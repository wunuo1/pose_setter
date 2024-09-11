#ifndef NAV_LOC_NODE_H_
#define NAV_LOC_NODE_H_

#include <vector>
#include <deque>
#include <unordered_set>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "tf2/convert.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/message_filter.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/create_timer_ros.h"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include <image_transport/image_transport.hpp>
#include "sensor_msgs/msg/image.hpp"
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <apriltag.h>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef std::function<geometry_msgs::msg::Transform(apriltag_detection_t* const, const std::array<double, 4>&, const double&)> pose_estimation_f;


class NavLocNode : public rclcpp::Node{
public:
    using NavigateToPose = nav2_msgs::action::NavigateToPose;
    using GoalHandleNavigateToPose = rclcpp_action::ClientGoalHandle<NavigateToPose>;

    NavLocNode(const std::string& node_name, const rclcpp::NodeOptions &option = rclcpp::NodeOptions());
    ~NavLocNode() override;

private:

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr init_pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
    rclcpp_action::Client<NavigateToPose>::SharedPtr client_ptr_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::CallbackGroup::SharedPtr callback_group_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    bool build_map_ = false;
    int max_hamming = 0;
    std::mutex mutex_;
    std::unordered_map<int, double> tag_sizes;
    std::unordered_map<int, std::string> tag_frames_;
    apriltag_detector_t* td_;
    pose_estimation_f estimate_pose = nullptr;
    double tag_edge_size = 1.0;
    int mean_num_ = 10;
    bool has_pub_ = false;
    std::shared_ptr<std::thread> client_;
    std::string tag_family = "36h11";
    apriltag_family_t* tf_;
    std::mutex mtx_;
    std::mutex task_mtx_;
    std::condition_variable task_cv_;
    std::condition_variable cv_;
    std::function<void(apriltag_family_t*)> tf_destructor;
    void sub_cam_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void mean_transform(const std::vector<tf2::Transform> &tf_tmp);
    void init_param();
    void goal_response_callback(std::shared_ptr<GoalHandleNavigateToPose> future);
    void result_callback(const GoalHandleNavigateToPose::WrappedResult & result);
    void feedback_callback(GoalHandleNavigateToPose::SharedPtr,
        const std::shared_ptr<const NavigateToPose::Feedback> feedback);
    void client_control();
    void publish_point_cloud();
    rclcpp_action::Client<NavigateToPose>::SendGoalOptions send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    std::vector<tf2::Transform> mean_tf_v_;
    tf2::Transform imu_tof_tf_;
    tf2::Transform imu_rgb_tf_;
    tf2::Transform tof_rgb_tf_;
    tf2::Transform rob_imu_tf_;
    tf2::Transform rob_rgb_tf_;

    tf2::Transform map_tag_tf_;
    tf2::Transform map_rob_tf_;
    bool process_stop_ = false;
    bool task_done_ = false;
    bool pub_inital_post_ = false;
    std::vector<float> point1_vec_;
    std::vector<float> point2_vec_;
    std::vector<float> point3_vec_;
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud_;
    nav2_msgs::action::NavigateToPose::Goal goal_msg_;
};




#endif //NAV_LOC_NODE_H_