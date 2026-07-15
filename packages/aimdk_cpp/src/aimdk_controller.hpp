#pragma once

#include <aimdk_msgs/msg/joint_command_array.hpp>
#include <aimdk_msgs/msg/joint_state_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

struct AimdkConfig {
  bool act{true};
  double control_dt{0.02};
  double publish_dt{0.002};
  double command_timeout{0.1};
  double shutdown_damping{5.0};
  double shutdown_publish_duration{0.2};
  double state_timeout{0.1};
  std::string base_imu_topic{"/aima/hal/imu/torso/state"};

  std::string leg_state_topic{"/aima/hal/joint/leg/state"};
  std::string waist_state_topic{"/aima/hal/joint/waist/state"};
  std::string arm_state_topic{"/aima/hal/joint/arm/state"};
  std::string head_state_topic{"/aima/hal/joint/head/state"};

  std::string leg_command_topic{"/aima/hal/joint/leg/command"};
  std::string waist_command_topic{"/aima/hal/joint/waist/command"};
  std::string arm_command_topic{"/aima/hal/joint/arm/command"};
  std::string head_command_topic{"/aima/hal/joint/head/command"};

  std::vector<std::string> joint_names;
  std::vector<std::string> leg_joint_names;
  std::vector<std::string> waist_joint_names;
  std::vector<std::string> arm_joint_names;
  std::vector<std::string> head_joint_names;

  std::vector<double> stiffness;
  std::vector<double> damping;
};

struct MotorState {
  std::vector<float> q;
  std::vector<float> dq;
  std::vector<float> tau_est;

  explicit MotorState(size_t num_motors = 0)
      : q(num_motors, 0.0F), dq(num_motors, 0.0F), tau_est(num_motors, 0.0F) {}
};

struct ImuState {
  std::vector<float> quaternion{0.0F, 0.0F, 0.0F, 1.0F};
  std::vector<float> gyroscope{0.0F, 0.0F, 0.0F};
  std::vector<float> accelerometer{0.0F, 0.0F, 0.0F};
};

struct RobotState {
  uint64_t tick{0};
  MotorState motor_state;
  ImuState imu_state;

  explicit RobotState(size_t num_motors = 0) : motor_state(num_motors) {}
};

enum class AimdkCommandMode { IDLE, POSITION, PASSIVE, DAMPING };

class AimdkController {
 public:
  explicit AimdkController(const AimdkConfig& cfg);
  ~AimdkController();

  bool self_check(double timeout_sec = 2.0);
  bool state_is_fresh(double timeout_sec);
  RobotState get_robot_state();
  void step(const std::vector<double>& positions);
  void arm_position_control();
  void set_passive();
  void set_damping(double damping);
  void set_gains(const std::vector<double>& stiffness, const std::vector<double>& damping);
  void set_control_joint_names(const std::vector<std::string>& joint_names);
  void shutdown();

 private:
  AimdkConfig cfg_;
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spin_thread_;
  std::thread publish_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> publish_running_{false};
  std::atomic<bool> shutdown_started_{false};
  bool rclcpp_registered_{false};

  rclcpp::Subscription<aimdk_msgs::msg::JointStateArray>::SharedPtr leg_sub_;
  rclcpp::Subscription<aimdk_msgs::msg::JointStateArray>::SharedPtr waist_sub_;
  rclcpp::Subscription<aimdk_msgs::msg::JointStateArray>::SharedPtr arm_sub_;
  rclcpp::Subscription<aimdk_msgs::msg::JointStateArray>::SharedPtr head_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

  rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr leg_pub_;
  rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr waist_pub_;
  rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr arm_pub_;
  rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr head_pub_;

  std::mutex state_mutex_;
  std::mutex command_mutex_;
  std::map<std::string, size_t> joint_index_;
  std::set<std::string> command_joint_names_;
  std::set<std::string> received_joint_names_;
  std::map<std::string, std::chrono::steady_clock::time_point> joint_update_times_;
  bool imu_received_{false};
  std::chrono::steady_clock::time_point imu_update_time_{};
  RobotState state_;
  std::vector<double> stiffness_;
  std::vector<double> damping_;
  std::vector<double> latest_positions_;
  std::chrono::steady_clock::time_point last_command_time_{};
  bool command_received_{false};
  AimdkCommandMode command_mode_{AimdkCommandMode::IDLE};
  double mode_damping_{5.0};
  bool watchdog_tripped_{false};

  void joint_callback(const aimdk_msgs::msg::JointStateArray::SharedPtr msg);
  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
  void publish_loop();
  void publish_passive_commands();
  void publish_damping_commands();
  void publish_group(const std::vector<std::string>& names,
                     const std::vector<double>& positions,
                     const std::vector<double>& stiffness,
                     const std::vector<double>& damping,
                     bool controlled_only,
                     const rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr& pub);
};
