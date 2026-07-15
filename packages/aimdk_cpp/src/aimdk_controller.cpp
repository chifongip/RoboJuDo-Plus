#include "aimdk_controller.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace {
std::atomic<int> g_controller_count{0};
std::mutex g_rclcpp_mutex;

void init_rclcpp_once() {
  std::lock_guard<std::mutex> lock(g_rclcpp_mutex);
  if (std::getenv("ROS_LOG_DIR") == nullptr) {
    std::filesystem::create_directories("/tmp/robojudo_ros_logs");
    setenv("ROS_LOG_DIR", "/tmp/robojudo_ros_logs", 0);
  }
  if (!rclcpp::ok()) {
    int argc = 0;
    char** argv = nullptr;
    rclcpp::init(argc, argv);
  }
  ++g_controller_count;
}

void shutdown_rclcpp_if_last() {
  std::lock_guard<std::mutex> lock(g_rclcpp_mutex);
  if (--g_controller_count == 0 && rclcpp::ok()) {
    rclcpp::shutdown();
  }
}

void validate_gains(const std::vector<double>& stiffness,
                    const std::vector<double>& damping,
                    size_t expected_size) {
  if (stiffness.size() != expected_size || damping.size() != expected_size) {
    throw std::invalid_argument("stiffness and damping must match joint_names length");
  }
  for (double value : stiffness) {
    if (!std::isfinite(value) || value < 0.0) {
      throw std::invalid_argument("stiffness must contain only finite, non-negative values");
    }
  }
  for (double value : damping) {
    if (!std::isfinite(value) || value < 0.0) {
      throw std::invalid_argument("damping must contain only finite, non-negative values");
    }
  }
}
}  // namespace

AimdkController::AimdkController(const AimdkConfig& cfg)
    : cfg_(cfg),
      state_(cfg.joint_names.size()),
      stiffness_(cfg.stiffness),
      damping_(cfg.damping),
      latest_positions_(cfg.joint_names.size(), 0.0) {
  if (!std::isfinite(cfg_.control_dt) || !std::isfinite(cfg_.publish_dt) ||
      !std::isfinite(cfg_.command_timeout) || !std::isfinite(cfg_.state_timeout) ||
      !std::isfinite(cfg_.shutdown_publish_duration) || !std::isfinite(cfg_.shutdown_damping) ||
      cfg_.control_dt <= 0.0 || cfg_.publish_dt <= 0.0 || cfg_.command_timeout <= 0.0 ||
      cfg_.state_timeout <= 0.0 || cfg_.shutdown_publish_duration < 0.0 || cfg_.shutdown_damping < 0.0) {
    throw std::invalid_argument("AimDK timing values must be positive and damping values must be non-negative");
  }
  validate_gains(cfg_.stiffness, cfg_.damping, cfg_.joint_names.size());

  std::set<std::string> unique_joint_names(cfg_.joint_names.begin(), cfg_.joint_names.end());
  if (unique_joint_names.size() != cfg_.joint_names.size()) {
    throw std::invalid_argument("joint_names must not contain duplicates");
  }

  for (size_t i = 0; i < cfg_.joint_names.size(); ++i) {
    joint_index_[cfg_.joint_names[i]] = i;
    command_joint_names_.insert(cfg_.joint_names[i]);
  }

  std::set<std::string> grouped_joint_names;
  const std::vector<const std::vector<std::string>*> groups = {
      &cfg_.leg_joint_names, &cfg_.waist_joint_names, &cfg_.arm_joint_names, &cfg_.head_joint_names};
  for (const auto* group : groups) {
    for (const auto& name : *group) {
      if (joint_index_.find(name) == joint_index_.end()) {
        throw std::invalid_argument("joint group contains unknown name: " + name);
      }
      if (!grouped_joint_names.insert(name).second) {
        throw std::invalid_argument("joint appears in multiple groups: " + name);
      }
    }
  }
  if (grouped_joint_names != unique_joint_names) {
    throw std::invalid_argument("joint groups must contain every configured joint exactly once");
  }

  init_rclcpp_once();
  rclcpp_registered_ = true;
  try {
    node_ = rclcpp::Node::make_shared("robojudo_aimdk_cpp");
    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    auto qos = rclcpp::SensorDataQoS();

    leg_sub_ = node_->create_subscription<aimdk_msgs::msg::JointStateArray>(
        cfg_.leg_state_topic, qos,
        [this](const aimdk_msgs::msg::JointStateArray::SharedPtr msg) { joint_callback(msg); });
    waist_sub_ = node_->create_subscription<aimdk_msgs::msg::JointStateArray>(
        cfg_.waist_state_topic, qos,
        [this](const aimdk_msgs::msg::JointStateArray::SharedPtr msg) { joint_callback(msg); });
    arm_sub_ = node_->create_subscription<aimdk_msgs::msg::JointStateArray>(
        cfg_.arm_state_topic, qos,
        [this](const aimdk_msgs::msg::JointStateArray::SharedPtr msg) { joint_callback(msg); });
    head_sub_ = node_->create_subscription<aimdk_msgs::msg::JointStateArray>(
        cfg_.head_state_topic, qos,
        [this](const aimdk_msgs::msg::JointStateArray::SharedPtr msg) { joint_callback(msg); });
    imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
        cfg_.base_imu_topic, qos, [this](const sensor_msgs::msg::Imu::SharedPtr msg) { imu_callback(msg); });

    leg_pub_ = node_->create_publisher<aimdk_msgs::msg::JointCommandArray>(cfg_.leg_command_topic, qos);
    waist_pub_ = node_->create_publisher<aimdk_msgs::msg::JointCommandArray>(cfg_.waist_command_topic, qos);
    arm_pub_ = node_->create_publisher<aimdk_msgs::msg::JointCommandArray>(cfg_.arm_command_topic, qos);
    head_pub_ = node_->create_publisher<aimdk_msgs::msg::JointCommandArray>(cfg_.head_command_topic, qos);

    executor_->add_node(node_);
    running_ = true;
    publish_running_ = true;
    spin_thread_ = std::thread([this]() {
      while (running_ && rclcpp::ok()) {
        executor_->spin_some();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
    publish_thread_ = std::thread(&AimdkController::publish_loop, this);
  } catch (...) {
    publish_running_ = false;
    running_ = false;
    if (publish_thread_.joinable()) publish_thread_.join();
    if (spin_thread_.joinable()) spin_thread_.join();
    if (node_ && executor_) executor_->remove_node(node_);
    shutdown_rclcpp_if_last();
    rclcpp_registered_ = false;
    throw;
  }
}

AimdkController::~AimdkController() {
  shutdown();
  if (publish_thread_.joinable()) {
    publish_thread_.join();
  }
  if (spin_thread_.joinable()) {
    spin_thread_.join();
  }
  if (node_ && executor_) {
    executor_->remove_node(node_);
  }
  if (rclcpp_registered_) {
    shutdown_rclcpp_if_last();
    rclcpp_registered_ = false;
  }
}

bool AimdkController::self_check(double timeout_sec) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);
  while (std::chrono::steady_clock::now() < deadline) {
    if (state_is_fresh(cfg_.state_timeout)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (!cfg_.act) {
    return true;
  }
  return state_is_fresh(cfg_.state_timeout);
}

bool AimdkController::state_is_fresh(double timeout_sec) {
  if (!cfg_.act) {
    return true;
  }
  if (timeout_sec <= 0.0) {
    throw std::invalid_argument("state freshness timeout must be positive");
  }

  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!imu_received_ || received_joint_names_.size() != cfg_.joint_names.size()) {
    return false;
  }
  const auto cutoff = std::chrono::steady_clock::now() - std::chrono::duration<double>(timeout_sec);
  if (imu_update_time_ < cutoff) {
    return false;
  }
  for (const auto& name : cfg_.joint_names) {
    const auto it = joint_update_times_.find(name);
    if (it == joint_update_times_.end() || it->second < cutoff) {
      return false;
    }
  }
  return true;
}

RobotState AimdkController::get_robot_state() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return state_;
}

void AimdkController::step(const std::vector<double>& positions) {
  if (!cfg_.act) {
    return;
  }
  if (positions.size() != cfg_.joint_names.size()) {
    throw std::invalid_argument("positions must match joint_names length");
  }
  for (double position : positions) {
    if (!std::isfinite(position)) {
      throw std::invalid_argument("positions must contain only finite values");
    }
  }
  std::lock_guard<std::mutex> lock(command_mutex_);
  if (watchdog_tripped_) {
    throw std::runtime_error("AimDK position watchdog is latched; re-arm position control before sending targets");
  }
  latest_positions_ = positions;
  last_command_time_ = std::chrono::steady_clock::now();
  command_received_ = true;
  command_mode_ = AimdkCommandMode::POSITION;
}

void AimdkController::arm_position_control() {
  if (!cfg_.act) {
    return;
  }
  std::lock_guard<std::mutex> lock(command_mutex_);
  watchdog_tripped_ = false;
  command_received_ = false;
  command_mode_ = AimdkCommandMode::IDLE;
}

void AimdkController::set_passive() {
  if (!cfg_.act) {
    return;
  }
  std::lock_guard<std::mutex> lock(command_mutex_);
  command_mode_ = AimdkCommandMode::PASSIVE;
  command_received_ = true;
}

void AimdkController::set_damping(double damping) {
  if (damping < 0.0 || !std::isfinite(damping)) {
    throw std::invalid_argument("damping must be finite and non-negative");
  }
  if (!cfg_.act) {
    return;
  }
  std::lock_guard<std::mutex> lock(command_mutex_);
  mode_damping_ = damping;
  command_mode_ = AimdkCommandMode::DAMPING;
  command_received_ = true;
}

void AimdkController::set_gains(const std::vector<double>& stiffness, const std::vector<double>& damping) {
  validate_gains(stiffness, damping, cfg_.joint_names.size());
  std::lock_guard<std::mutex> lock(command_mutex_);
  stiffness_ = stiffness;
  damping_ = damping;
}

void AimdkController::set_control_joint_names(const std::vector<std::string>& joint_names) {
  std::lock_guard<std::mutex> lock(command_mutex_);
  command_joint_names_.clear();
  for (const auto& name : joint_names) {
    if (joint_index_.find(name) == joint_index_.end()) {
      throw std::invalid_argument("control joint name is not in joint_names: " + name);
    }
    command_joint_names_.insert(name);
  }
}

void AimdkController::shutdown() {
  if (shutdown_started_.exchange(true)) {
    return;
  }
  publish_running_ = false;
  if (publish_thread_.joinable() && publish_thread_.get_id() != std::this_thread::get_id()) {
    publish_thread_.join();
  }

  if (cfg_.act && rclcpp::ok()) {
    command_mode_ = AimdkCommandMode::DAMPING;
    mode_damping_ = cfg_.shutdown_damping;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(cfg_.shutdown_publish_duration);
    while (std::chrono::steady_clock::now() < deadline) {
      publish_damping_commands();
      std::this_thread::sleep_for(std::chrono::duration<double>(cfg_.publish_dt));
    }
  }
  running_ = false;
}

void AimdkController::joint_callback(const aimdk_msgs::msg::JointStateArray::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  const auto update_time = std::chrono::steady_clock::now();
  for (const auto& joint : msg->joints) {
    auto it = joint_index_.find(joint.name);
    if (it == joint_index_.end()) {
      continue;
    }
    const size_t idx = it->second;
    state_.motor_state.q[idx] = static_cast<float>(joint.position);
    state_.motor_state.dq[idx] = static_cast<float>(joint.velocity);
    state_.motor_state.tau_est[idx] = static_cast<float>(joint.effort);
    received_joint_names_.insert(joint.name);
    joint_update_times_[joint.name] = update_time;
  }
  ++state_.tick;
}

void AimdkController::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  imu_received_ = true;
  imu_update_time_ = std::chrono::steady_clock::now();
  state_.imu_state.quaternion = {
      static_cast<float>(msg->orientation.x),
      static_cast<float>(msg->orientation.y),
      static_cast<float>(msg->orientation.z),
      static_cast<float>(msg->orientation.w),
  };
  state_.imu_state.gyroscope = {
      static_cast<float>(msg->angular_velocity.x),
      static_cast<float>(msg->angular_velocity.y),
      static_cast<float>(msg->angular_velocity.z),
  };
  state_.imu_state.accelerometer = {
      static_cast<float>(msg->linear_acceleration.x),
      static_cast<float>(msg->linear_acceleration.y),
      static_cast<float>(msg->linear_acceleration.z),
  };
}

void AimdkController::publish_loop() {
  auto next_publish = std::chrono::steady_clock::now();
  while (publish_running_ && rclcpp::ok()) {
    next_publish += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(cfg_.publish_dt));

    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      if (command_received_) {
        const bool position_timed_out =
            command_mode_ == AimdkCommandMode::POSITION &&
            std::chrono::steady_clock::now() - last_command_time_ >
                std::chrono::duration<double>(cfg_.command_timeout);
        if (position_timed_out) {
          watchdog_tripped_ = true;
          command_mode_ = AimdkCommandMode::DAMPING;
          mode_damping_ = cfg_.shutdown_damping;
          publish_damping_commands();
        } else if (command_mode_ == AimdkCommandMode::PASSIVE) {
          publish_passive_commands();
        } else if (command_mode_ == AimdkCommandMode::DAMPING) {
          publish_damping_commands();
        } else if (command_mode_ == AimdkCommandMode::POSITION) {
          publish_group(cfg_.leg_joint_names, latest_positions_, stiffness_, damping_, true, leg_pub_);
          publish_group(cfg_.waist_joint_names, latest_positions_, stiffness_, damping_, true, waist_pub_);
          publish_group(cfg_.arm_joint_names, latest_positions_, stiffness_, damping_, true, arm_pub_);
          publish_group(cfg_.head_joint_names, latest_positions_, stiffness_, damping_, true, head_pub_);
        }
      }
    }
    std::this_thread::sleep_until(next_publish);
  }
}

void AimdkController::publish_passive_commands() {
  const std::vector<double> zeros(cfg_.joint_names.size(), 0.0);
  publish_group(cfg_.leg_joint_names, zeros, zeros, zeros, false, leg_pub_);
  publish_group(cfg_.waist_joint_names, zeros, zeros, zeros, false, waist_pub_);
  publish_group(cfg_.arm_joint_names, zeros, zeros, zeros, false, arm_pub_);
  publish_group(cfg_.head_joint_names, zeros, zeros, zeros, false, head_pub_);
}

void AimdkController::publish_damping_commands() {
  const std::vector<double> positions(cfg_.joint_names.size(), 0.0);
  const std::vector<double> stiffness(cfg_.joint_names.size(), 0.0);
  const double damping_value = command_mode_ == AimdkCommandMode::DAMPING ? mode_damping_ : cfg_.shutdown_damping;
  const std::vector<double> damping(cfg_.joint_names.size(), damping_value);
  publish_group(cfg_.leg_joint_names, positions, stiffness, damping, false, leg_pub_);
  publish_group(cfg_.waist_joint_names, positions, stiffness, damping, false, waist_pub_);
  publish_group(cfg_.arm_joint_names, positions, stiffness, damping, false, arm_pub_);
  publish_group(cfg_.head_joint_names, positions, stiffness, damping, false, head_pub_);
}

void AimdkController::publish_group(
    const std::vector<std::string>& names,
    const std::vector<double>& positions,
    const std::vector<double>& stiffness,
    const std::vector<double>& damping,
    bool controlled_only,
    const rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr& pub) {
  if (names.empty() || !pub) {
    return;
  }

  aimdk_msgs::msg::JointCommandArray msg;
  msg.joints.reserve(names.size());
  for (const auto& name : names) {
    auto it = joint_index_.find(name);
    if (it == joint_index_.end()) {
      continue;
    }
    if (controlled_only && command_joint_names_.find(name) == command_joint_names_.end()) {
      continue;
    }
    const size_t idx = it->second;
    aimdk_msgs::msg::JointCommand cmd;
    cmd.name = name;
    cmd.position = positions[idx];
    cmd.velocity = 0.0;
    cmd.effort = 0.0;
    cmd.stiffness = stiffness[idx];
    cmd.damping = damping[idx];
    msg.joints.push_back(cmd);
  }
  if (!msg.joints.empty()) {
    pub->publish(msg);
  }
}
