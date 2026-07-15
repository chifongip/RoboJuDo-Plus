#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "aimdk_controller.hpp"

namespace py = pybind11;

AimdkConfig config_from_dict(py::dict cfg_dict) {
  AimdkConfig cfg;
  if (cfg_dict.contains("act")) cfg.act = cfg_dict["act"].cast<bool>();
  if (cfg_dict.contains("control_dt")) cfg.control_dt = cfg_dict["control_dt"].cast<double>();
  if (cfg_dict.contains("publish_dt")) cfg.publish_dt = cfg_dict["publish_dt"].cast<double>();
  if (cfg_dict.contains("command_timeout")) cfg.command_timeout = cfg_dict["command_timeout"].cast<double>();
  if (cfg_dict.contains("shutdown_damping")) cfg.shutdown_damping = cfg_dict["shutdown_damping"].cast<double>();
  if (cfg_dict.contains("shutdown_publish_duration")) {
    cfg.shutdown_publish_duration = cfg_dict["shutdown_publish_duration"].cast<double>();
  }
  if (cfg_dict.contains("state_timeout")) cfg.state_timeout = cfg_dict["state_timeout"].cast<double>();
  if (cfg_dict.contains("base_imu_topic")) cfg.base_imu_topic = cfg_dict["base_imu_topic"].cast<std::string>();

  if (cfg_dict.contains("leg_state_topic")) cfg.leg_state_topic = cfg_dict["leg_state_topic"].cast<std::string>();
  if (cfg_dict.contains("waist_state_topic")) cfg.waist_state_topic = cfg_dict["waist_state_topic"].cast<std::string>();
  if (cfg_dict.contains("arm_state_topic")) cfg.arm_state_topic = cfg_dict["arm_state_topic"].cast<std::string>();
  if (cfg_dict.contains("head_state_topic")) cfg.head_state_topic = cfg_dict["head_state_topic"].cast<std::string>();

  if (cfg_dict.contains("leg_command_topic")) cfg.leg_command_topic = cfg_dict["leg_command_topic"].cast<std::string>();
  if (cfg_dict.contains("waist_command_topic")) {
    cfg.waist_command_topic = cfg_dict["waist_command_topic"].cast<std::string>();
  }
  if (cfg_dict.contains("arm_command_topic")) cfg.arm_command_topic = cfg_dict["arm_command_topic"].cast<std::string>();
  if (cfg_dict.contains("head_command_topic")) {
    cfg.head_command_topic = cfg_dict["head_command_topic"].cast<std::string>();
  }

  cfg.joint_names = cfg_dict["joint_names"].cast<std::vector<std::string>>();
  cfg.leg_joint_names = cfg_dict["leg_joint_names"].cast<std::vector<std::string>>();
  cfg.waist_joint_names = cfg_dict["waist_joint_names"].cast<std::vector<std::string>>();
  cfg.arm_joint_names = cfg_dict["arm_joint_names"].cast<std::vector<std::string>>();
  cfg.head_joint_names = cfg_dict["head_joint_names"].cast<std::vector<std::string>>();
  cfg.stiffness = cfg_dict["stiffness"].cast<std::vector<double>>();
  cfg.damping = cfg_dict["damping"].cast<std::vector<double>>();
  return cfg;
}

PYBIND11_MODULE(aimdk_cpp, m) {
  m.doc() = "pybind11 backend for AgiBot AimDK ROS 2 communication";

  py::class_<AimdkConfig>(m, "AimdkConfig")
      .def(py::init<>())
      .def_readwrite("act", &AimdkConfig::act)
      .def_readwrite("control_dt", &AimdkConfig::control_dt)
      .def_readwrite("publish_dt", &AimdkConfig::publish_dt)
      .def_readwrite("command_timeout", &AimdkConfig::command_timeout)
      .def_readwrite("shutdown_damping", &AimdkConfig::shutdown_damping)
      .def_readwrite("shutdown_publish_duration", &AimdkConfig::shutdown_publish_duration)
      .def_readwrite("state_timeout", &AimdkConfig::state_timeout)
      .def_readwrite("base_imu_topic", &AimdkConfig::base_imu_topic)
      .def_readwrite("joint_names", &AimdkConfig::joint_names)
      .def_readwrite("leg_joint_names", &AimdkConfig::leg_joint_names)
      .def_readwrite("waist_joint_names", &AimdkConfig::waist_joint_names)
      .def_readwrite("arm_joint_names", &AimdkConfig::arm_joint_names)
      .def_readwrite("head_joint_names", &AimdkConfig::head_joint_names)
      .def_readwrite("stiffness", &AimdkConfig::stiffness)
      .def_readwrite("damping", &AimdkConfig::damping);

  py::class_<MotorState>(m, "MotorState", py::module_local())
      .def(py::init<size_t>(), py::arg("num_motors") = 0)
      .def_readwrite("q", &MotorState::q)
      .def_readwrite("dq", &MotorState::dq)
      .def_readwrite("tau_est", &MotorState::tau_est);

  py::class_<ImuState>(m, "ImuState", py::module_local())
      .def(py::init<>())
      .def_readwrite("quaternion", &ImuState::quaternion)
      .def_readwrite("gyroscope", &ImuState::gyroscope)
      .def_readwrite("accelerometer", &ImuState::accelerometer);

  py::class_<RobotState>(m, "RobotState", py::module_local())
      .def(py::init<size_t>(), py::arg("num_motors") = 0)
      .def_readwrite("tick", &RobotState::tick)
      .def_readwrite("motor_state", &RobotState::motor_state)
      .def_readwrite("imu_state", &RobotState::imu_state);

  py::class_<AimdkController>(m, "AimdkController")
      .def(py::init([](py::dict cfg_dict) { return new AimdkController(config_from_dict(cfg_dict)); }))
      .def(py::init<const AimdkConfig&>())
      .def("self_check", &AimdkController::self_check, py::arg("timeout_sec") = 2.0)
      .def("state_is_fresh", &AimdkController::state_is_fresh, py::arg("timeout_sec"))
      .def("get_robot_state", &AimdkController::get_robot_state)
      .def("step", &AimdkController::step, py::arg("positions"))
      .def("arm_position_control", &AimdkController::arm_position_control)
      .def("set_passive", &AimdkController::set_passive)
      .def("set_damping", &AimdkController::set_damping, py::arg("damping"))
      .def("set_gains", &AimdkController::set_gains, py::arg("stiffness"), py::arg("damping"))
      .def("set_control_joint_names", &AimdkController::set_control_joint_names, py::arg("joint_names"))
      .def("shutdown", &AimdkController::shutdown);
}
