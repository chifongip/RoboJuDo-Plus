from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import (
    CasiaHandCfg,
    Gr00tCameraCfg,
    Gr00tZmqCtrlCfg,
    UnitreeCtrlCfg,
    UpperBodyCasiaHandZmqCtrlCfg,
)
from robojudo.recording import RecordCfg

from .env.g1_real_env_cfg import G1_23RealEnvCfg, G1UnitreeCfg
from .g1_cfg import (
    G1_23_UPPER_BODY_DEFAULT_POSE,
    _g1_23_joint_default_dof_with_upper_pose,
    g1_23_locomanipulation_stiff_real,
)
from .policy.g1_gr00t_locomanipulation_policy_cfg import G1Gr00tLocomanipulation23PolicyCfg
from .policy.g1_locomanipulation_policy_cfg import G1Locomanipulation23ObsDoF


# ======================== Configs for G1 23dof with CASIA-hand Teleop ======================== #
def _g1_casia_locomanipulation_real_ctrl(
    joint_names: list[str],
) -> list[UnitreeCtrlCfg | UpperBodyCasiaHandZmqCtrlCfg]:
    return [
        UnitreeCtrlCfg(
            combination_init_buttons=["L1", "R1"],
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "Start": "[UPPER_BODY_TOGGLE]",
                "L1+R1+Start": "[RECORD_START_STOP]",
                "L1+R1+Select": "[RECORD_PAUSE_RESUME]",
                "L1+R1+X": "[RECORD_CONFIRM_SAVE]",
                "L1+R1+B": "[RECORD_DISCARD]",
                "L1+R1+A": "[SHUTDOWN]",
            },
        ),
        UpperBodyCasiaHandZmqCtrlCfg(
            joint_names=joint_names,
            upper_body_default_pose=G1_23_UPPER_BODY_DEFAULT_POSE,
            endpoint="tcp://192.168.252.72:8560",
            casia_hand=CasiaHandCfg(),
        )
    ]


@cfg_registry.register
class g1_23_casia_locomanipulation_stiff_real(g1_23_locomanipulation_stiff_real):
    """G1 23-DOF stiff-gain policy with direct dual CASIA Hand control."""
    env: G1_23RealEnvCfg = G1_23RealEnvCfg(
        dof=G1Locomanipulation23ObsDoF.from_preset("stiff"),
        unitree=G1UnitreeCfg(
            net_if="eth0",
            command_timeout=0.1,
            state_timeout=0.1,
            shutdown_damping=5.0,
        ),
    )

    pipeline_type: str = "G1CasiaHandLocomanipulationPipeline"
    ctrl: list[UnitreeCtrlCfg | UpperBodyCasiaHandZmqCtrlCfg] = _g1_casia_locomanipulation_real_ctrl(
        G1Locomanipulation23ObsDoF().joint_names[13:]
    )


# ======================== Configs for G1 23dof with CASIA-hand GR00T policy ======================== #
def _g1_gr00t_locomanipulation_real_ctrl(
    joint_names: list[str],
    *,
    offline_dagger: bool = False,
) -> list[UnitreeCtrlCfg | Gr00tZmqCtrlCfg]:
    camera_specs = (
        ("head_rgb", "ego_view", "242222070519", 8571),
        ("left_wrist_rgb", "left_wrist_view", "130322272857", 8572),
        ("right_wrist_rgb", "right_wrist_view", "130322273712", 8573),
    )
    return [
        UnitreeCtrlCfg(
            combination_init_buttons=["L1", "R1"],
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "Start": "[UPPER_BODY_TOGGLE]",
                # Offline DAgger recording controls retain the established
                # Unitree shoulder chords; bare held Select remains intervention.
                **(
                    {
                        "L1+R1+Start": "[RECORD_START_STOP]",
                        "L1+R1+Select": "[RECORD_PAUSE_RESUME]",
                        "L1+R1+X": "[RECORD_CONFIRM_SAVE]",
                        "L1+R1+B": "[RECORD_DISCARD]",
                    }
                    if offline_dagger
                    else {}
                ),
                "L1+R1+A": "[SHUTDOWN]",
            },
        ),
        Gr00tZmqCtrlCfg(
            joint_names=joint_names,
            # GR00T deployment endpoint
            endpoint=(
                "tcp://192.168.123.222:8559"
            ),
            # RL_DEFAULT fallback only; B/DAMPING_DEFAULT remains pure damping.
            upper_body_default_pose=G1_23_UPPER_BODY_DEFAULT_POSE,
            casia_hand=CasiaHandCfg(),
            ema_alpha=0.0,
            observation_enabled=True,
            observation_profile="g1_23dof",
            # Offline DAgger is opt-in: dex-teleop expert frames arrive on
            # 8560 while measured robot/camera feedback is published on 8561.
            offline_dagger_enabled=offline_dagger,
            # Offline DAgger: dex-teleop publishes expert arm/hand targets from
            # the same remote workstation as GR00T deploy.
            expert_endpoint=(
                "tcp://192.168.123.222:8560"
            ),
            intervention_button="Select",
            cameras=[
                Gr00tCameraCfg(
                    type="zmq" if offline_dagger else "realsense",
                    name=name,
                    image_key=image_key,
                    options=(
                        # Both processes share the same host monotonic clock.
                        {"endpoint": f"tcp://127.0.0.1:{port}", "encoding": "jpeg", "timestamp_mode": "source"}
                        if offline_dagger
                        else {"serial_number": serial, "width": 640, "height": 480, "fps": 30}
                    ),
                )
                for name, image_key, serial, port in camera_specs
            ],
        ),
    ]

def _g1_gr00t_locomanipulation_single_cam_real_ctrl(
    joint_names: list[str],
) -> list[UnitreeCtrlCfg | Gr00tZmqCtrlCfg]:
    return [
        UnitreeCtrlCfg(
            combination_init_buttons=["L1", "R1"],
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "Start": "[UPPER_BODY_TOGGLE]",
                "L1+R1+A": "[SHUTDOWN]",
            },
        ),
        Gr00tZmqCtrlCfg(
            joint_names=joint_names,
            # RL_DEFAULT fallback only; B/DAMPING_DEFAULT remains pure damping.
            upper_body_default_pose=G1_23_UPPER_BODY_DEFAULT_POSE,
            casia_hand=CasiaHandCfg(),
            ema_alpha=0.0,
            observation_enabled=True,
            observation_profile="g1_23dof",
            cameras=[
                Gr00tCameraCfg(
                    type="realsense",
                    name="head_rgb",
                    image_key="ego_view",
                    options={
                        "serial_number": "242222070519",
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                    },
                ),
            ],
        ),
    ]

@cfg_registry.register
class g1_23_gr00t_locomanipulation_stiff_real(g1_23_locomanipulation_stiff_real):
    """G1 23-DoF stiff-gain GR00T Locomanipulation, Sim2Real."""
    env: G1_23RealEnvCfg = G1_23RealEnvCfg(
            dof=G1Locomanipulation23ObsDoF.from_preset("stiff"),
            unitree=G1UnitreeCfg(
                net_if="eth0",
                command_timeout=0.1,
                state_timeout=0.1,
                shutdown_damping=5.0,
            ),
        )
    pipeline_type: str = "G1Gr00tLocomanipulationPipeline"
    ctrl: list[UnitreeCtrlCfg | Gr00tZmqCtrlCfg] = _g1_gr00t_locomanipulation_real_ctrl(
        G1Locomanipulation23ObsDoF().joint_names[13:]
    )
    policy: G1Gr00tLocomanipulation23PolicyCfg = G1Gr00tLocomanipulation23PolicyCfg(
        policy_name="policy_23dof_stiff",
        pd_gain_preset="stiff",
    )
    joint_default_dof: G1Locomanipulation23ObsDoF = _g1_23_joint_default_dof_with_upper_pose(
        "stiff",
        G1_23_UPPER_BODY_DEFAULT_POSE,
    )


@cfg_registry.register
class g1_23_gr00t_offline_dagger_stiff_real(g1_23_gr00t_locomanipulation_stiff_real):
    """G1 GR00T rollout with Select-held full-action intervention for offline DAgger."""

    # Offline DAgger uses a dedicated config so the existing GR00T deployment
    # keeps its original policy-only transport and joystick behavior.
    ctrl: list[UnitreeCtrlCfg | Gr00tZmqCtrlCfg] = _g1_gr00t_locomanipulation_real_ctrl(
        G1Locomanipulation23ObsDoF().joint_names[13:],
        offline_dagger=True,
    )
    # Offline DAgger recorder traffic uses 8562 because dex-teleop already owns
    # the expert-action PUB on 8560. The recorder connects to this bind address.
    record: RecordCfg = RecordCfg(
        enabled=True,
        endpoint="tcp://*:8562",
        task="offline DAgger arm hand locomotion intervention",
    )
