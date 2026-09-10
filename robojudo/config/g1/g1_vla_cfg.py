from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import (
    CasiaHandCfg,
    Gr00tCameraCfg,
    Gr00tZmqCtrlCfg,
    UnitreeCtrlCfg,
    UpperBodyCasiaHandZmqCtrlCfg,
)
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
            net_if="eth1",
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
                Gr00tCameraCfg(
                    type="realsense",
                    name="left_wrist_rgb",
                    image_key="left_wrist_view",
                    options={
                        "serial_number": "130322272857",
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                    },
                ),
                Gr00tCameraCfg(
                    type="realsense",
                    name="right_wrist_rgb",
                    image_key="right_wrist_view",
                    options={
                        "serial_number": "130322273712",
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
                net_if="eth1",
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
