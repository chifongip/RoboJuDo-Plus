# XRoboToolkit X2 sample setup

The optional `third_party/xrobotoolkit_teleop` submodule provides an upstream fixed-base X2 simulation.
XRoboToolkit control of RoboJuDo pipelines is not yet implemented.

## Environment and installation

The RoboJuDo installer supports Python 3.11+ on Linux x86_64/aarch64. A dedicated environment is recommended for
dependency isolation, but an existing compatible environment also works. With Conda installed, run these commands
from the RoboJuDo-Plus repository root:

```bash
conda create -n robojudo-xr python=3.11 pip -y
conda activate robojudo-xr

# The optional-module installer reads submodule_cfg.yaml with PyYAML.
python -m pip install pyyaml
python submodule_install.py xrobotoolkit_teleop
```

To reuse an environment, activate it and run the final two commands, skipping environment creation. The installer
preserves the submodule worktree and installs its `x2` extra in editable mode using the invoking Python interpreter.
It neither creates nor switches environments and does not run upstream setup scripts or install the full RoboJuDo
runtime. When returning in a new terminal, activate `robojudo-xr` before using the sample.

The `x2` extra installs NumPy, Meshcat, Placo, XRoboToolkit SDK, MuJoCo, and Tyro, plus their transitive dependencies.
It supports the arm-only and simulated OmniHand Pro modes. It excludes other robot SDKs, Torch, camera packages,
and hand-retargeting packages. The upstream `[all]` extra remains available for other robot examples.

## Verify and launch

Installation checks dependency and controller imports without initializing XR or opening a viewer. You can repeat
the checks without a headset or PC Service:

```bash
python -m pip check
python -c "from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController; print('OK')"
python third_party/xrobotoolkit_teleop/scripts/simulation/teleop_agibot_x2_mujoco.py --help
```

Install and start [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) separately,
connect the headset/controllers, and use a graphical desktop to launch the sample:

```bash
python third_party/xrobotoolkit_teleop/scripts/simulation/teleop_agibot_x2_mujoco.py
```

Use `--with-omnihand` for simulated hands. Neither this option nor installation connects to physical robot hardware.
The [upstream README](../third_party/xrobotoolkit_teleop/README.md) documents controller bindings and collision checks.

## Troubleshooting

- If Python 3.10 is selected, activate the Python 3.11+ environment and check `python --version` before rerunning.
- If the installer reports a missing `x2` extra, run `git submodule update --init third_party/xrobotoolkit_teleop`
  to select the pinned revision, then retry installation.
- If `pip check` reports unrelated ROS packages in an isolated environment, use a fresh terminal without a sourced
  ROS workspace; a ROS `PYTHONPATH` can expose packages outside the active environment.
- Missing native build or graphics libraries must be installed separately if pip or the import check reports them.
  Upstream setup is tested on Ubuntu 22.04; live simulation requires a working graphical desktop.
