# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.terrains import TerrainImporterCfg
from gymnasium.spaces.discrete import Discrete
# from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors import TiledCameraCfg
from semantic_manager import SemanticManager

ROBOT_CONFIGS = {
    "jackal": {
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/Jackal/jackal_basic.usd",
        "wheel_joint_expr": ".*wheel.*",
        "action_space": 4  # 4 wheels
    },
    "jetbot": {
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd", 
        "wheel_joint_expr": ".*wheel.*",
        "action_space": 2  # 2 wheels
    },
}
@configclass
class Isaac3dinspectionEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 8
    semantic_config_path = "source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/semantic_config_warehouse.json"
    episode_length_s = 1000.0
    action_scale = 2.0  # [N]
    action_space = Discrete(3)

    state_space = 0
    wheel_velocity_scale = 2

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_CONFIGS["jackal"]["usd_path"],
        ),
        actuators={
            "wheel_acts": ImplicitActuatorCfg(
                joint_names_expr=ROBOT_CONFIGS["jackal"]["wheel_joint_expr"],
                damping=None,
                stiffness=None
            )
        },
        debug_vis=True
    
    )
    tiled_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/front_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "semantic_segmentation", "instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.15),
            rot=(-0.5, 0.5, -0.5, 0.5),
            convention="ros"
        ),
        semantic_filter=SemanticManager.get_semantic_filter_from_config(semantic_config_path),
        colorize_semantic_segmentation=True,  # Raw data for processing
        # colorize_instance_segmentation=True,
        debug_vis=True  # Disable for performance
    )
    observation_space = [tiled_camera.height, tiled_camera.width, 3]
    terrain_cfg: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        debug_vis=False,
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    max_robot_distance = 2000

    #reward
    forklift_reward_scale = 3.0  # Scale for forklift coverage reward
    distance_reward_scale = 0.1  # Scale for distance-based rewards

    save_inspection_images = True       # Whether to save images of good inspections
    inspection_threshold = 0.25         # Coverage % threshold to count as valid inspection
    inspection_save_dir = "inspection_captures"

    terminate_on_all_inspected = True
    min_episode_length = 1000
    image_observation_size = (64, 64)
    include_robot_state = True