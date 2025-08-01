# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np


from isaaclab.envs.utils import spaces
from isaaclab.sensors.camera import tiled_camera
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.terrains import TerrainImporterCfg
from gymnasium.spaces.discrete import Discrete
from isaaclab.sensors import TiledCameraCfg, CameraCfg, RayCasterCameraCfg, patterns
from gymnasium import spaces

# from semantic_manager import SemanticManager
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
    use_camera_obs: bool = True
    _width, _height = 200, 200
    # inspection_objective_prim_path = "/World/ground/terrain/Forklift"
    #inspection_objective_path = "/World/ground/terrain/SM_FuseBox_24"
    # inspection_objective_path = "/World/envs/env_.*/Cube"
    inspection_objective_prim_path = "/World/ground/terrain/_61_foam_brick"


    decimation = 2
    # semantic_config_path = "source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/semantic_config_warehouse.json"
    episode_length_s = 17
    action_scale = 1.0  # [N]
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    # action_space = spaces.Discrete(3)

    state_space = 0
    wheel_velocity_scale = 1.0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt= 1 / 120, 
                                       render_interval=decimation,
                                       physx=PhysxCfg(
                                        solver_type="tgs",
                                        min_position_iteration_count=8,
                                        max_position_iteration_count=8,
                                        min_velocity_iteration_count=1,
                                        max_velocity_iteration_count=1
                    ))
    # viewer = ViewerCfg(eye=(0, 0, 12.0), lookat=(0, 0, 0))

    # robot
    wheel_seperation = 0.4 # 0.37558
    wheel_radius = 0.098

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
        debug_vis=False
    
    )
    #SENSOR (Raycaster + Tiled Camera)
    if use_camera_obs:
        observation_camera = CameraCfg(
            prim_path="/World/envs/env_.*/Robot/base_link/front_camera",
            update_period=0.1,
            height=_height,
            width=_width,
            # data_types=["rgb", "semantic_segmentation", "instance_segmentation_fast"],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5)
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.3, 0.15),
                rot=(-0.5, 0.5, -0.5, 0.5),
                convention="ros"
            ),
            debug_vis=False  # Disable for performance
        )

        inspection_camera = CameraCfg(
            prim_path="/World/envs/env_.*/Robot/base_link/inspection_camera",
            update_period=0.1,
            height=_height,
            width=_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5)
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.3, 0.0, 0.15),
                rot = (0,  0, -0.7071068, 0.7071068),
                convention="ros"
            ),
            debug_vis=False  # Disable for performance
        )
    
    raycaster_camera_cfg = RayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        update_period=0.1,
        data_types=["face_ids"],
        offset=RayCasterCameraCfg.OffsetCfg(
            # pos=(0.3, 0.0, 0.15),
            # rot=(-0.5, 0.5, -0.5, 0.5),
            pos=(0.3, 0.0, 0.15),
            rot = (0,  0, -0.7071068, 0.7071068),
            convention="ros"
        ),
        pattern_cfg= patterns.PinholeCameraPatternCfg(
            height=_height,
            width=_width,
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        mesh_prim_paths = [inspection_objective_prim_path]
    ) 

    if use_camera_obs:
        observation_space = spaces.Dict({
            "robot-pose": spaces.Box(low=float("-inf"), high=float("inf"), shape=(13,)),
            "cameras": spaces.Box(low=float("-inf"), high=float("inf"), shape=(_height, _height, 6)),
        })

        # observation_space = spaces.Box(
        #         low=float("-inf"), high=float("inf"), shape=(_height, _width, 6)
        # )
    else:
        observation_space = 13
        # observation_space = spaces.Box(
        #         low=-np.inf,
        #         high=np.inf,
        #         shape=(7,),
        #         dtype=np.float32,
        #     )
    # observation_space = spaces.Dict({
    #     "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    #     "camera": spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
    # })  # or for simplicity: {"joint-velocities": 2, "camera": [height, width, 3]}

    terrain_cfg: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path="/home/tosin/Desktop/IsaacLab/environments/ware_house_brick.usd",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        debug_vis=False,
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    max_robot_distance = 2000

    #reward
    inspection_coverage_reward_scale = 0.005 # Scale for inspection coverage reward
    distance_reward_scale = 0.5  # Scale for distance-based rewards
    time_penalty = -0.1
    spin_penalty_scale = 0.05
    movement_reward_scale = 0.05

    save_inspection_images = True       # Whether to save images of good inspections
    inspection_threshold = 0.9   # Coverage % threshold to count as valid inspection
    coverage_reward = 10.0
    # inspection_save_dir = "inspection_captures"

    terminate_on_all_inspected = True
    min_episode_length = 1100
    max_faces_to_inspect = 6000
    include_robot_state = True

    max_linear_velocity = 1.0
    max_angular_velocity = 1.0
    min_discovery_interval = 1.0
    max_wheel_velocity = 10.0  # Max wheel velocity for the robot




#     cube_cfg = RigidObjectCfg( 
#         prim_path="/World/envs/env_.*/Cube",
#         spawn=sim_utils.CuboidCfg(
#             size=(0.5, 0.5, 1.0),
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(),
#             mass_props=sim_utils.MassPropertiesCfg(density=500.0, mass=100.0),
#             collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2.0, 0.00001))
# )

    # viewer = ViewerCfg( eye=(5.0, -21.0, 5.0), lookat=(-15, 10, 1.0))


# usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
# usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd",
# usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",