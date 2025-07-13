# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import torch
from collections.abc import Sequence
import numpy as np

from datetime import datetime
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .inspection_cfg import Isaac3dinspectionEnvCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.sensors import TiledCamera, RayCasterCamera
import isaacsim.core.utils.stage as stage_utils
from .semantic_manager import SemanticManager, add_semantic_tags_from_config
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics
#View logs
# execute from the root directory of the repository
# ./isaaclab.sh -p -m tensorboard.main --logdir logs/skrl/3DInspection_direct
#Train
# ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Inspection-Camera-Direct-v0 --num_envs 1 --headless --video
#PLAY
## execute from the root directory of the repository
# ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Inspection-Camera-Direct-v0 --num_envs 1 --use_last_checkpoint
debug = False

class Isaac3dinspectionEnv(DirectRLEnv):
    cfg: Isaac3dinspectionEnvCfg

    def __init__(self, cfg: Isaac3dinspectionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._wheel_joint_indices, self._wheel_joint_names = self.robot.find_joints(".*wheel.*")
        self.wheel_velocity_scale = self.cfg.wheel_velocity_scale

        self.robot_pos = self.robot.data.root_pos_w
        self.robot_vel = self.robot.data.root_lin_vel_w

        self.forward_velocities = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        self.turn_left_velocities = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=self.device)
        self.turn_right_velocities = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device)

        # self.action_scale = self.cfg.action_scale
        self.semantic_manager = SemanticManager(self.cfg.semantic_config_path, initialize_stage=False)
        self.target_prim_map = {
            obj["prim_path"]: obj["object_name"]
            for obj in self.semantic_manager.config.get("semantic_objects", [])
        }
        self.target_object_names = list(self.target_prim_map.values())

        self.objective_position = torch.tensor([3.18, 10.8, 0.0], device=self.device) 

        os.makedirs(self.cfg.inspection_save_dir, exist_ok=True)

        if debug:
            print(f"✅ Tracking {len(self.target_prim_map)} target objects: {self.target_object_names}")
        self._setup_semantics()
        self._setup_tensor_buffers()

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _setup_tensor_buffers(self):
        """Pre-allocate all tensors to avoid memory allocation during runtime."""
        height = self.cfg.tiled_camera.height
        width = self.cfg.tiled_camera.width
        num_envs = self.num_envs
        
        # Camera data buffers (reused every frame)
        self.rgb_buffer = torch.zeros((num_envs, height, width, 3), 
                                     dtype=torch.float32, device=self.device)
        # Buffers for mean calculation (avoid creating new tensors)
        self.mean_buffer = torch.zeros((num_envs, 1, 1, 3), 
                                      dtype=torch.float32, device=self.device)
        
        # Observation buffer (final output)
        self.obs_buffer = torch.zeros((num_envs, height, width, 3), 
                                     dtype=torch.float32, device=self.device)
        
        # If you add semantic detection back later:
        self.semantic_buffer = torch.zeros((num_envs, height, width, 4), 
                                          dtype=torch.uint8, device=self.device)
        self.max_face_ids = 10000 
        self.discovered_faces_buffer = torch.zeros(self.num_envs, self.max_face_ids,
                                                    dtype=torch.bool, device=self.device)
        self.total_faces_discovered = torch.zeros(self.num_envs, 
                                                  dtype=torch.int32, device=self.device)
        self.face_ids_buffer = torch.zeros((num_envs, height, width, 1), 
                                          dtype=torch.int32, device=self.device)
        self.face_flat_buffer = torch.zeros((num_envs, height * width), 
                                       dtype=torch.int32, device=self.device)
        self.valid_mask_buffer = torch.zeros((num_envs, height * width), 
                                        dtype=torch.bool, device=self.device)
        self.newly_discovered_count = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self.face_rewards = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

    def _setup_semantics(self):
        """Setup semantic tags using the semantic manager."""
        try:
            print("Setting up semantic tags from configuration...")
            success = add_semantic_tags_from_config(self.cfg.semantic_config_path)
            if success:
                if debug:
                    print("✅ Successfully applied semantic tags from configuration")
                # Initialize semantic manager for runtime use
                self.semantic_manager = SemanticManager(self.cfg.semantic_config_path)
            else:
                if debug:
                    print("❌ Failed to apply some or all semantic tags")

        except Exception as e:
            if debug:
                print(f"❌ ERROR: Failed to setup semantics: {e}")
                print("Continuing without semantic tags...")
            pass

    def _setup_scene(self):
        #Add robot, camera and terain to the scene
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["camera"] = self._tiled_camera

        self._raycaster_camera = RayCasterCamera(self.cfg.raycaster_camera_cfg)
        self.scene.sensors["raycaster_camera"] = self._raycaster_camera

        self.cfg.terrain_cfg.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain_cfg.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain_cfg)
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        wheel_commands = torch.zeros(self.num_envs, 4, device=self.device)
        wheel_commands = torch.where(
            (actions == 0).unsqueeze(1), 
            self.forward_velocities.unsqueeze(0).expand(self.num_envs, -1),
            wheel_commands
        )
        wheel_commands = torch.where(
            (actions == 1).unsqueeze(1), 
            self.turn_left_velocities.unsqueeze(0).expand(self.num_envs, -1),
            wheel_commands
        )
        wheel_commands = torch.where(
            (actions == 2).unsqueeze(1), 
            self.turn_right_velocities.unsqueeze(0).expand(self.num_envs, -1),
            wheel_commands
        )

        self.actions = wheel_commands.clone() * self.wheel_velocity_scale

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self._wheel_joint_indices)

    def _get_observations(self) -> dict:
        # self.robot_pos = self.robot.data.root_pos_w
        rgb_data = self._tiled_camera.data.output.get("rgb")
        if rgb_data is not None:
            self.rgb_buffer.copy_(rgb_data)
            del rgb_data
            torch.div(self.rgb_buffer, 255.0, out=self.rgb_buffer)
            torch.mean(self.rgb_buffer, dim=(1, 2), keepdim=True, out=self.mean_buffer)
            torch.sub(self.rgb_buffer, self.mean_buffer, out=self.obs_buffer)
            observations = {"policy": self.obs_buffer.clone()}
        return observations

    def _compute_face_discovery_reward(self):
        self.face_rewards.zero_()
        face_id_data = self._raycaster_camera.data.output.get("face_ids")
        if face_id_data is None:
           return self.face_rewards
        self.face_ids_buffer.copy_(face_id_data)
        del face_id_data
        batch_size = self.num_envs
        height, width = self.face_ids_buffer.shape[1], self.face_ids_buffer.shape[2]
        self.face_flat_buffer.copy_(self.face_ids_buffer.view(batch_size, -1))
        torch.logical_and(
            self.face_flat_buffer >= 0,
            self.face_flat_buffer < self.max_face_ids,
            out=self.valid_mask_buffer
        )
        for env_idx in range(batch_size):
            valid_ids = self.face_flat_buffer[env_idx][self.valid_mask_buffer[env_idx]]
            if valid_ids.numel() > 0:
                unique_ids = torch.unique(valid_ids)
                already_discovered_mask = self.discovered_faces_buffer[env_idx, unique_ids]
                newly_discovered_ids = unique_ids[~already_discovered_mask]
                if newly_discovered_ids.numel() > 0:
                    self.newly_discovered_count[env_idx] = newly_discovered_ids.numel()
                    self.discovered_faces_buffer[env_idx, newly_discovered_ids] = True
    
        # OPTIMIZATION 8: Batch reward calculation
        self.face_rewards.copy_(self.newly_discovered_count.float())
        return self.face_rewards
       
    def _get_distance_reward(self) -> float:
        """NEW: Calculate distance-based reward to uninspected objects."""
        robot_pos = self.robot_pos[:, :2]  # x, y position
        distance_reward = 0.0
        distance = torch.norm(robot_pos - self.objective_position[:2])
        
        #consider only giving a reward when within a certain distance
        if distance > 1.0:
            distance = 1/(0.0001 + distance)  # Inverse distance for reward
            distance_reward = self.cfg.distance_reward_scale * distance.item()
        return distance_reward
    
    def _get_rewards(self) -> torch.Tensor:
        
        # return  torch.ones(self.num_envs, device=self.device)
        # return self.get_semantic_reward()
        face_discovery_reward = self._compute_face_discovery_reward()
        
        # Get reward based on distance to a target objective
        distance_reward = self._get_distance_reward()
        
        # Combine the rewards
        total_reward = self.cfg.forklift_reward_scale * face_discovery_reward + distance_reward
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.robot_pos = self.robot.data.root_pos_w
        
        # Check for timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Check if robot is too far from origin
        distance_from_origin = torch.norm(self.robot_pos[:, :2], dim=1)  # x, y distance
        out_of_bounds = distance_from_origin > self.cfg.max_robot_distance
        return out_of_bounds, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if env_ids:
            self.discovered_faces_buffer[env_ids] = False
            self.total_faces_discovered[env_ids] = 0
        super()._reset_idx(env_ids)

        # Sample random positions within specified range
        num_resets = len(env_ids)
        # Set FIXED robot position instead of random sampling
        new_pos = torch.zeros((num_resets, 3), device=self.device)
        # new_pos[:, 0] = 0.0  # Fixed X position
        # new_pos[:, 1] = 0.0  # Fixed Y position  
        new_pos[:, 0] = 0  # Fixed X position
        new_pos[:, 1] = 0.0  # Fixed Y position  
        new_pos[:, 2] = 0.01  # Fixed Z position (adjust height as needed)
        
        # Set FIXED robot velocity (usually zero for consistent start)
        new_vel = torch.zeros((num_resets, 3), device=self.device)
        new_vel[:, 0] = 0.0  # Fixed X velocity
        new_vel[:, 1] = 0.0  # Fixed Y velocity
        new_vel[:, 2] = 0.0  # Fixed Z velocity
        
        # Set default orientation (no rotation)
        new_quat = torch.zeros((num_resets, 4), device=self.device)
        #[W, X, Y, Z] format for quaternion
        # For a 45-degree rotation around the Z-axis, we can use:
        new_quat[:, 0] = 0.7071  # w
        new_quat[:, 1] = 0.0  # x
        new_quat[:, 2] = 0.0  # y
        new_quat[:, 3] = 0.7071  # z
        
        # Combine into root state
        new_root_state = torch.cat([new_pos, new_quat, new_vel, torch.zeros((num_resets, 3), device=self.device)], dim=-1)
        
        # Add environment origins
        new_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Reset joint positions and velocities to default
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        
        # Write states to simulation
        self.robot.write_root_pose_to_sim(new_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(new_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
