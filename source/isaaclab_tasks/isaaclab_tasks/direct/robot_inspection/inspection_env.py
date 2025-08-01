# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import gymnasium as gym
import torch
from collections.abc import Sequence
import numpy as np

from datetime import datetime
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


from isaaclab.terrains import TerrainImporter
from isaaclab.sensors import TiledCamera, RayCasterCamera, Camera
import isaacsim.core.utils.stage as stage_utils
# from semantic_manager import SemanticManager, add_semantic_tags_from_config
from .inspection_cfg import Isaac3dinspectionEnvCfg
import wandb

try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics
import omni.usd
from pxr import UsdGeom, Gf
#View logs

debug = False
use_wandb = not debug


class Isaac3dinspectionEnv(DirectRLEnv):
    cfg: Isaac3dinspectionEnvCfg

    def __init__(self, cfg: Isaac3dinspectionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._wheel_joint_indices, self._wheel_joint_names = self.robot.find_joints(".*wheel.*")
        self.wheel_velocity_scale = self.cfg.wheel_velocity_scale

        self.robot_pos = self.robot.data.root_pos_w
        self.robot_vel = self.robot.data.root_lin_vel_w

        # self.forward_velocities = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        # self.turn_left_velocities = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=self.device)
        # self.turn_right_velocities = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device)

        self.objective_position = torch.tensor([3.18, 10.8, 0.0], device=self.device)
         
        self._setup_tensor_buffers()

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _setup_tensor_buffers(self):
        """Pre-allocate all tensors to avoid memory allocation during runtime."""

        num_envs = self.num_envs
     
        self.discovered_faces_buffer = [set() for _ in range(num_envs)]
        self.face_rewards = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        self.last_face_discovery_time = torch.zeros(self.num_envs, device=self.device)
        self.log_cache = {
            "policy_max": float('-inf'),
            "policy_min": float('inf'),
            "wheel_max": float('-inf'), 
            "wheel_min": float('inf'),
            "linear_vel_max": float('-inf'),
            "angular_vel_max": float('-inf')
        }

    def _setup_scene(self):
        #Add robot, camera and terain to the scene
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

           
        self.cfg.terrain_cfg.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain_cfg.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain_cfg)

         #spawn cube objective
        # self.cube = RigidObject(self.cfg.cube_cfg)
        # self.scene.rigid_objects["cube"] = self.cube
        self.scene.clone_environments(copy_from_source=False)

        if self.cfg.use_camera_obs:
            self._obs_camera = Camera(self.cfg.observation_camera)
            self.scene.sensors["camera"] = self._obs_camera

            self._inspection_camera = Camera(self.cfg.inspection_camera)
            self.scene.sensors["inspection_camera"] = self._inspection_camera

        self._raycaster_camera = RayCasterCamera(self.cfg.raycaster_camera_cfg)
        self.scene.sensors["raycaster_camera"] = self._raycaster_camera
        # clone and replicate
    
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

 
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Diffrential Drive
        # print(f"[INFO]  Action bounds - Min: {actions.min().item():.3f}, Max: {actions.max().item():.3f}")
        self.log_cache["policy_max"] = max(self.log_cache["policy_max"], actions.max().item())
        self.log_cache["policy_min"] = min(self.log_cache["policy_min"], actions.min().item())
    
        linear_velocity = actions[:, 0] * self.cfg.max_linear_velocity  # Forward/Backward command
        angular_velocity = actions[:, 1] * self.cfg.max_angular_velocity  # Left/Right turn command

        self.log_cache["linear_vel_max"] = max(self.log_cache["linear_vel_max"], torch.abs(linear_velocity).max().item())
        self.log_cache["angular_vel_max"] = max(self.log_cache["angular_vel_max"], torch.abs(angular_velocity).max().item())

        left_wheel_velocity = (linear_velocity - (angular_velocity * self.cfg.wheel_seperation / 2)) / self.cfg.wheel_radius
        right_wheel_velocity = (linear_velocity + (angular_velocity * self.cfg.wheel_seperation / 2)) / self.cfg.wheel_radius

        # Clamp wheel velocities to avoid exceeding max limits
        left_wheel_velocity = torch.clamp(left_wheel_velocity, -self.cfg.max_wheel_velocity, self.cfg.max_wheel_velocity)
        right_wheel_velocity = torch.clamp(right_wheel_velocity, -self.cfg.max_wheel_velocity, self.cfg.max_wheel_velocity)
        # if debug:
        #     print(f"Linear Velocity: {linear_velocity}, Angular Velocity: {angular_velocity}\n")
        #     print(f"Left Wheel Velocities: {left_wheel_velocity}, Right Wheel Velocities: {right_wheel_velocity}\n")

        self.wheel_commands = torch.stack([left_wheel_velocity, right_wheel_velocity,
                                       left_wheel_velocity, right_wheel_velocity], dim=1)
        self.log_cache["wheel_max"] = max(self.log_cache["wheel_max"], self.wheel_commands.max().item())
        self.log_cache["wheel_min"] = min(self.log_cache["wheel_min"], self.wheel_commands.min().item())

        # self.actions = wheel_commands.clone()
    
    def _apply_action(self) -> None:
        

        target = self.wheel_commands * self.cfg.action_scale
        self.robot.set_joint_velocity_target(target, joint_ids=self._wheel_joint_indices)

    def _get_observations(self) -> dict:
        if self.cfg.use_camera_obs:
            data_type = "rgb" if "rgb" in self.cfg.observation_camera.data_types else "depth"
            if "rgb" in self.cfg.observation_camera.data_types:
                front_camera_data = self._obs_camera.data.output[data_type] / 255.0
                # normalize the camera data for better training results
                front_mean = torch.mean(front_camera_data, dim=(1, 2), keepdim=True)
                front_camera_data -= front_mean

                side_camera_data = self._inspection_camera.data.output["rgb"] / 255.0
                side_mean = torch.mean(side_camera_data, dim=(1, 2), keepdim=True)
                side_camera_data -= side_mean
    
                combined_camera_data = torch.cat([front_camera_data, side_camera_data], dim=-1)
            # return {"policy": combined_camera_data.clone()}
            if isinstance(self.single_observation_space["policy"], gym.spaces.Box):
                obs = combined_camera_data.clone()
            elif isinstance(self.single_observation_space["policy"], gym.spaces.Dict):
                obs = {'robot-pose': self.robot.data.root_state_w.clone(),
                        "cameras": combined_camera_data.clone()}
            return {"policy": obs}
        else:
            robot_state = self.robot.data.root_state_w.clone()
            obs = robot_state  # x, y, z, qx, qy, qz, qw
            return {"policy": obs}

    def _compute_face_discovery_reward(self):
        self.face_rewards.zero_()
        face_id_data = self._raycaster_camera.data.output.get("face_ids")

        if face_id_data is None:
           return self.face_rewards
        
        face_ids_np = face_id_data.cpu().numpy()
        current_step = self.episode_length_buf

        for env_idx in range(self.num_envs):
            # if current_step[env_idx] - self.last_face_discovery_time[env_idx] < self.cfg.min_discovery_interval:
            #     continue
            env_faces = face_ids_np[env_idx].flatten()

            valid_faces = env_faces[env_faces >= 0]
            if len(valid_faces) == 0:
                continue
            current_faces = set(valid_faces)
            newly_discovered_ids = current_faces - self.discovered_faces_buffer[env_idx]

            if newly_discovered_ids:
                self.discovered_faces_buffer[env_idx].update(newly_discovered_ids)
                self.face_rewards[env_idx] = len(newly_discovered_ids)
                # self.face_rewards[env_idx] = len(self.discovered_faces_buffer[env_idx]) / self.cfg.max_faces_to_inspect

                # self.last_face_discovery_time[env_idx] = current_step[env_idx]
                # if debug:
                #      print(f"Env {env_idx}: Discovered {len(newly_discovered_ids)} new faces. Total: {len(self.discovered_faces_buffer[env_idx])}")
        return self.face_rewards

    def _get_distance_reward(self, func_ = 'exp'):
        """NEW: Calculate distance-based reward to uninspected objects."""
        robot_pos = self.robot_pos[:, :2]  # x, y position
        distance = torch.norm(robot_pos - self.objective_position[:2])

        if func_ == 'inverse':
            # org_distance = distance.clone()
            #consider only giving a reward when within a certain distance
            if distance.item() > 1.0:
                distance = 1/(0.0001 + distance)  # Inverse distance for reward
                distance_reward = self.cfg.distance_reward_scale * distance.item()
                return distance_reward
        elif func_ == 'exp':
            activation_radius = 3.0
            k = 2.0
            distance_reward = torch.exp(-k * distance )
            final_reward = torch.where(distance > activation_radius, distance_reward, 0.0)
            return self.cfg.distance_reward_scale *  final_reward

    def _get_rewards(self) -> torch.Tensor:
        
        # return  torch.ones(self.num_envs, device=self.device)
        # return self.get_semantic_reward()
        face_discovery_reward = self._compute_face_discovery_reward()
        
        num_faces_inspected = torch.tensor([len(faces) for faces in self.discovered_faces_buffer],
                                            device=self.device,  
                                            dtype=torch.float32   # Also specify dtype for consistency
        )
        coverage_ratio = num_faces_inspected / self.cfg.max_faces_to_inspect
        success_bonus = torch.where(
            coverage_ratio >= self.cfg.inspection_threshold,
                self.cfg.coverage_reward,  # Large one-time bonus
                0.0
        )
        # # Get reward based on distance to a target objective
        distance_reward = self._get_distance_reward()
        

        #angular velocity penalty Z axis
        angular_vel = self.robot.data.root_ang_vel_w[:, 2]
        #encourage linear movement in X-Y plane
        linear_vel = self.robot.data.root_lin_vel_w[:, :2]
        forward_movement = torch.norm(linear_vel, dim=1)
        movement_reward = torch.where(forward_movement > 0.5, 1.0, 0.0)

        z_pos = self.robot.data.root_pos_w[:, 2]
        z_penalty = torch.where(z_pos > 1.0, -5.0, 0.0)
        total_reward = (self.cfg.inspection_coverage_reward_scale * face_discovery_reward
                        + success_bonus 
                        + distance_reward
                        # + self.cfg.movement_reward_scale * movement_reward
                        # + self.cfg.spin_penalty_scale * spin_penalty
                        + z_penalty)
        # if debug:
        #     print(f"Face Reward: {face_discovery_reward}, Movement Reward: {movement_reward}, Spin Penalty: {spin_penalty}, Z Penalty: {z_penalty}, Total Reward: {total_reward} \n")
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.robot_pos = self.robot.data.root_pos_w
        
        # Check for timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Check if robot is too far from origin
        # distance_from_origin = torch.norm(self.robot_pos[:, :2], dim=1)  # x, y distance
        #observed max number of faces is 6000+
        num_faces_inspected = torch.tensor([len(faces) for faces in self.discovered_faces_buffer])

        coverage_condition = (num_faces_inspected / self.cfg.max_faces_to_inspect) >= self.cfg.inspection_threshold
        # out_of_bounds = distance_from_origin > self.cfg.max_robot_distance
        return coverage_condition, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if use_wandb:
            if hasattr(self, 'log_cache'):
                wandb.log({
                    "episode_summary/policy_action_max": self.log_cache["policy_max"],
                    "episode_summary/policy_action_min": self.log_cache["policy_min"],
                    "episode_summary/wheel_velocity_max": self.log_cache["wheel_max"],
                    "episode_summary/wheel_velocity_min": self.log_cache["wheel_min"],
                    "episode_summary/linear_velocity_max": self.log_cache["linear_vel_max"],
                    "episode_summary/angular_velocity_max": self.log_cache["angular_vel_max"],
                    "episode_summary/faces_discovered": len(self.discovered_faces_buffer[0]) if self.discovered_faces_buffer else 0
                })
                
                # Reset cache for next episode
                self.log_cache = {
                    "policy_max": float('-inf'),
                    "policy_min": float('inf'),
                    "wheel_max": float('-inf'),
                    "wheel_min": float('inf'),
                    "linear_vel_max": float('-inf'),
                    "angular_vel_max": float('-inf')
                }
        super()._reset_idx(env_ids)
     
        if debug:
            print("Number of Faces Discovered in env 0 before reset:", len(self.discovered_faces_buffer[0]))
        for env_idx in env_ids:
            self.discovered_faces_buffer[env_idx].clear()
            # print(f"Reset env {env_idx}: Cleared face cache")
        
        # Sample random positions within specified range
        num_resets = len(env_ids)
        # Set FIXED robot position instead of random sampling
        new_pos = torch.zeros((num_resets, 3), device=self.device)
        new_pos[:, 0] = 0.0  # Fixed X position
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
