# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import deque
import math
import os
import cv2
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
from isaaclab.utils.math import transform_points, unproject_depth
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
import isaacsim.core.utils.stage as stage_utils
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
# from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_mul
from isaaclab.utils.math import quat_apply
# from semantic_manager import SemanticManager, add_semantic_tags_from_config
from .inspection_cfg import Isaac3dinspectionEnvCfg
from .map2d import Map2D
import wandb



try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics
import omni.usd
from pxr import UsdGeom, Gf
#View logs

debug = True
use_wandb = not debug


class Isaac3dinspectionEnv(DirectRLEnv):
    cfg: Isaac3dinspectionEnvCfg

    def __init__(self, cfg: Isaac3dinspectionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._wheel_joint_indices, self._wheel_joint_names = self.robot.find_joints(".*wheel.*")
        self.wheel_velocity_scale = self.cfg.wheel_velocity_scale

        self.robot_pos = self.robot.data.root_pos_w
        self.robot_vel = self.robot.data.root_lin_vel_w

        self.objective_position = torch.tensor([2.97099, 1.51382, 0.08852], device=self.device)
         
        self._setup_tensor_buffers()

        self.map2d = Map2D(
            self.cfg._map_x_lower, self.cfg._map_y_lower,
            self.cfg._map_x_upper, self.cfg._map_y_upper,
            resolution=self.cfg._map_resolution)
        self.last_map_entropy = 0.0
        self.last_visible_areas = 0.0

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _setup_tensor_buffers(self):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        """Pre-allocate all tensors to avoid memory allocation during runtime."""
     
        self.discovered_faces_buffer = set()

        self.log_cache = {}
        
        self.episode_distance_reward = 0
        self.success_buffer = deque(maxlen=60)
        self.curriculum_threshold = 0.8  # steps
        self.max_inspection_threshold = 0.99
        self.min_episodes_for_curriculum = 50
        self.difficulty_increment = 0.05
        self.success_rate = 0.0

        # Add Prim
    def _add_semantics(self):
        if self.cfg.env_parameters["semantics_type"] is not None:
            self.stage = stage_utils.get_current_stage()
            prim_path = self.cfg.env_parameters["prim_path"]
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                print(f"WARNING: Prim at {prim_path} not found")
                return False
            instance_name = f"{self.cfg.env_parameters['semantics_type']}_{self.cfg.env_parameters['semantics_name']}"
            sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
            sem.CreateSemanticTypeAttr().Set(self.cfg.env_parameters['semantics_type'])
            sem.CreateSemanticDataAttr().Set(self.cfg.env_parameters['semantics_name'])
    def _setup_scene(self):
        #Add robot, camera and terain to the scene
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

           
        self.cfg.terrain_cfg.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain_cfg.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain_cfg)

        self.scene.clone_environments(copy_from_source=False)


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

        self._add_semantics()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
    
    def _apply_action(self) -> None:
        if isinstance(self.single_action_space, gym.spaces.Box):
            self.log_cache["policy_max"] = max(self.log_cache["policy_max"], self.actions.max().item())
            self.log_cache["policy_min"] = min(self.log_cache["policy_min"], self.actions.min().item())

            linear_velocity = self.actions[:, 0] * self.cfg.max_linear_velocity  # Forward/Backward command
            angular_velocity = self.actions[:, 1] * self.cfg.max_angular_velocity  # Left/Right turn command

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
            # Scale the wheel commands
            target = self.wheel_commands * self.cfg.action_scale

        elif isinstance(self.single_action_space, gym.spaces.Discrete):
            left_wheel_velocity = torch.zeros_like(self.actions, dtype=torch.float32, device=self.device)
            right_wheel_velocity = torch.zeros_like(self.actions, dtype=torch.float32, device=self.device)
            # Action 0: Forward
            left_wheel_velocity[self.actions == 0] = self.cfg.forward_vel
            right_wheel_velocity[self.actions == 0] = self.cfg.forward_vel
           
            # Action 2: Turn Left
            left_wheel_velocity[self.actions == 1] = -self.cfg.turn_vel
            right_wheel_velocity[self.actions == 1] = self.cfg.turn_vel
            # Action 3: Turn Right

            left_wheel_velocity[self.actions == 2] = self.cfg.turn_vel
            right_wheel_velocity[self.actions == 2] = -self.cfg.turn_vel


            #  # Action 1: Backward
            # left_wheel_velocity[self.actions == 1] = -self.cfg.forward_vel
            # right_wheel_velocity[self.actions == 1] = -self.cfg.forward_vel
            

            self.wheel_commands = torch.stack([left_wheel_velocity, right_wheel_velocity,
                                       left_wheel_velocity, right_wheel_velocity], dim=1)
            target = self.wheel_commands.clone()
            target = target.view(self.num_envs, -1)
            # print(target.shape) # torch.Size([1, 4])

        # print(f"[INFO] Wheel Commands: {self.wheel_commands.clone()}")
        self.robot.set_joint_velocity_target(target, joint_ids=self._wheel_joint_indices)

    def _update_visibility_map(self):
        if "distance_to_image_plane" in self.cfg.inspection_camera.data_types:
            robot_pos = self.robot.data.root_pos_w[0, :]
            robot_quat = self.robot.data.root_quat_w[0, :]  # (w, x, y, z)

            inspection_camera_local_pos = torch.tensor(self.cfg.inspection_camera.offset.pos, device=self.device)
            inspection_camera_local_quat = torch.tensor(self.cfg.inspection_camera.offset.rot, device=self.device)

            rotated_offset = quat_apply(robot_quat, inspection_camera_local_pos)
            camera_world_pos = robot_pos + rotated_offset

            camera_world_quat = quat_mul(robot_quat, inspection_camera_local_quat)
            
            robot_quat_w = self.robot.data.root_state_w[0, 3:7]
            inspection_camera_local_quat_ros = self._inspection_camera.data.quat_w_ros[0]
            camera_world_quat = quat_mul(robot_quat_w, inspection_camera_local_quat_ros)
            pointcloud = create_pointcloud_from_depth(
                intrinsic_matrix=self._inspection_camera.data.intrinsic_matrices[0],
                depth=self._inspection_camera.data.output["distance_to_image_plane"][0],
                position=camera_world_pos, #self._obs_camera.data.pos_w[0],
                orientation= camera_world_quat,
                device=self.device,
            )
            front_points_3d_world_np = pointcloud.cpu().numpy()
            robot_pos_np = self.robot_pos[0, :3].cpu().numpy()
            self.map2d.update_visibility_map(front_points_3d_world_np, robot_pos_np)
           
    def _update_occupancy_map(self):
        if "distance_to_image_plane" in self.cfg.observation_camera.data_types:
            robot_pos = self.robot.data.root_pos_w[0, :]
            robot_quat = self.robot.data.root_quat_w[0, :]  # (w, x, y, z)

            camera_local_pos = torch.tensor(self.cfg.observation_camera.offset.pos, device=self.device)
            camera_local_quat = torch.tensor(self.cfg.observation_camera.offset.rot, device=self.device)

            rotated_offset = quat_apply(robot_quat, camera_local_pos)
            camera_world_pos = robot_pos + rotated_offset

            camera_world_quat = quat_mul(robot_quat, camera_local_quat)
            
            robot_quat_w = self.robot.data.root_state_w[0, 3:7]
            camera_local_quat_ros = self._obs_camera.data.quat_w_ros[0]
            camera_world_quat = quat_mul(robot_quat_w, camera_local_quat_ros)
            pointcloud = create_pointcloud_from_depth(
                intrinsic_matrix=self._obs_camera.data.intrinsic_matrices[0],
                depth=self._obs_camera.data.output["distance_to_image_plane"][0],
                position=camera_world_pos,
                orientation= camera_world_quat,
                device=self.device,
            )
            # cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
            # cfg.markers["hit"].radius = 0.002
            # pc_markers = VisualizationMarkers(cfg)
            # if pointcloud.size()[0] > 0:
            #     pc_markers.visualize(translations=pointcloud)
            # print(f"[INFO] Point Cloud Size: {pointcloud.size()}")
            front_points_3d_world_np = pointcloud.cpu().numpy()
            robot_pos_np = self.robot_pos[0, :3].cpu().numpy()
            self.map2d.update_occupancy_map(front_points_3d_world_np, robot_pos_np)

    def _update_map(self):
        robot_pos_np = self.robot_pos[0, :3].cpu().numpy()
        self._update_visibility_map()
        self._update_occupancy_map()
        if debug:
            self.map2d.display_map(robot_pos_np)

    def _get_observations(self) -> dict:
        self._update_map()

        if  "rgb" in self.cfg.observation_camera.data_types:
            front_camera_data = self._obs_camera.data.output[ "rgb"] / 255.0
            # normalize the camera data for better training results
            front_mean = torch.mean(front_camera_data, dim=(1, 2), keepdim=True)
            front_camera_data -= front_mean

            side_camera_data = self._inspection_camera.data.output["rgb"] / 255.0
            side_mean = torch.mean(side_camera_data, dim=(1, 2), keepdim=True)
            side_camera_data -= side_mean

            combined_camera_data = torch.cat([front_camera_data, side_camera_data], dim=-1)
        # Depth information is enough from front camera
        elif "distance_to_image_plane" in self.cfg.observation_camera.data_types:
            front_camera_data = self._obs_camera.data.output["distance_to_image_plane"]
            front_camera_data[front_camera_data == float("inf")] = 0


        occ_map_np = self.occupancy_map.get_map_single_channel()
        # Shape to (N, H, W, C)
        occ_map_tensor = torch.from_numpy(occ_map_np).to(self.device).float().unsqueeze(0).unsqueeze(-1)


        if isinstance(self.single_observation_space["policy"], gym.spaces.Box):
            obs = combined_camera_data.clone()

        elif isinstance(self.single_observation_space["policy"], gym.spaces.Dict):
            # create dummies for debugging stick all same values
            obs = {'robot-pose': self.robot.data.root_state_w.clone(),
                    "cameras": combined_camera_data.clone(),
                    "occupancy_map": occ_map_tensor# Remove batch dimension
                }
        elif isinstance(self.single_observation_space["policy"], gym.spaces.Tuple):
            
            obs = (combined_camera_data.clone(), self.robot.data.root_state_w.clone())

        return {"policy": obs}
    #    /World/ware_house_brick/_61_foam_brick/_61_foam_brick.faceVertexIndices
    def _visualise_faces(self, face_ids_to_show):
        """Visualize the discovered faces in the scene."""
        if face_ids_to_show is None:
            return
        face_ids = face_ids_to_show.cpu().numpy().squeeze()
        valid_mask = face_ids != -1


        face_vis = np.zeros((*face_ids.shape, 3), dtype=np.uint8)
        face_vis[valid_mask] = [0, 255, 0]  # Green for detected faces

        face_vis_large = cv2.resize(face_vis, (640, 480), interpolation=cv2.INTER_NEAREST)
        cv2.putText(face_vis_large, f"Face IDs (Green=Hit)", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Count valid detections
        valid_count = np.sum(valid_mask)
        cv2.putText(face_vis_large, f"Valid pixels: {valid_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
        cv2.imshow("Face ID Detection", face_vis_large)
        cv2.waitKey(1)  # Update the display

    def _compute_face_discovery_reward(self):
        face_rewards = 0.0
        occlusion_filtered_face_ids = None

        segmentation_data = self._inspection_camera.data.output.get("semantic_segmentation")
        face_id_data = self._raycaster_camera.data.output.get("face_ids")

         # Exit if either camera data is missing
        if segmentation_data is None or face_id_data is None:
            return 0.0
        
        class_ids = self._inspection_camera.data.info[0].get("semantic_segmentation").get('idToLabels')
        forklift_id = -1
        for ids in class_ids:
            _dict = class_ids[ids]
            if _dict['class'] == self.cfg.env_parameters["semantics_name"]:
                forklift_id = int(ids)
                break

        # If forklift is not in the semantic map, no reward
        if forklift_id != -1:        
            segmentation_image = segmentation_data[0]
            #get the actual segemntation mask of the forklift alone
            forklift_mask = (segmentation_image == forklift_id)
            raw_face_ids = face_id_data[0]

            # Create a tensor of -1s, then fill in the valid face IDs where the forklift is visible.
            occlusion_filtered_face_ids = torch.full_like(raw_face_ids, -1)
            occlusion_filtered_face_ids[forklift_mask] = raw_face_ids[forklift_mask]

            valid_faces = occlusion_filtered_face_ids.flatten()
            valid_faces = valid_faces[valid_faces >= 0]

            if len(valid_faces) > 0:
                current_faces = set(valid_faces.cpu().numpy())
                newly_discovered_ids = current_faces - self.discovered_faces_buffer

                if newly_discovered_ids:
                    self.discovered_faces_buffer.update(newly_discovered_ids)
                    face_rewards = len(newly_discovered_ids)
    

        if debug:
            self._visualise_faces(face_ids_to_show=occlusion_filtered_face_ids)

        return face_rewards

    def _get_distance_reward(self, func_ = 'exp'):
        """NEW: Calculate distance-based reward to uninspected objects."""
        robot_pos = self.robot_pos[:, :2]  # x, y position
        distance = torch.norm(robot_pos - self.objective_position[:2])
        print(f"[INFO] Robot Position: {robot_pos}, Objective Position: {self.objective_position[:2]}, Distance: {distance.item():.3f}")
        print(f"[INFO] Distance to objective: {distance.item():.3f}")
        reward_mask = distance < self.cfg.max_reward_distance

        if func_ == 'inverse':
            # org_distance = distance.clone()
            #consider only giving a reward when within a certain distance
            if distance.item() > 1.0:
                distance = 1/(0.0001 + distance)  # Inverse distance for reward
                distance_reward = self.cfg.distance_reward_scale * distance.item()
                return distance_reward
        elif func_ == 'exp':
            # activation_radius = 2.0
            distance_reward = torch.exp(-self.cfg.distance_reward_scale_beta * distance )
            # final_reward = torch.where(distance > activation_radius, distance_reward, 0.0)
            cliped_reward = reward_mask.float() * distance_reward
            print(f"[INFO] Distance Reward: {cliped_reward.item():.3f}")

    def compute_exploration_reward(self):
        current_map_entropy  = self.occupancy_map.calculate_entropy()
        information_gain = self.last_map_entropy - current_map_entropy
        information_gain = max(0.0, information_gain)
        self.last_map_entropy = current_map_entropy
        return information_gain
    
    def _compute_visibility_reward(self):
        current_visible_areas = self.occupancy_map.get_visibility_map()
        information_gain = self.last_visible_areas - current_visible_areas
        information_gain = max(0.0, information_gain)
        self.last_visible_areas = current_visible_areas
        return information_gain
    
    def _get_rewards(self) -> torch.Tensor:
        face_discovery_reward = self._compute_face_discovery_reward()
        exploration_reward = self.compute_exploration_reward()
        visibility_reward = self._compute_visibility_reward()
        num_faces_inspected = len(self.discovered_faces_buffer)
        coverage_ratio = num_faces_inspected / self.cfg.max_faces_to_inspect
        success_bonus = self.cfg.coverage_reward if coverage_ratio >= self.cfg.inspection_threshold else 0.0

        total_reward = (self.cfg.inspection_coverage_reward_scale * face_discovery_reward
                        + self.cfg.information_gain_reward_scale * exploration_reward
                        +success_bonus
                        + self.cfg.time_penalty
                        )
        return torch.tensor([total_reward], device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.robot_pos = self.robot.data.root_pos_w
        
        # Check for timeout
        time_out = self.episode_length_buf >= self.cfg.min_episode_length - 1
        

        num_faces_inspected = len(self.discovered_faces_buffer)

        coverage_condition = (num_faces_inspected / self.cfg.max_faces_to_inspect) >= self.cfg.inspection_threshold
    
        return coverage_condition, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        num_faces_inspected =  len(self.discovered_faces_buffer)
        coverage_ratio = num_faces_inspected / self.cfg.max_faces_to_inspect
        episode_success = coverage_ratio >= self.cfg.inspection_threshold
        self.success_buffer.append(1 if episode_success else 0)

        if len(self.success_buffer) >= self.min_episodes_for_curriculum:
            self.success_rate = sum(self.success_buffer) / len(self.success_buffer)

            if self.success_rate >= self.curriculum_threshold and self.cfg.inspection_threshold < self.max_inspection_threshold:
                new_threshold = self.cfg.inspection_threshold + self.difficulty_increment
                self.cfg.inspection_threshold = min(new_threshold, self.max_inspection_threshold)
                self.success_buffer.clear()

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if use_wandb:
            if hasattr(self, 'log_cache'):
                wandb.log({
                    "episode_summary/inspection_threshold": self.cfg.inspection_threshold,
                    "episode_summary/faces_discovered": num_faces_inspected,
                    "episode_summary/coverage_ratio": coverage_ratio * 100,
                    "episode_summary/success_rate": self.success_rate,
                    "episode_summary/episode_length": self.episode_length_buf[env_ids[0]].item(),
                    "episode_summary/last_entropy": self.last_map_entropy,
                })
        if debug:
               print(f"Episode Length: {self.episode_length_buf[env_ids].item()}")
               print(f"Number of Faces Discovered: {num_faces_inspected}")

        super()._reset_idx(env_ids)
        self.occupancy_map.reset_map()
        self.last_map_entropy = self.occupancy_map.calculate_entropy()
        self.discovered_faces_buffer.clear()
     
        if debug:
            print("Number of Faces Discovered in env 0 before reset:", len(self.discovered_faces_buffer))
            #number of steps taken in the episode
         
        
        # Sample random positions within specified range
        num_resets = len(env_ids)

        # num_positions = self.position_list.shape[0]
        # num_orientations = self.orientation_list.shape[0]
        # random_pos_indices = torch.randint(0, num_positions, (num_resets,), device=self.device)
        # random_orient_indices = torch.randint(0, num_orientations, (num_resets,), device=self.device)

        # new_pos = self.position_list[random_pos_indices]
        # new_quat = self.orientation_list[random_orient_indices]

        
        # Set FIXED robot position instead of random sampling
        new_pos = torch.zeros((num_resets, 3), device=self.device)

        #Brick Env

        # new_pos[:, 0] = -13.0  # Fixed X position
        # new_pos[:, 1] = 27.6  # Fixed Y position
        # new_pos[:, 2] = 0.01  # Fixed Z position (adjust height as needed)

        # Forklift Env
        # new_pos[:, 0] = -13.0  # Fixed X position
        # new_pos[:, 1] = 27.6  # Fixed Y position
        # new_pos[:, 2] = 0.01  # Fixed Z position (adjust height as needed)

        # next to the Goal
        # new_pos[:, 0] = -13.0  # Fixed X position
        # new_pos[:, 1] = 5.0  # Fixed Y position
        # new_pos[:, 2] = 0.01  # Fixed Z position 

        #outside wall
        # new_pos[:, 0] = -35.0  # Fixed X position
        # new_pos[:, 1] = 5.0  # Fixed Y position
        # new_pos[:, 2] = 0.01  
        
        # #Behind the shelves
        # new_pos[:, 0] = -10.0  # Fixed X position
        # new_pos[:, 1] = 27.6  # Fixed Y position
        # new_pos[:, 2] = 0.01  # Fixed Z position 
        
        # Set FIXED robot velocity (usually zero for consistent start)
        new_vel = torch.zeros((num_resets, 3), device=self.device)

        # # Set default orientation (no rotation)
        new_quat = torch.zeros((num_resets, 4), device=self.device)
        # [W, X, Y, Z] format for quaternion
        # For a 45-degree rotation around the Z-axis, we can use:
        new_quat[:, 0] = 0.7071 # w 0.7071
        new_quat[:, 1] = 0.0  # x
        new_quat[:, 2] = 0.0  # y
        new_quat[:, 3] = 0.7071 # z 0.7071

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



# self.position_list = torch.tensor([
        #                     [0, 0,  0], #x, y, z
        #                       [0, 4.17, 0],
        #                       [3,4.17, 0],
        #                       [4.3, 4.17, 0],
        #                       [5.7, 4.17, 0],
        #                       [5.7, 1.03, 0],
        #                       [6.4, -1.77, 0],
        #                       [2.7, -1.77, 0],
        #                       [-0.8, -1.77, 0]
        # ], device=self.device)
        # w, x,y,z     
        # self.orientation_list = torch.tensor([
        #     [1.0, 0.0, 0.0, 0.0],         # 0 degrees 
        #     [0.7071, 0.0, 0.0, 0.7071],  # 90 degrees
        #     [0.0, 0.0, 0.0, 1.0],         # 180 degrees
        #     [-0.7071, 0.0, 0.0, 0.7071]  # 270 degrees
        # ], device=self.device)
          


# "episode_summary/policy_action_max": self.log_cache["policy_max"],
                    # "episode_summary/policy_action_min": self.log_cache["policy_min"],
                    # "episode_summary/wheel_velocity_max": self.log_cache["wheel_max"],
                    # "episode_summary/wheel_velocity_min": self.log_cache["wheel_min"],
                    # "episode_summary/linear_velocity_max": self.log_cache["linear_vel_max"],
                    # "episode_summary/angular_velocity_max": self.log_cache["angular_vel_max"],


                                    # Reset cache for next episode
                # self.log_cache = {
                #     "policy_max": float('-inf'),
                #     "policy_min": float('inf'),
                #     "wheel_max": float('-inf'),
                #     "wheel_min": float('inf'),
                #     "linear_vel_max": float('-inf'),
                #     "angular_vel_max": float('-inf')
                # }

# # Get reward based on distance to a target objective
# distance_reward = self._get_distance_reward()
# self.episode_distance_reward += distance_reward



    #spawn cube objective
# self.cube = RigidObject(self.cfg.cube_cfg)
# self.scene.rigid_objects["cube"] = self.cube