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
from isaaclab.sensors import TiledCamera, save_images_to_file
import isaacsim.core.utils.stage as stage_utils
import cv2
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

    def close(self):
        """Cleanup for the environment."""
        super().close()

            
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

    def detect_semantic_objects(self) -> dict:
        """
        Detect semantic objects in the current camera view.
        
        Returns:
            Dictionary with detection results for each target object
        """

        detection_results = { 
            name: {'visible': False, 'pixel_count': 0, 'coverage_percentage': 0.0, 'bbox': None}
                for name in self.target_object_names
        }

        semantic_info = self._tiled_camera.data.info.get("semantic_segmentation")
        image_data = self._tiled_camera.data.output.get("semantic_segmentation")

        if semantic_info is None or image_data is None:
            if debug:
                print("⚠️ Warning: Semantic info or image data not available this frame.")
            return detection_results

        if not semantic_info or "idToLabels" not in semantic_info:
            return detection_results  # Skip if no mapping info for this environment
        id_to_labels = semantic_info["idToLabels"]

        class_to_color = {}

        for color_str, label_dict in id_to_labels.items():
            class_name = label_dict.get("class")
            if class_name:
                # The key is a string "(R, G, B, A)", convert it to a tuple of numbers
                color_tuple = eval(color_str)
                # Create a tensor for the color, matching the image's data type (usually uint8)
                color_tensor = torch.tensor(color_tuple, dtype=torch.uint8, device=self.device)
                class_to_color[class_name] = color_tensor

        # 4. For each target object, check for its color in the image
        for object_name in self.target_object_names:
            target_color = class_to_color.get(object_name)

            # Proceed only if the target object was found in the color map
            if target_color is not None:
                # Create a boolean mask where the image pixel color matches the target color.
                # torch.all() checks for equality across the RGBA channel dimension.
                mask = torch.all(image_data == target_color, dim=-1)
                pixel_count = torch.sum(mask).item()

                if pixel_count > 0:
                    total_pixels = mask.numel()
                    coverage_percentage = (pixel_count / total_pixels) * 100

                    # Update the results with the found data
                    detection_results[object_name] = {
                        'visible': True,
                        'pixel_count': pixel_count,
                        'coverage_percentage': coverage_percentage,
                        'bbox': None
                    }

        return detection_results
    
    def _save_inspection_image(self, env_idx: int, object_name: str, coverage: float):
        """Save inspection image when coverage threshold is met."""
        if not self.cfg.save_inspection_images:
            return
            
        try:
            rgb_data = self._tiled_camera.data.output.get("rgb")
            if rgb_data is not None:
                rgb_image = rgb_data[env_idx].cpu().numpy()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"inspection_{object_name}_env{env_idx}_{coverage:.1f}pct_{timestamp}.png"
                filepath = os.path.join(self.cfg.inspection_save_dir, filename)
                
                # Convert RGBA to RGB for saving
                rgb_image_save = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)
                cv2.imwrite(filepath, rgb_image_save)
                if debug:
                    print(f"📸 Saved inspection image: {filename}")
                
        except Exception as e:
            if debug:
                print(f"Failed to save inspection image: {e}")

    
    def _setup_scene(self):
        #Add robot, camera and terain to the scene
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["camera"] = self._tiled_camera

        self.cfg.terrain_cfg.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain_cfg.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain_cfg)
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scen

        # add lights
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
            rgb_img = rgb_data/255.0  # Shape: [H, W, 4] (RGBA)
            #normalise 
            mean_tensor = torch.mean(rgb_img, dim=(1, 2), keepdim=True)
            rgb_img -= mean_tensor
            # rgb_img = torch.nn.functional.interpolate(rgb_img.permute(2,0,1).unsqueeze(0), size=(64, 64)).squeeze(0).permute(1,2,0)
            observations = {"policy": rgb_img.clone()}
        return observations
            
    
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

    def get_semantic_reward(self) -> torch.Tensor:
        """
        Calculate rewards based on semantic object detection.
        """
        rewards = torch.zeros(self.num_envs, device=self.device)
        env_ids = self.robot._ALL_INDICES
        detection_results = self.detect_semantic_objects()
        reward = 0

        if detection_results['forklift'].get('visible'):
            coverage = detection_results['forklift'].get('coverage_percentage', 0)
            coverage_reward = self.cfg.forklift_reward_scale * coverage
            reward += coverage_reward
        # Add distance-based reward to uninspected objects
        distance_reward = self._get_distance_reward()
        reward += distance_reward

        rewards[0] = reward
    
        return rewards
    
    def _get_rewards(self) -> torch.Tensor:
        
        # return  torch.ones(self.num_envs, device=self.device)
        return self.get_semantic_reward()

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
        super()._reset_idx(env_ids)

        # Sample random positions within specified range
        num_resets = len(env_ids)
        # Set FIXED robot position instead of random sampling
        new_pos = torch.zeros((num_resets, 3), device=self.device)
        # new_pos[:, 0] = 0.0  # Fixed X position
        # new_pos[:, 1] = 0.0  # Fixed Y position  
        new_pos[:, 0] = -12  # Fixed X position
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
