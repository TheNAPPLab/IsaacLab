#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to test Cartpole environment using DirectRLEnv approach."""

"""Launch Isaac Sim Simulator first."""
local_args = {
    "headless": True,  # Run in headless mode
    "enable_cameras": True,  # Enable cameras for rendering
    "display_feed": True,  # Display camera feed

}
import argparse

from isaaclab.app import AppLauncher
import cv2
import isaacsim
import numpy as np
# add argparse arguments
parser = argparse.ArgumentParser(description="Test Cartpole environment with DirectRLEnv setup.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the simulation on.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = local_args["enable_cameras"]
args_cli.display_feed = local_args["display_feed"]
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import isaaclab_tasks 
import torch
from inspection_cfg import Isaac3dinspectionEnvCfg
from inspection_env import Isaac3dinspectionEnv

def main():
    """Main function."""
    # Create environment configuration
    env_cfg = Isaac3dinspectionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = 'cpu'


    #/World/ground/terrain/S_TrafficCone_2
    # Setup the environment using DirectRLEnv
    env = Isaac3dinspectionEnv(cfg=env_cfg)
    print(f"[INFO]: Environment Device: {env_cfg.sim.device}")
    print(f"[INFO]: Environment created with {env.num_envs} environments.")
    print(f"[INFO]: Action space: {env.action_space}")
    print(f"[INFO]: Observation space: {env.observation_space}")
    
    # Simulate physics
    count = 0
    env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # actions = env.action_space.sample()
            # actions = torch.from_numpy(actions).to(env.device)
            # actions = torch.randint(0, env.action_space.nvec[0]+1, (env.num_envs,), device=env.device)
            actions = torch.tensor([[0.5, 0.8]], device=env.unwrapped.device) 
            # Step the environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            if terminated.any() or truncated.any():
                env.reset()
                print("[INFO]: Some environments were reset.")
            if local_args["display_feed"]:
                camera_data = env.scene["camera"].data
                
                # RGB Camera Feed (simplified for single env)
                rgb_image = camera_data.output.get("rgb")
                if rgb_image is not None:
                    # Get the image from the first (and likely only) environment
                    img_np = rgb_image[0].cpu().numpy()
                    # Convert RGB to BGR for OpenCV
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    
                    # Optional: Scale up the image for better visibility
                    # Remove this line if you want original size
                    img_bgr = cv2.resize(img_bgr, (200, 200))  # 2x scale
                    
                    cv2.imshow("RGB Camera Feed", img_bgr)
                
                # Raycast Camera Feed
                raycast_data = env.scene["raycaster_camera"].data
                face_id_data = raycast_data.output.get("face_ids")
                if face_id_data is not None:
                    # Get face IDs for first environment
                    face_ids = face_id_data[0].cpu().numpy().squeeze()
                    
                    # Create colored visualization
                    face_vis = np.zeros((*face_ids.shape, 3), dtype=np.uint8)
                    
                    # Color pixels that aren't -1
                    valid_mask = face_ids != -1
                    face_vis[valid_mask] = [0, 255, 0]  # Green for detected faces
                    face_vis[~valid_mask] = [0, 0, 0]   # Black for no detection
                    
                    # Scale up for better visibility
                    face_vis_large = cv2.resize(face_vis, (640, 480), interpolation=cv2.INTER_NEAREST)
                    
                    # Add text overlay
                    cv2.putText(face_vis_large, f"Face IDs (Green=Hit)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Count valid detections
                    valid_count = np.sum(valid_mask)
                    cv2.putText(face_vis_large, f"Valid pixels: {valid_count}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow("Face ID Detection", face_vis_large)
                
                cv2.waitKey(1)
            # if local_args["display_feed"]:
            #     camera_data = env.scene["camera"].data
                
            #     # Option 1: Display all feeds in a grid (RECOMMENDED)
            #     rgb_image = camera_data.output.get("rgb")
            #     if rgb_image is not None:
            #         # Calculate grid dimensions
            #         num_envs = env.num_envs
            #         grid_cols = int(np.ceil(np.sqrt(num_envs)))
            #         grid_rows = int(np.ceil(num_envs / grid_cols))
                    
            #         height, width = rgb_image.shape[1], rgb_image.shape[2]
            #         grid_image = np.zeros((grid_rows * height, grid_cols * width, 3), dtype=np.uint8)
                    
            #         for env_idx in range(num_envs):
            #             row = env_idx // grid_cols
            #             col = env_idx % grid_cols
                        
            #             img_np = rgb_image[env_idx].cpu().numpy()
            #             img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        
            #             # Add environment label
            #             cv2.putText(img_bgr, f"Env {env_idx}", (10, 30), 
            #                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
            #             # Place in grid
            #             y_start = row * height
            #             y_end = y_start + height
            #             x_start = col * width
            #             x_end = x_start + width
                        
            #             grid_image[y_start:y_end, x_start:x_end] = img_bgr
                    
            #         cv2.imshow("All RGB Camera Feeds", grid_image)
            #     raycast_data = env.scene["raycaster_camera"].data
            #     face_id_data = raycast_data.output.get("face_ids")
            #     if face_id_data is not None:
            #         # Get face IDs for first environment
            #         face_ids = face_id_data[0].cpu().numpy().squeeze()  # Shape: (64, 64)
                    
            #         # Create colored visualization
            #         face_vis = np.zeros((640, 480, 3), dtype=np.uint8)
                    
            #         # Color pixels that aren't -1
            #         valid_mask = face_ids != -1
            #         face_vis[valid_mask] = [0, 255, 0]  # Green for detected faces
            #         face_vis[~valid_mask] = [0, 0, 0]   # Black for no detection
                    
            #         # Scale up for better visibility
            #         face_vis_large = cv2.resize(face_vis, (256, 256), interpolation=cv2.INTER_NEAREST)
                    
            #         # Add text overlay
            #         cv2.putText(face_vis_large, f"Face IDs (Green=Hit)", (10, 30), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
            #         # Count valid detections
            #         valid_count = np.sum(valid_mask)
            #         cv2.putText(face_vis_large, f"Valid pixels: {valid_count}", (10, 60), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
            #         cv2.imshow("Face ID Detection", face_vis_large)

            

            # # 2. Check if the image data exists
            # if local_args["display_feed"]:
            #     # only show the first environment's camera feed rgba image is of shape [num_envs, H, W, C]
            #     rgb_image = camera_data.output.get("rgb")
            #     if rgb_image is not None:
            #         # Get the image from the first environment
            #         img_np = rgb_image[0].cpu().numpy()
            #         # The image is in RGB format, convert it to BGR for OpenCV
            #         img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            #         cv2.imshow("Live Camera Feed", img_bgr)

                # semantic_image = camera_data.output.get("semantic_segmentation")
                # if semantic_image is not None:
                #     img_np = semantic_image[0].cpu().numpy()
                #     img_bgra = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)
                #     cv2.imshow("Live Semantic Feed", img_bgra)

            
            # Print some debug information
            if count % 50 == 0:  # Print every 50 steps
                # Print cartpole state from first environment
                policy_obs = obs["policy"][0]
                robot_x = policy_obs[0]         # Robot x position
                robot_y = policy_obs[1]         # Robot y position  
                robot_z = policy_obs[2]         # Robot z position

                # print(f"[Env 0]: Robot position: x={robot_x:.3f}, y={robot_y:.3f}, z={robot_z:.3f}")
                print(f"[Env 0]: Reward:", rewards[0].item())
                # print(f"[Env 0]: Actions: {actions[0]}")
                
                # Print some environment statistics
                num_terminated = torch.sum(terminated).item()
                if num_terminated > 0:
                    print(f"[INFO]: {num_terminated} environments terminated")
            # Update counter
            count += 1

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()