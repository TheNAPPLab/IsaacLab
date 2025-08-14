# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script provides teleoperation control for an Isaac Lab environment using the keyboard.

Usage:
- Press 'w' to move the robot forward.
- Press 'a' to turn the robot left.
- Press 'd' to turn the robot right.
- Release all keys to make the robot stop.
"""

import argparse
import torch
import gymnasium as gym
import keyboard  # Import the keyboard library

from isaaclab.app import AppLauncher

# Import Isaac Lab tasks and utilities
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

def main():
    """Teleoperation agent for Isaac Lab environment."""
    # --- Boilerplate setup from your original script ---
    parser = argparse.ArgumentParser(description="Teleoperation agent for Isaac Lab environments.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default="Isaac-Inspection-Camera-Direct-v0", help="Name of the task.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    # --- Instructions for the user ---
    print("\n" + "="*50)
    print("Teleoperation Controls:")
    print("  - 'w': Move Forward")
    print("  - 'a': Turn Left")
    print("  - 'd': Turn Right")
    print("  - Release keys to Stop.")
    print("  - Close the simulation window to exit.")
    print("="*50 + "\n")

    # Reset environment
    env.reset()

    # Define actions based on your environment's action space
    # Action 0: Move forward
    # Action 1: Turn right
    # Action 2: Turn left
    # Action 3: Stop (assuming this is a valid action, otherwise we send a zero-velocity command)
    MOVE_FORWARD = 0
    TURN_RIGHT = 1
    TURN_LEFT = 2
    STOP = 3 # A dedicated stop action is often cleaner if available.

    # Simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # Default action is to do nothing (stop)
            # For a discrete action space, we can use a dedicated STOP action.
            # If your environment uses continuous actions (like velocity), this would be torch.tensor([[0.0, 0.0]])
            action_id = STOP 

            # Check for keyboard inputs
            if keyboard.is_pressed('w'):
                action_id = MOVE_FORWARD
            elif keyboard.is_pressed('a'):
                action_id = TURN_LEFT
            elif keyboard.is_pressed('d'):
                action_id = TURN_RIGHT

            # Create the action tensor to send to the environment
            actions = torch.tensor([[action_id]], device=env.unwrapped.device)

            # Apply actions
            env.step(actions)

    # Close the simulator
    env.close()

if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
