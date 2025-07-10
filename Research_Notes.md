# File Locations and description
The source files are located in
```
source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection
```

To run a basic interactive file for debuging and visualing run
```
source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/run_direct_rl_env.py
```

# Reward Design
# First Version
Simple Reward function
$reward = -a*distance + b *Number of Segementation pixels in camera view$

Drive the reward to a goal and capture the pixels.

# Environmental Setup
I have a jackal with a RBG and Segementation Camera,
The Inpsection objective is a forklift that comes with the Warehouse Evironment
we rewrote the segemantaion name for the forklift

# Sample Image of Environment (Single Robot/ENV)
![multi Robot/Env](Images/single_robot.png)


# Sample Image of Environment (Multi Robot/ENV) STILL BROKEN CAUSE TERRAIN WONT REPLICATE!
![Single Robot/Env](Images/multi_robot.png)

# Things to add
- [ ] Different Inspection goals (Fire extinguisher, Traffic cone)
- [ ] Some Goal oriented setup, randomise the inspection goal and give as an input the image of the goal
- [ ] Randomise the starting pose of the robot as well as potentially the Goals
- [ ] Better Mesh environments to experiment on, environment is in ROWS not ideal or interesting maybe find from Fab.com or build ourselves.
- [ ] Integrate with the training PPO environments
- [ ] 3D reconstrunction setup and reward for inspection quality and coverage( Gaussian Splatting)
- [ ] Surface normal counting as coverage function ( massk image trajectory with segmantics )


